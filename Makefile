# difx-prototyping: one build for every sub-project (fft-fusion, xmac, two-stage), the shared code in
# common/.  See README.md.
#
#   source <gpudifx>/setup.bash; make -j8      # CUDA and the GPU architecture from setup.bash,
#                                              # MathDx fetched from PyPI on the first build
#   make SM=80 ...                             # another GPU architecture (80: A100)
#   make xmacbench twostagebench               # only some binaries (fusebench is the slow one)
#   make SIZES="256 2048" ...                  # only some FFT lengths in fusebench's fused kernel
#   make mathdx                                # only fetch MathDx (on a login node with network
#                                              # access, before building on a node without it)
#
# The fused kernel is compiled once per FFT size, ~20-60 s each, so build in parallel (-j8).
#
# Each GPU architecture builds in its own build-sm<SM>/, objects and binaries both, so a build for
# one never touches another's: build-sm80/fusebench and build-sm120/fusebench sit side by side and
# switching between them rebuilds nothing.  fft-fusion/fusebench, xmac/xmacbench,
# two-stage/twostagebench and phase-centres/phasecentrebench are links to the last build's
# binaries, for running by hand; the run-*.sh scripts take the binaries for the GPU they find
# (common/scripts/bench-bin.sh), so jobs on different GPUs sharing this checkout never run each
# other's.
#
# Every setting can be given on the command line.  Without one:
#   CUDA_HOME  $CUDAROOT (setup.bash), else $CUDA_HOME, else the CUDA holding the nvcc on PATH,
#              else /usr/local/cuda
#   SM         the -arch=sm_NN in $NVCCFLAGS (setup.bash), else the first GPU nvidia-smi reports,
#              else 75 (RTX 2070), with a warning
#   MATHDX     MathDx (header-only, contains cuFFTDx): the directory holding include/ and
#              external/.  Default: mathdx-$(MATHDX_VERSION)/ here, downloaded from PyPI (the
#              nvidia-mathdx wheel, checked against its sha256) and unpacked on the first build.
#              NVIDIA's licence does not allow us to commit it to this repository.

MATHDX_VERSION := 25.6.0
MATHDX_WHEEL   := nvidia_mathdx-$(MATHDX_VERSION)-py3-none-any.whl
MATHDX_URL     := https://files.pythonhosted.org/packages/20/1a/a418b8c1adc58abd87fd69414c19883af5c1b10514e3dbfcc27cde831b13/$(MATHDX_WHEEL)
MATHDX_SHA256  := 22e6ad5d0d005f836be5cbd14e836cf2e9ea42c82deb602707246ce8198eaa96
MATHDX_LOCAL   := mathdx-$(MATHDX_VERSION)
MATHDX_STAMP   := $(MATHDX_LOCAL)/.unpacked

# --- CUDA
ifneq ($(origin CUDA_HOME),command line)
  ifneq ($(strip $(CUDAROOT)),)
    CUDA_HOME := $(CUDAROOT)
  endif
endif
ifeq ($(strip $(CUDA_HOME)),)
  CUDA_HOME := $(patsubst %/bin/nvcc,%,$(shell command -v nvcc 2>/dev/null))
endif
ifeq ($(strip $(CUDA_HOME)),)
  CUDA_HOME := /usr/local/cuda
endif
NVCC ?= $(CUDA_HOME)/bin/nvcc

# --- GPU architecture.  NVCCFLAGS from the environment is only read for its -arch: the compiler
# flags for fusebench are FLAGS below.
ifeq ($(origin SM),undefined)
  SM := $(patsubst -arch=sm_%,%,$(patsubst --gpu-architecture=sm_%,%,$(filter -arch=sm_% --gpu-architecture=sm_%,$(NVCCFLAGS))))
  SM := $(firstword $(SM))
  SM_FROM := NVCCFLAGS
  ifeq ($(SM),)
    # only a compute capability: without a GPU (a login node) nvidia-smi prints an error message
    SM := $(subst .,,$(firstword $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                                          | grep -E '^[0-9]+\.[0-9]+$$')))
    SM_FROM := nvidia-smi
  endif
  ifeq ($(SM),)
    SM := 75
    SM_FROM := default
    $(warning No SM given, no -arch=sm_NN in NVCCFLAGS and no GPU from nvidia-smi: building for sm_75 (give SM=80 for an A100))
  endif
else
  SM_FROM := $(origin SM)
endif

# --- MathDx
ifeq ($(origin MATHDX),undefined)
  MATHDX     := $(MATHDX_LOCAL)/nvidia/mathdx
  MATHDX_DEP := $(MATHDX_STAMP)
endif

# FFT lengths (2 x channels for real data, channels for complex) built into the fused kernel
SIZES     ?= 64 128 256 512 1024 2048 4096 8192 16384

C := common
COMMON_H  := $(C)/pipelines.cuh $(C)/stages.cuh $(C)/common.cuh $(C)/decode.cuh $(C)/gpudecode_copy.cuh $(C)/xmac.cuh \
             $(C)/benchutil.cuh
FLAGS     := -std=c++17 -O3 -arch=sm_$(SM) -lineinfo -I$(CUDA_HOME)/include -I$(C) \
             -Xcompiler -Wno-deprecated-declarations
DXFLAGS    = -I$(MATHDX)/include -I$(MATHDX)/external/cutlass/include -DFUSE_SM=$(patsubst %a,%,$(SM))0
BUILD     := build-sm$(SM)
FUSED_OBJ := $(foreach n,$(SIZES),$(BUILD)/fused_$(n).o)

ifeq ($(filter clean mathdx,$(MAKECMDGOALS)),)
  $(info difx-prototyping: sm_$(SM) (from $(SM_FROM)), CUDA $(CUDA_HOME), MathDx $(MATHDX))
endif

all: fusebench xmacbench twostagebench phasecentrebench

# the links to this architecture's binaries in each sub-project (always redone: the last build's
# architecture wins)
.PHONY: all fusebench xmacbench twostagebench phasecentrebench mathdx clean
fusebench: $(BUILD)/fusebench
	ln -sfn ../$< fft-fusion/$@
xmacbench: $(BUILD)/xmacbench
	ln -sfn ../$< xmac/$@
twostagebench: $(BUILD)/twostagebench
	ln -sfn ../$< two-stage/$@
phasecentrebench: $(BUILD)/phasecentrebench
	ln -sfn ../$< phase-centres/$@

# ---- fft-fusion: the station kernel (common/fused_inst.cu, once per FFT size) against gpudifx
# today (common/unfused.cu), and the whole chain with the XMAC
$(BUILD)/fusebench: $(BUILD)/fusebench.o $(BUILD)/unfused.o $(BUILD)/xmac.o $(FUSED_OBJ)
	$(NVCC) $(FLAGS) $^ -o $@ -L$(CUDA_HOME)/lib64 -lcufft

$(BUILD)/fusebench.o: fft-fusion/fusebench.cu $(COMMON_H) | $(BUILD) $(MATHDX_DEP)
	$(NVCC) $(FLAGS) -I$(MATHDX)/include -c $< -o $@

$(BUILD)/fused_%.o: $(C)/fused_inst.cu $(COMMON_H) | $(BUILD) $(MATHDX_DEP)
	$(NVCC) $(FLAGS) $(DXFLAGS) -DFUSE_N=$* -c $< -o $@

$(BUILD)/unfused.o: $(C)/unfused.cu $(COMMON_H) | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@

# ---- xmac: the XMAC stage on its own (common/xmac.cuh)
$(BUILD)/xmacbench: $(BUILD)/xmacbench.o $(BUILD)/xmac.o
	$(NVCC) $(FLAGS) $^ -o $@

$(BUILD)/xmacbench.o: xmac/xmacbench.cu $(C)/xmac.cuh $(C)/common.cuh $(C)/benchutil.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@

$(BUILD)/xmac.o: $(C)/xmac.cu $(C)/xmac.cuh $(C)/xmac_copy.cuh $(C)/common.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@

# ---- two-stage: the PFB stage and the fine-FFT stage, compiled for their FFT sizes
# (common/twostage1.cu, twostage2.cu), and the benchmark
$(BUILD)/twostagebench: $(BUILD)/twostagebench.o $(BUILD)/twostage1.o $(BUILD)/twostage2.o $(BUILD)/unfused.o
	$(NVCC) $(FLAGS) $^ -o $@ -L$(CUDA_HOME)/lib64 -lcufft

$(BUILD)/twostagebench.o: two-stage/twostagebench.cu $(C)/twostage.cuh $(COMMON_H) | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@

$(BUILD)/twostage%.o: $(C)/twostage%.cu $(C)/twostage.cuh $(COMMON_H) | $(BUILD) $(MATHDX_DEP)
	$(NVCC) $(FLAGS) $(DXFLAGS) -c $< -o $@

# ---- phase-centres: the shift to many phase centres after the staged XMAC
$(BUILD)/phasecentrebench: $(BUILD)/phasecentrebench.o $(BUILD)/xmac.o
	$(NVCC) $(FLAGS) $^ -o $@

$(BUILD)/phasecentrebench.o: phase-centres/phasecentrebench.cu $(C)/xmac.cuh $(C)/common.cuh $(C)/benchutil.cuh | $(BUILD)
	$(NVCC) $(FLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $@

mathdx: $(MATHDX_STAMP)

# Unpacked in place; the stamp is written last, so an interrupted fetch is simply redone.
$(MATHDX_STAMP):
	@echo "Fetching MathDx $(MATHDX_VERSION) (cuFFTDx) from PyPI into $(MATHDX_LOCAL)/"
	mkdir -p $(MATHDX_LOCAL)
	curl -fL --retry 3 -o $(MATHDX_LOCAL)/$(MATHDX_WHEEL) $(MATHDX_URL)
	cd $(MATHDX_LOCAL) && echo "$(MATHDX_SHA256)  $(MATHDX_WHEEL)" | sha256sum -c -
	cd $(MATHDX_LOCAL) && python3 -I -m zipfile -e $(MATHDX_WHEEL) .
	test -f $(MATHDX_LOCAL)/nvidia/mathdx/include/cufftdx.hpp
	rm -f $(MATHDX_LOCAL)/$(MATHDX_WHEEL)
	touch $@

clean:
	rm -rf build-sm* fft-fusion/fusebench xmac/xmacbench two-stage/twostagebench phase-centres/phasecentrebench

