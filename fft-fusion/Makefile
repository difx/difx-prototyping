# fusebench - cuFFT + fractional-sample correction (mpifxcorr today) vs one fused cuFFTDx kernel.
# See README.md.
#
#   make MATHDX=<MathDx root>                   # build for the RTX 2070 (sm_75)
#   make MATHDX=<MathDx root> SM=80             # A100
#   make -j8 ...                                # the fused kernel is compiled once per FFT
#                                               # size, ~20-60 s each, so build in parallel
#   make SIZES="256 2048" ...                   # only some FFT lengths
#
# MathDx (which contains cuFFTDx) is header-only: either NVIDIA's tarball
# (https://developer.nvidia.com/cufftdx-downloads) or the nvidia-mathdx wheel on PyPI, unpacked
# (its nvidia/mathdx directory). MATHDX is the directory holding include/ and external/.

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
MATHDX    ?= $(error set MATHDX to the MathDx directory that holds include/cufftdx.hpp)
SM        ?= 75
# FFT lengths (2 x channels for real data, channels for complex) built into the fused kernel
SIZES     ?= 64 128 256 512 1024 2048 4096 8192 16384

NVCCFLAGS := -std=c++17 -O3 -arch=sm_$(SM) -lineinfo -I$(CUDA_HOME)/include \
             -Xcompiler -Wno-deprecated-declarations
DXFLAGS    = -I$(MATHDX)/include -I$(MATHDX)/external/cutlass/include -DFUSE_SM=$(SM)0
BUILD     := build-sm$(SM)
FUSED_OBJ := $(foreach n,$(SIZES),$(BUILD)/fused_$(n).o)

fusebench: $(BUILD)/fusebench.o $(BUILD)/unfused.o $(FUSED_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -L$(CUDA_HOME)/lib64 -lcufft

$(BUILD)/fusebench.o: fusebench.cu pipelines.cuh stages.cuh common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -I$(MATHDX)/include -c $< -o $@

$(BUILD)/unfused.o: unfused.cu pipelines.cuh stages.cuh common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD)/fused_%.o: fused_inst.cu pipelines.cuh stages.cuh common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(DXFLAGS) -DFUSE_N=$* -c $< -o $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf build-sm* fusebench

.PHONY: clean
