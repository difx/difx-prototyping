// pipelines.cuh - the two ways of running pre-FFT stage + FFT + fractional-sample correction.
#pragma once
#include <map>
#include <vector>
#include "stages.cuh"

// Fills p.fixedPhase for FracRotateFixed: one thread per (antenna, window, band).
void launchFixedPhase(const Geometry& g, const FracParams& p, cudaStream_t st);

// ---- unfused: mpifxcorr's structure (unfused.cu) ---------------------------------------------
// One cuFFT plan per antenna, batch nwin * nband, then the rotation kernel per antenna on its own
// slice, all on one stream (one GPUMode per antenna sharing the Core's stream).  If the pre-FFT
// stage is anything but an FP32 buffer per window, it is first materialised as one (an extra
// kernel and an extra write and read of the FFT input).  batched=true uses one plan over all
// antennas (the rotation kernel is per antenna either way, as it takes per-antenna pointers).
struct Unfused {
    Geometry g{};
    bool batched = false;
    int maxThreadsPerBlock = 1024;
    std::vector<cufftHandle> plans;
    cuFloatComplex* staging = nullptr;    // FP32 FFT input when the pre-FFT stage is not direct
    // packed input: DOUBLE runs gpudifx's decode + fringe kernel verbatim (FP64 phases), which
    // skips the windows validSamples ([ant][win]) flags invalid; FIXED materialises PreDecode
    Rotator decodeRotator = Rotator::FIXED;
    const bool* validSamples = nullptr;

    void init(const Geometry& geom, bool batchedPlan, bool needStaging);
    void destroy();
    // the three stages, separately so they can be timed separately
    void runPre(const PreVariant& pre, cudaStream_t st);
    void runFFT(const PreVariant& pre, cuFloatComplex* out, cudaStream_t st);
    // DOUBLE: mpifxcorr's kernel (verbatim copy); FIXED: FracRotateFixed in the same launch
    // geometry, after launchFixedPhase
    void runFrac(const FracParams& p, AutoMode mode, Rotator rot, cuFloatComplex* out, cudaStream_t st);
    void run(const PreVariant& pre, const FracParams& p, AutoMode mode, Rotator rot, cuFloatComplex* out,
             cudaStream_t st) {
        runPre(pre, st);
        runFFT(pre, out, st);
        runFrac(p, mode, rot, out, st);
    }
};

// ---- fused: one cuFFTDx kernel (fused_inst.cu, compiled once per FFT size) -------------------
// FP32 FFT whatever the input precision; FP32 output.  Writes only the nchan channels the
// correlator uses; for real-sampled data the unused upper half of each spectrum is not written
// (cuFFT writes it in the unfused pipeline; mpifxcorr's XMAC never reads it).
struct FusedVariant {
    unsigned fpb;          // FFTs per block
    unsigned ept;          // elements per thread
    unsigned threads;      // threads per block
    unsigned smem;         // dynamic shared memory, bytes
    bool suggested;        // cuFFTDx's suggested FFTs per block
    unsigned regcap;       // 0, or the threads per SM the registers are capped for (__launch_bounds__)
    // all antennas in [ant0, ant0 + nant) in one launch (after launchFixedPhase); returns true if
    // the kernel also formed the cross-pol products (p.crossPolInBlock, even FFTs per block)
    bool (*launch)(const Geometry& g, const PreVariant& pre, const FracParams& p, bool doAutocorr,
                   cuFloatComplex* out, int ant0, int nant, cudaStream_t st);
};
// registers per thread of the last fused kernel launched (for the report)
extern int fusedLastRegisters;
std::map<int, std::vector<FusedVariant>>& fusedRegistry();   // FFT size -> variants

// The cross-pol autocorrelation pass, when the fused kernel cannot form the products itself.
void launchCrossPol(const Geometry& g, const FracParams& p, const cuFloatComplex* out, int maxThreads,
                    cudaStream_t st);
