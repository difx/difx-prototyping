// common.cuh - shared definitions for the FFT + fractional-sample-correction fusion benchmark.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

// As in mpifxcorr (architecture.h generic value; IPP_2PI on IPP builds is the same double).
#define TWO_PI 6.283185307179586476925286766559
// As in mpifxcorr/src/gpumode.cu: per-frequency slots in the band-pair map.
static const int MAX_INDICIES = 10;

#define CHECK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)
#define CHECK_CUFFT(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1); } } while (0)

// a * conj(b), as mpifxcorr/src/gpumode_kernels.cuh computes it
__host__ __device__ static __inline__ cuFloatComplex cuCmulConjf(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x * b.x + a.y * b.y,
                               a.y * b.x - a.x * b.y);
}

// The shape of one "subint" of work: nant antennas (one GPUMode each in mpifxcorr), each with
// nwin FFT windows (numBufferedFFTs) of nband recorded bands. Layout matches mpifxcorr's
// complex_fringe_rotated_gpu / fftd_gpu buffers, one block per antenna:
//     [ant][window][band][fftchannels]
// nchan is recordedbandchannels: fftchannels/2 for real-sampled data (only the lower half of the
// spectrum is used), fftchannels for complex-sampled data.
struct Geometry {
    int nant, nband, nwin, fftchannels, nchan;
    __host__ __device__ size_t perAnt() const { return (size_t)nwin * nband * fftchannels; }
    __host__ __device__ size_t spectra() const { return (size_t)nant * nwin * nband; }
    __host__ __device__ size_t index(int ant, int win, int band, int k) const {
        return (size_t)ant * perAnt() + ((size_t)win * nband + band) * fftchannels + k;
    }
};

// Autocorrelation work done alongside the fractional-sample correction, as in mpifxcorr:
//   NONE  DIFX_GPU_XMAC_AUTOCORR=1 (autocorrelations in the XMAC): pure rotation
//   BAND  per-band autocorrelations accumulated with atomics, no cross-pol products
//   FULL  production default (gate off) with cross-pol autocorrelations: per-band plus the
//         cross-pol pass over each frequency's band pair
enum class AutoMode { NONE, BAND, FULL };
