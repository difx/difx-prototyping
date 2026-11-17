// xmac_copy.cuh - gpudifx's cross-multiply-accumulate (XMAC) kernel, copied from
// mpifxcorr/src/gpucore.cu (difx/difx branch gpudifx, 40cd7afb4): the kernel and the complex
// atomic add it uses, verbatim, and (xmacGpudifxLaunch, below the copy) the launch geometry
// GPUCore::processgpudata gives it, reproduced line for line.  cuCmulConjf is common.cuh's (the
// same expression as mpifxcorr/src/gpumode_kernels.cuh).  If the production code changes, copy
// it again.
#pragma once
#include <cuComplex.h>
#include "common.cuh"

// ---- copied from mpifxcorr/src/gpucore.cu ----------------------------------------------------
// Adapted from https://forums.developer.nvidia.com/t/atomic-add-for-complex-numbers/39757
// todo: deduplicate this function
__device__ void atomicAddFloatComplex1(cuFloatComplex* a, cuFloatComplex b){
    // transform the addresses of real and imag. parts to double pointers
    float *x = (float*)a;
    float *y = x+1;
    //use atomicAdd for double variables
    atomicAdd(x, cuCrealf(b));
    atomicAdd(y, cuCimagf(b));
}

/**
 * @brief Fused Cross-Multiply-Accumulate (XMAC) and Frequency Averaging Kernel
 * * Unlike the CPU `threadcrosscorrs` loop which chunks data by `xmacstridelen`
 * for cache locality, this kernel relies on coalesced memory access. Threads
 * are mapped to specific, final *output* channels. Each thread traverses the
 * time-domain (FFT buffers) and the fine-frequency domain (channels to average),
 * collapsing them into a single point.
 *
 * The output averaged-channel dimension is spread across BOTH the block
 * (threadIdx.x) and the grid (blockIdx.z) so that the launch produces enough
 * blocks to fill the device, rather than the numbaselines*numpol (~handful)
 * blocks of the previous grid-stride design.
 *
 * When baselines*pols*channels alone cannot fill the device (the geodesy
 * regime: 1-6 baselines, often 1 pol, modest channel counts), the FFT/time
 * dimension is also split across blockIdx.z: the host divides the
 * numBufferedFFTs integration into chunks of fftsPerChunk FFTs, and each block
 * accumulates a partial sum over its chunk. With a single chunk every
 * (baseline, pol, avg_chan) maps to a unique slot in the pre-zeroed results
 * buffer, so the thread writes its total with a plain store; with multiple
 * chunks the partial sums are combined with an atomic add into the pre-zeroed
 * slot (the extra atomic traffic is one add per chunk, only taken when the
 * grid would otherwise be too small to occupy the device anyway).
 */
__global__ void gpu_fuse_xmac_and_average(
        const cuFloatComplex* const * const gpuM1Freqs,
        const cuFloatComplex* const * const gpuM2Freqs,
        const bool* const * const gpuM1Valid,
        const bool* const * const gpuM2Valid,
        const int* const stream1BandIndexes,
        const int* const stream2BandIndexes,
        const int* const coreResultOffsets,
        cuFloatComplex* const results_gpu,
        int numbaselines,
        int numPolarisationProducts,
        int numBufferedFFTs,
        int fftsPerChunk,
        int num_averaged_channels,
        int channelstoaverage,
        const int* const stream1BandStride,
        const int* const stream1WindowStride,
        const int* const stream2BandStride,
        const int* const stream2WindowStride
) {
    // gridDim.x  = numbaselines
    // gridDim.y  = numPolarisationProducts
    // gridDim.z  = numChanBlocks * numFftChunks, where
    //              numChanBlocks = ceil(num_averaged_channels / blockDim.x)
    //              numFftChunks  = ceil(numBufferedFFTs / fftsPerChunk)
    // blockDim.x = output averaged channels processed per block (<= 256)

    int baseline = blockIdx.x;
    int pol = blockIdx.y;

    // Unpack the combined (FFT chunk, channel block) z index.
    int numChanBlocks = (num_averaged_channels + blockDim.x - 1) / blockDim.x;
    int fftChunk = blockIdx.z / numChanBlocks;
    int chanBlock = blockIdx.z % numChanBlocks;

    // Each thread owns exactly one output averaged channel within its FFT chunk.
    int avg_chan = chanBlock * blockDim.x + threadIdx.x;
    if (avg_chan >= num_averaged_channels) return;

    // The range of buffered FFTs this block integrates over.
    int fft_begin = fftChunk * fftsPerChunk;
    int fft_end = fft_begin + fftsPerChunk;
    if (fft_end > numBufferedFFTs) fft_end = numBufferedFFTs;

    // 1. Resolve Band Indexes
    // We pre-flattened the 2D [baseline][pol] config arrays on the host.
    int b1 = stream1BandIndexes[baseline * numPolarisationProducts + pol];
    int b2 = stream2BandIndexes[baseline * numPolarisationProducts + pol];

    // If the baseline doesn't participate in this frequency, it was flagged as -1.
    if (b1 < 0 || b2 < 0) return;

    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

    const bool* valid1 = gpuM1Valid[baseline];
    const bool* valid2 = gpuM2Valid[baseline];

    // 2. Time Integration
    // Summation over this block's chunk of the FFTs residing in the GPU buffer.
    for (int fft = fft_begin; fft < fft_end; fft++) {

        // Skip FFT windows that were invalid on either datastream: the mode
        // kernels leave their spectra stale/uninitialised in the FFT buffers.
        // The CPU path zeroes such spectra, so skipping matches it exactly
        // (zero contribution, and the data weight already excludes them).
        if (!valid1[fft] || !valid2[fft]) continue;

        // Calculate the absolute starting index for this specific FFT and Band,
        // using each stream's OWN buffer strides (they differ on mixed
        // real x complex baselines).
        int base_idx1 = fft * stream1WindowStride[baseline] + b1 * stream1BandStride[baseline];
        int base_idx2 = fft * stream2WindowStride[baseline] + b2 * stream2BandStride[baseline];

        // 3. Frequency Integration
        // Collapse 'channelstoaverage' fine channels into this single output channel.
        for (int c = 0; c < channelstoaverage; c++) {
            int fine_chan = avg_chan * channelstoaverage + c;

            cuFloatComplex v1 = gpuM1Freqs[baseline][base_idx1 + fine_chan];
            cuFloatComplex v2 = gpuM2Freqs[baseline][base_idx2 + fine_chan];

            // Cross-multiply: V1 * conj(V2). Both operands come from the
            // rotated spectra and the conjugation happens here, in the
            // multiply, at no instruction cost - see cuCmulConjf.
            sum = cuCaddf(sum, cuCmulConjf(v1, v2));
        }
    }

    // 4. Apply Scaling
    // Time integration in DiFX is a pure SUM, but frequency integration is a MEAN.
    // We multiply by the reciprocal to avoid expensive division operations.
    float scale = 1.0f / (float)channelstoaverage;
    sum.x *= scale;
    sum.y *= scale;

    // 5. Direct Output Mapping
    // We completely bypass the CPU's intermediate `threadcrosscorrs` array.
    // One offset per (baseline, polproduct): for a cross baseline these are the
    // contiguous [Baseline][Polarisation][Channel] run the results buffer has
    // always had, but an autocorrelation baseline's products live in two
    // separate sub-blocks of the autocorrelation region (all parallel hands,
    // then all cross-pol), so the placement has to be per product. -1 means the
    // product does not exist for this baseline at this frequency.
    int base_offset = coreResultOffsets[baseline * numPolarisationProducts + pol];
    if (base_offset < 0) return;
    int final_index = base_offset + avg_chan;

    if (fftsPerChunk >= numBufferedFFTs) {
        // Single FFT chunk: this (baseline, pol, avg_chan) triple is unique
        // within the launch and the results buffer was pre-zeroed, so a plain
        // store is correct and avoids the serialisation of an atomic add.
        results_gpu[final_index] = sum;
    } else {
        // Multiple FFT chunks contribute partial sums to the same pre-zeroed
        // output slot; combine them atomically.
        atomicAddFloatComplex1(&results_gpu[final_index], sum);
    }
}
// ---- end of the copy -------------------------------------------------------------------------

// GPUCore::processgpudata's launch of gpu_fuse_xmac_and_average for one frequency (gpucore.cu,
// "FUSED XMAC AND AVERAGE SETUP"): the block size, the channel blocks, and the split of the
// integration over the FFTs when the baselines, products and channels alone would not fill the
// GPU.
struct XmacLaunch { dim3 blocks, threads; int fftsPerChunk; };
inline XmacLaunch xmacGpudifxLaunch(int numBaselines, int numPolarisationProducts, int num_averaged_channels,
                                    int numBufferedFFTs, int cudaMultiProcessorCount) {
    int threadsPerBlock = 256;
    if (num_averaged_channels < threadsPerBlock)
        threadsPerBlock = ((num_averaged_channels + 31) / 32) * 32;
    int numChanBlocks = (num_averaged_channels + threadsPerBlock - 1) / threadsPerBlock;
    const long long launchThreads = (long long)numBaselines * numPolarisationProducts * numChanBlocks * threadsPerBlock;
    const long long targetThreads = (long long)cudaMultiProcessorCount * 2048;
    int numFftChunks = 1;
    if (launchThreads < targetThreads) {
        numFftChunks = (int)((targetThreads + launchThreads - 1) / launchThreads);
        if (numFftChunks > numBufferedFFTs)
            numFftChunks = numBufferedFFTs;
        if (numFftChunks > 65535 / numChanBlocks)
            numFftChunks = 65535 / numChanBlocks;
        if (numFftChunks < 1)
            numFftChunks = 1;
    }
    int fftsPerChunk = (numBufferedFFTs + numFftChunks - 1) / numFftChunks;
    if (fftsPerChunk < 1)
        fftsPerChunk = 1;
    numFftChunks = (numBufferedFFTs + fftsPerChunk - 1) / fftsPerChunk;
    if (numFftChunks < 1)
        numFftChunks = 1;
    return {dim3(numBaselines, numPolarisationProducts, numChanBlocks * numFftChunks), dim3(threadsPerBlock), fftsPerChunk};
}
