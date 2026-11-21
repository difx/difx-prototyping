// unfused.cu - the current mpifxcorr approach: cuFFT, then the fractional-sample rotation kernel.
//
// gpu_resultsrotatorMultiply below is copied verbatim from mpifxcorr/src/gpumode.cu (gpudifx
// e2e1e58ab), as are atomicAddFloatComplex and the launch geometry of
// GPUMode::fractionalRotation. If the production kernel changes, copy it again.
#include <algorithm>
#include "pipelines.cuh"

#ifndef GPUDECODE_BAND_CONJ_DC
#define GPUDECODE_BAND_CONJ_DC FUSE_BAND_CONJ_DC
#endif

// ---- copied from mpifxcorr/src/gpumode.cu ----------------------------------------------------
// Adapted from https://forums.developer.nvidia.com/t/atomic-add-for-complex-numbers/39757
__device__ void atomicAddFloatComplex(cuFloatComplex* a, cuFloatComplex b){
    // transform the addresses of real and imag. parts to double pointers
    auto *x = (float*) a;
    auto *y = x + 1;
    //use atomicAdd for float variables
    atomicAdd(x, cuCrealf(b));
    atomicAdd(y, cuCimagf(b));
}

/** The fractional-sample rotation. With DOAUTOCORR it also accumulates the
 * per-band autocorrelations (an atomicAdd per element) and the cross-pol
 * autocorrelations (a second pass over the bands, re-reading the rotated
 * spectra); without it, it is a pure elementwise rotate and the XMAC computes
 * the autocorrelations as ordinary baselines instead - see
 * docs/gpu-autocorr-design.md. On the A100 those two phases were measured at
 * 24% and 25% of this kernel. */
template<bool DOAUTOCORR>
__global__ void gpu_resultsrotatorMultiply(
        cuFloatComplex* const gpufftd_gpu,
        cuFloatComplex* const autocorrelations,
        const float* const fracSampleError,
        const bool* const validSamples,
        const unsigned int* const indices,
        const double* const recordedfreqclockoffsets,
        const double* const recordedfreqclockoffsetsdelta,
        const double* const fracFreqOffset,
        const unsigned char* const bandflags,
        const double recordedbandwidth,
        int fftloop,
        int startblock,
        int numblocks,
        size_t fftchannels,
        size_t recordedbandchannels,
        size_t numrecordedbands,
        size_t numrecordedfreqs,
	bool calccrosspolautocorrs,
	int* counts_gpu
    ) {

    //for (int ii=0; ii<numrecordedfreqs; ii++){
    //    printf("counts_gpu[%d] = %d\n",ii,counts_gpu[ii]);
    //}

    // numBufferedFFTs(blockIdx.x) * fftchannels(threadIdx.x)

    // blockIdx.x in this case is the subloopindex index [0 .. numBufferedFFTs]
    // blockIdx.y in this case is the fftchannels_grid. The actual fftchannels value is calculated by fftchannels_grid idx * fftchannels_block size + fftchannels idx (blockIdx.y * blockDim.y) + threadIdx.y
    // threadIdx.x in this case is the fftchannels_block index [0 .. fftchannels_block]
    // blockDim.x in this case is the fftchannels_block size
    // gridDim.x in this case is the numBufferedFFTs size
    // gridDim.y in this case is the fftchannels_grid size

    // Check if this subloopindex is valid
    const size_t subloopindex = blockIdx.x;
    if (!validSamples[subloopindex]) {
        // Not valid, so don't do anything
        return;
    }

    // Check if we should bother processing this sample
    size_t index = fftloop * gridDim.x + subloopindex + startblock;
    if (index >= startblock + numblocks) {
        // May not have to fully complete last fftloop, drop out
        return;
    }

    const size_t channelindex = (blockIdx.y * blockDim.x) + threadIdx.x;

    if (channelindex >= recordedbandchannels) {
        return;
    }

    for (size_t bandindex = 0; bandindex < numrecordedbands; bandindex++) {
        /* Creating a fractional sample rotation array
         *  The actual calculation being performed is as follows:
         *  Assume we know the frequency of every FFT output channel, and it is stored in an array of length fftchannels, called channelfreq
         *  then for every frequency subband f (in the range 0 … recordedbandchannels), calculate the slope as:
         *  A = fracsampleerror - recordedfreqclockoffsets[f] + recordedfreqclockoffsetsdelta[f]/2
         *  (for the second polarisation, a is identical except subtracting recordedfreqclockoffsetsdelta[f]/2)
         * then calculate complexrotator[j]  (for j = 0 to fftchannels-1) as:
         * complexrotator[j] = exp( 2 pi i * (A*fftchannels[j]) )
         *
         * So how is fftchannels calculated? For “regular data” it is as follows (for j = 0 to fftchannels-1)
         * fftchannels[j] = recordedbandwidth * j / fftchannels
         * For lower sideband data it is:
         * fftchannels[j] = -recordedbandwidth * j / fftchannels
         * For double sideband data it is:
         * fftchannels[j] = recordedbandwidth * j / fftchannels - recordedbandwidth/2.0
        */

        // todo: Move recorded freq out of the kernel as a dim?
        const size_t dataIndex = (subloopindex * fftchannels * numrecordedbands) + (bandindex * fftchannels) + channelindex;

        // Calculate fracsampleerror - recordedfreqclockoffsets[f] + recordedfreqclockoffsetsdelta[f]/2
        double bigAval = fracSampleError[subloopindex] - recordedfreqclockoffsets[bandindex] + recordedfreqclockoffsetsdelta[bandindex] / 2;

        // Calculate fftchannels[j] = recordedbandwidth * j / fftchannels (every
        // band is upper sideband by now; fracFreqOffset is non-zero only on the
        // CPU-compatible sideband test path)
        double subFreq = recordedbandwidth * (double) channelindex / (double) recordedbandchannels + fracFreqOffset[bandindex];

        // Calculate
        double exponent = bigAval * subFreq;
        exponent -= int(exponent);
        cuFloatComplex cr;
        __sincosf(TWO_PI * exponent, &cr.y, &cr.x);
        cuFloatComplex in = gpufftd_gpu[dataIndex];
        if (channelindex == 0 && (bandflags[bandindex] & GPUDECODE_BAND_CONJ_DC))
            in.y = -in.y;
        const cuFloatComplex v = cuCmulf(in, cr);
        gpufftd_gpu[dataIndex] = v;

        // Autocorrelation, straight from the register: v * conj(v) is |v|^2,
        // which is real by construction, so only the real component is
        // accumulated (the imaginary one is provably zero and used to cost a
        // second atomic on every element).
        if (DOAUTOCORR) {
            const size_t autocorrIndex = (bandindex * recordedbandchannels) + channelindex;
            atomicAdd(&((float*)&autocorrelations[autocorrIndex])[0],
                      v.x * v.x + v.y * v.y);
        }
    }

    if (DOAUTOCORR) {
        // Cross-polarisation autocorrelations. Both operands now come from the
        // rotated spectra and the second is conjugated in the multiply
        // (cuCmulConjf below) - there is no materialised conjugate array any more.
        // Each value read here was written by THIS thread earlier in the band loop
        // (same window, same channel, a different band), so there is no race.
        for (size_t recordedfreq = 0; recordedfreq < numrecordedfreqs; recordedfreq++) {
            if (calccrosspolautocorrs && counts_gpu[recordedfreq] > 1) {
                const size_t bandA = indices[(recordedfreq * MAX_INDICIES) + 0];
                const size_t bandB = indices[(recordedfreq * MAX_INDICIES) + 1];
                const size_t windowBase = (subloopindex * fftchannels * numrecordedbands) + channelindex;
                const size_t idxA = windowBase + (bandA * fftchannels);
                const size_t idxB = windowBase + (bandB * fftchannels);
                const size_t crossBase = (numrecordedbands * recordedbandchannels) + channelindex;

                atomicAddFloatComplex(&autocorrelations[crossBase + (bandA * recordedbandchannels)],
                                      cuCmulConjf(gpufftd_gpu[idxA], gpufftd_gpu[idxB]));
                atomicAddFloatComplex(&autocorrelations[crossBase + (bandB * recordedbandchannels)],
                                      cuCmulConjf(gpufftd_gpu[idxB], gpufftd_gpu[idxA]));
            }
        }
    }
}

// ---- end of copy -----------------------------------------------------------------------------

// Builds the FP32 FFT input of a pre-FFT stage in the staging buffer (unfused pipeline only).
template<class Pre>
__global__ void materialise(Geometry g, Pre pre, cuFloatComplex* out, int ant) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.perAnt()) return;
    const int s = i % g.fftchannels;
    const size_t t = i / g.fftchannels;
    out[(size_t)ant * g.perAnt() + i] = pre(g, ant, t / g.nband, t % g.nband, s);
}

// mpifxcorr's rotation loop, written over a post-FFT stage object: thread per (window, channel),
// looping over the bands, then the cross-pol pass. Same launch geometry as the production kernel.
template<class Post>
__global__ void post_kernel(Geometry g, Post post, cuFloatComplex* out, int ant) {
    const int win = blockIdx.x;
    const int chan = blockIdx.y * blockDim.x + threadIdx.x;
    if (!post.windowValid(g, ant, win) || chan >= g.nchan)
        return;
    cuFloatComplex* spectra = out + ant * g.perAnt();
    for (int band = 0; band < g.nband; band++) {
        const size_t idx = ((size_t)win * g.nband + band) * g.fftchannels + chan;
        spectra[idx] = post.apply(g, ant, win, band, chan, spectra[idx]);
    }
    if (Post::doAutocorr)
        post.crossPol(g, ant, win, chan, spectra);
}

__global__ void fixed_phase_kernel(Geometry g, FracParams p, ulonglong2* out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.spectra()) return;
    const int band = i % g.nband;
    const size_t aw = i / g.nband;               // ant * nwin + win
    const int ab = (int)(aw / g.nwin) * g.nband + band;
    // the same bigA and channel frequencies as the production kernel
    const double bigAval = p.fracSampleError[aw] - p.clockOffsets[ab] + p.clockOffsetsDelta[ab] / 2;
    double slope = bigAval * p.recordedBandwidth / (double)g.nchan;   // turns per channel
    double offset = bigAval * p.fracFreqOffset[ab];                    // turns
    slope -= floor(slope);
    offset -= floor(offset);
    const double two64 = 18446744073709551616.0;
    out[i] = make_ulonglong2(offset < 1.0 ? __double2ull_rz(offset * two64) : 0ull,
                             slope < 1.0 ? __double2ull_rz(slope * two64) : 0ull);
}

// The fringe phase exp(-2 pi i (A j + B)) of gpudifx's decode kernels, as a fixed-point offset
// (B) and slope (A) per (antenna, window, band); one thread each.
__global__ void fringe_phase_kernel(Geometry g, PackedInput d) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.spectra()) return;
    double a = d.bigA[i], b = d.bigBred[i];
    a -= floor(a);
    b -= floor(b);
    const double two64 = 18446744073709551616.0;
    d.fringePhase[i] = make_ulonglong2(b < 1.0 ? __double2ull_rz(b * two64) : 0ull,
                                       a < 1.0 ? __double2ull_rz(a * two64) : 0ull);
}

// Each window's place in the packed data (WinDesc in decode.cuh); one thread per (antenna, window).
__global__ void window_desc_kernel(Geometry g, PackedInput d) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)g.nant * g.nwin) return;
    const int ant = (int)(i / g.nwin);
    const long fs = d.ms.framesamples;
    const long gs0 = d.sampleIndex[i];
    const long frame0 = gs0 >= 0 ? gs0 / fs : -((-gs0 + fs - 1) / fs);      // floor
    const int sif0 = (int)(gs0 - frame0 * fs);
    const int nf = (sif0 + g.fftchannels - 1) / (int)fs + 1;               // <= 32 (leanDecodeFits)
    unsigned mask = 0;
    for (int k = 0; k < nf; k++) {
        const long f = frame0 + k;
        if (f >= 0 && f < d.nframes && d.validFrames[(size_t)ant * d.nframes + f])
            mask |= 1u << k;
    }
    d.winDesc[i] = make_int4((int)(frame0 * d.ms.framebytes + d.ms.payloadoffset), sif0, (int)mask,
                             (int)((d.datasamples + gs0) & 1));
}

void launchPackedSetup(const Geometry& g, const PackedInput& d, cudaStream_t st) {
    const unsigned tpb = 256;
    fringe_phase_kernel<<<(unsigned)((g.spectra() + tpb - 1) / tpb), tpb, 0, st>>>(g, d);
    if (d.winDesc)
        window_desc_kernel<<<(unsigned)(((size_t)g.nant * g.nwin + tpb - 1) / tpb), tpb, 0, st>>>(g, d);
    CHECK_CUDA(cudaGetLastError());
}

void launchFixedPhase(const Geometry& g, const FracParams& p, cudaStream_t st) {
    const unsigned tpb = 256;
    fixed_phase_kernel<<<(unsigned)((g.spectra() + tpb - 1) / tpb), tpb, 0, st>>>(
            g, p, const_cast<ulonglong2*>(p.fixedPhase));
}

void Unfused::init(const Geometry& geom, bool batchedPlan, bool needStaging) {
    g = geom;
    batched = batchedPlan;
    cudaDeviceProp prop;
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    maxThreadsPerBlock = prop.maxThreadsPerBlock;
    // the plan exactly as GPUMode makes it, one per antenna (or one for all)
    int n[] = {g.fftchannels};
    int inembed[] = {0}, onembed[] = {0};
    const int nplans = batched ? 1 : g.nant;
    const int batch = (batched ? g.nant : 1) * g.nwin * g.nband;
    plans.resize(nplans);
    for (auto& p : plans)
        CHECK_CUFFT(cufftPlanMany(&p, 1, n, inembed, 1, g.fftchannels, onembed, 1, g.fftchannels, CUFFT_C2C, batch));
    if (needStaging)
        CHECK_CUDA(cudaMalloc(&staging, g.perAnt() * g.nant * sizeof(cuFloatComplex)));
}

void Unfused::destroy() {
    for (auto p : plans) cufftDestroy(p);
    plans.clear();
    if (staging) cudaFree(staging);
    staging = nullptr;
}

void Unfused::runPre(const PreVariant& pre, cudaStream_t st) {
    const unsigned tpb = 256;
    if (pre.packed && decodeRotator == Rotator::DOUBLE) {
        // gpudifx today: its tiled decode + fringe-rotation kernel, FP64 phases, one launch per
        // antenna (GPUMode::fringeRotation)
        const PackedInput& d = pre.pk;
        for (int a = 0; a < g.nant; a++) {
            FringeTileArgs args{};
            args.dest = staging + (size_t)a * g.perAnt();
            args.sampleIndexes = d.sampleIndex + (size_t)a * g.nwin;
            args.validSamples = validSamples + (size_t)a * g.nwin;
            args.bigA = d.bigA + (size_t)a * g.nwin * g.nband;
            args.bigBred = d.bigBred + (size_t)a * g.nwin * g.nband;
            args.fftloop = 0;
            args.startblock = 0;
            args.numblocks = g.nwin;
            args.numBufferedFFTs = g.nwin;
            args.fftchannels = g.fftchannels;
            args.numrecordedbands = g.nband;
            args.ms = d.ms;
            args.packed = d.packed + (size_t)a * d.bytesPerAnt;
            args.valid_frames = d.validFrames + (size_t)a * d.nframes;
            args.skipped = 0;
            args.framestounpack = d.nframes;
            args.datasamples = (int)d.datasamples;
            args.codif = false;
            args.bandflags = d.decodeFlags + (size_t)a * g.nband;
            dispatch_fused_fringe_tiled<false, false>(st, g.nwin, args);
        }
        CHECK_CUDA(cudaGetLastError());
        return;
    }
    if (pre.packed)
        launchPackedSetup(g, pre.pk, st);
    const unsigned blocks = (g.perAnt() + tpb - 1) / tpb;
    withPre(pre, [&](auto stage) {
        if (stage.direct())
            return;     // already an FP32 buffer per window: cuFFT reads it directly
        for (int a = 0; a < g.nant; a++)
            materialise<<<blocks, tpb, 0, st>>>(g, stage, staging, a);
    });
}

void Unfused::runFFT(const PreVariant& pre, cuFloatComplex* out, cudaStream_t st) {
    bool direct = false;
    withPre(pre, [&](auto stage) { direct = stage.direct(); });
    cuFloatComplex* in = direct ? (cuFloatComplex*)pre.in : staging;
    for (size_t i = 0; i < plans.size(); i++) {
        const size_t off = batched ? 0 : i * g.perAnt();
        CHECK_CUFFT(cufftSetStream(plans[i], st));
        CHECK_CUFFT(cufftExecC2C(plans[i], in + off, out + off, CUFFT_FORWARD));
    }
}

void Unfused::runFrac(const FracParams& p, AutoMode mode, Rotator rot, cuFloatComplex* out, cudaStream_t st) {
    // launch geometry of GPUMode::fractionalRotation
    size_t fftchannels_block = g.nchan;
    size_t fftchannels_grid = 1;
    size_t divisor = maxThreadsPerBlock;
    if ((size_t)g.nchan > divisor) {
        fftchannels_block = divisor;
        fftchannels_grid = (g.nchan + divisor - 1) / divisor;
    }
    const dim3 rotgrid(g.nwin, fftchannels_grid);
    const dim3 rotblock(fftchannels_block);
    if (rot == Rotator::FIXED) {
        launchFixedPhase(g, p, st);
        FracParams q = p;
        q.crossPol = (mode == AutoMode::FULL) && p.crossPol;
        for (int a = 0; a < g.nant; a++) {
            if (mode == AutoMode::NONE) post_kernel<<<rotgrid, rotblock, 0, st>>>(g, FracRotateFixed<false>{{q}}, out, a);
            else                        post_kernel<<<rotgrid, rotblock, 0, st>>>(g, FracRotateFixed<true>{{q}}, out, a);
        }
        return;
    }
    for (int a = 0; a < g.nant; a++) {
        cuFloatComplex* data = out + a * g.perAnt();
        const float* fse = p.fracSampleError + (size_t)a * g.nwin;
        const bool* valid = p.validSamples + (size_t)a * g.nwin;
        const size_t ab = (size_t)a * g.nband;
        if (mode == AutoMode::NONE) {
            gpu_resultsrotatorMultiply<false><<<rotgrid, rotblock, 0, st>>>(
                    data, nullptr, fse, valid, nullptr,
                    p.clockOffsets + ab, p.clockOffsetsDelta + ab, p.fracFreqOffset + ab, p.bandFlags + ab,
                    p.recordedBandwidth, 0, 0, p.numBlocks,
                    g.fftchannels, g.nchan, g.nband, p.numRecordedFreqs, false, nullptr);
        } else {
            gpu_resultsrotatorMultiply<true><<<rotgrid, rotblock, 0, st>>>(
                    data, p.autocorr + (size_t)a * 2 * g.nband * g.nchan, fse, valid, p.indices,
                    p.clockOffsets + ab, p.clockOffsetsDelta + ab, p.fracFreqOffset + ab, p.bandFlags + ab,
                    p.recordedBandwidth, 0, 0, p.numBlocks,
                    g.fftchannels, g.nchan, g.nband, p.numRecordedFreqs,
                    mode == AutoMode::FULL && p.crossPol, const_cast<int*>(p.counts));
        }
    }
}
