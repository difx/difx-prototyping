// gpudecode_copy.cuh - gpudifx's decode helpers and its tiled decode + fringe-rotation kernel,
// copied from mpifxcorr/src/gpudecode.cu (difx/difx branch gpudifx, e2e1e58ab).  Included by
// decode.cuh, which defines the struct mark5_stream these use.  Changes from the original:
//   - the __constant__ lookup tables are static (this header is included in several files)
//   - the phase-cal folding (the DOPCAL branch of gpu_fused_fringe_tiled, with its debug print)
//     is cut out; the benchmark only instantiates DOPCAL = false, which compiles it away anyway
// Everything else is verbatim. If the production code changes, copy it again.
#pragma once

// ---- copied from mpifxcorr/src/gpudecode.cu --------------------------------------------------
#ifndef GPUDECODE_TWO_PI
#define GPUDECODE_TWO_PI 6.283185307179586476925286766559
#endif

/* the high mag value for 2-bit reconstruction */
static const float HiMag = OPTIMAL_2BIT_HIGH;
static const float FourBit1sigma = 2.95;

// Stack all quantization levels (nbit <= 2) float values in one array and use nbit to offset accordingly
static __constant__ float lutall[6] = {-1.0, 1.0, -HiMag, -1.0, 1.0, HiMag};
// The same for CODIF, whose 1- and 2-bit encodings differ from VDIF's
// (mark5access format_codif.c initluts): 1 bit 0 -> +1, 1 -> -1;
// 2 bit {+1, +HiMag, -HiMag, -1}.
static __constant__ float lutcodif[6] = {1.0, -1.0, 1.0, HiMag, -HiMag, -1.0};

__device__ __forceinline__ float lut_gpu(int index, bool codif) {
	return codif ? lutcodif[index] : lutall[index];
}

// CODIF samples of 4 bits or more are two's complement, scaled as in mark5access
// format_codif.c: 4 bit /FourBit1sigma, 8 bit /71, 16 bit /8.
__device__ __forceinline__ float codif_multibits_gpu(int32_t word, int pos, int nbit) {
	int v = (int)((word >> pos) & ((1L << nbit) - 1));
	if (v >= (1 << (nbit - 1)))
		v -= (1 << nbit);
	switch (nbit) {
	case 4:
		return v / FourBit1sigma;
	case 8:
		return v / 71.0f;
	default:
		return v / 8.0f;
	}
}

__device__ __forceinline__ float bitsread_gpu(char byte, int pos, int nbit) {
	// std::cout << "Byte : " << std::bitset<8>(byte) << std::endl;
	// std::cout << "Pos :  " << pos << "\tnbit :  " << nbit << std::endl;
	// std::cout << "Index: " << ((byte >> pos) & ((2 << (nbit - 1)) - 1)) + (nbit << 1) - 2 << std::endl << std::endl;
	return lutall[((byte >> pos) & ((1 << nbit) - 1)) + (nbit << 1) - 2];		// Should see if this can be optimised
}

__device__ __forceinline__ cuFloatComplex complex_bitsread_gpu(char byte, int pos, int nbit) {
	return make_cuFloatComplex(lutall[((byte >> pos) & ((1 << nbit) - 1)) + (nbit << 1) - 2], lutall[((byte >> (pos+nbit)) & ((1 << nbit) - 1)) + (nbit << 1) - 2]);
}


// Offset-binary zero point of VDIF samples of 4 bits or more, as in mark5access
// format_vdif.c: 4 bit levels are (v - 7.5)/FourBit1sigma, symmetric about
// zero with no zero level; 8 and 16 bit are (v - 2^(nbit-1)) scaled.
__device__ __forceinline__ float vdif_multibit_offset(int nbit) {
	return (nbit == 4) ? 7.5f : (float)(1L << (nbit - 1));
}

__device__ __forceinline__ float multibitsread_gpu(int32_t word, int pos, int nbit) {
	// Larger numbers of bits have equidistant quantization spacing

	float quant_factor;
	// TODO: define constants in a header maybe?
	switch (nbit) {
	case 4:
		quant_factor = 1.0 / FourBit1sigma;
		break;
	case 8:
		quant_factor = 1.0 / 3.3;
		break;
	case 16:
		quant_factor = 1.0 / 8.0;
		break;
	case 32:
		quant_factor = 1.0 / 8.0;		// ERROR: 32 bit doesn't work for some reason?
		break;
	default:
		break;
	}

	return (((word >> pos) & ((1L << nbit) - 1)) - vdif_multibit_offset(nbit)) * quant_factor;
}


__device__ __forceinline__ cuFloatComplex complex_multibitsread_gpu(int32_t word, int pos, int nbit) {
	// Larger numbers of bits have equidistant quantization spacing

	float quant_factor;
	// TODO: define constants in a header maybe?
	switch (nbit) {
	case 4:
		quant_factor = 1.0 / FourBit1sigma;
		break;
	case 8:
		quant_factor = 1.0 / 3.3;
		break;
	case 16:
		quant_factor = 1.0 / 8.0;
		break;
	default:
		break;
	}

	return make_cuFloatComplex(((((word >> pos) & ((1L << nbit) - 1)) - vdif_multibit_offset(nbit)) * quant_factor),((((word >> (pos+nbit)) & ((1L << nbit) - 1)) - vdif_multibit_offset(nbit)) * quant_factor));
}
// Decode a SINGLE real sample for one band directly from the packed frame
// payload. This is the per-(band, sample) primitive shared by the fused
// decode+fringe kernel; the old gpu_unpack looped this over every channel and
// wrote the result to a global unpacked buffer that the fringe kernel then read
// straight back. framepayload = packed + frame*framebytes + payloadoffset.
// codif selects CODIF's sample encoding instead of VDIF's.
__device__ __forceinline__ float decode_one_gpu(
		const unsigned char *framepayload, int nbit, int nchan, int decimation,
		int skipped, int sample_in_frame, int band, bool codif) {
	const bool bitreadflag = (nbit == 1) || (nbit == 2);
	int bit_counter = sample_in_frame * nbit * (nchan * decimation + skipped) + band * nbit;
	if (bitreadflag) {
		const int pos = bit_counter % 8;
		return lut_gpu(((framepayload[bit_counter / 8] >> pos) & ((1 << nbit) - 1)) + (nbit << 1) - 2, codif);
	}
	const int32_t word = ((const int32_t *)framepayload)[bit_counter / 32];
	if (codif)
		return codif_multibits_gpu(word, bit_counter % 32, nbit);
	return multibitsread_gpu(word, bit_counter % 32, nbit);
}

// Complex twin of decode_one_gpu (real+imag packed adjacently per sample, so
// twice the bits per band and per sample; mirrors the old
// mk5_decode_complex_sample_gpu bit layout).
__device__ __forceinline__ cuFloatComplex decode_one_complex_gpu(
		const unsigned char *framepayload, int nbit, int nchan, int decimation,
		int skipped, int sample_in_frame, int band, bool codif) {
	const bool bitreadflag = (nbit == 1) || (nbit == 2);
	int bit_counter = sample_in_frame * 2 * nbit * (nchan * decimation + skipped) + band * 2 * nbit;
	if (bitreadflag) {
		const unsigned char byte = framepayload[bit_counter / 8];
		const int pos = bit_counter % 8;
		const int mask = (1 << nbit) - 1;
		return make_cuFloatComplex(lut_gpu(((byte >> pos) & mask) + (nbit << 1) - 2, codif),
				lut_gpu(((byte >> (pos + nbit)) & mask) + (nbit << 1) - 2, codif));
	}
	const int32_t word = ((const int32_t *)framepayload)[bit_counter / 32];
	if (codif)
		return make_cuFloatComplex(codif_multibits_gpu(word, bit_counter % 32, nbit),
				codif_multibits_gpu(word, bit_counter % 32 + nbit, nbit));
	return complex_multibitsread_gpu(word, bit_counter % 32, nbit);
}

// Apply a band's decode-time transforms (GPUDECODE_BAND_* in gpudecode.cuh) to
// one decoded sample. absample is the sample's absolute number within the
// second (datasamples + global_sample), so the (-1)^n of ALTSIGN is anchored to
// station time - like a physical LO - and is continuous from one subint to the
// next. windowpos is its position in the FFT window, used instead by the
// CPU-compatible test path (GPUDECODE_BAND_ALTSIGN_WINDOW).
__device__ __forceinline__ cuFloatComplex band_transform_gpu(cuFloatComplex c, unsigned char flags,
		long absample, long windowpos) {
	if (flags & GPUDECODE_BAND_CONJ)
		c.y = -c.y;
	if (flags & GPUDECODE_BAND_ALTSIGN) {
		const long n = (flags & GPUDECODE_BAND_ALTSIGN_WINDOW) ? windowpos : absample;
		if (n & 1) {
			c.x = -c.x;
			c.y = -c.y;
		}
	}
	return c;
}

// Tiled twin of the fused decode+fringe kernels above: identical arithmetic,
// but the global write is transposed through shared memory so that it
// coalesces.
//
// The two halves of the kernel want opposite thread mappings. decode_one_gpu
// reads band-adjacent bits of one sample word, so band-on-lane costs a warp a
// single sector - that is why L1 hit rate is ~91% and DRAM throughput only
// ~21%. But `dest` is band-major (it is the cuFFT batched input layout, which
// is not ours to change), so band-on-lane scatters a warp's 8-byte stores over
// as many sectors as there are lanes: measured on a 2070, 29% excessive sectors
// and 58% of warp stall cycles waiting on L1TEX. This kernel keeps band on the
// lane for the decode+rotate, stages the result in shared memory, and puts
// channel on the lane for the store. Rationale, tile policy and measurements:
// docs/gpu-fringetile-design.md.
//
// One block owns BT bands x CT channels of one FFT window, BT*CT threads (256
// whenever the job is wide enough to fill them - see fringeTileCT), so every
// (bands, channels) shape stores the same 256-contiguous-byte run per warp. Grid is
// (numBufferedFFTs, ceil(fftchannels/CT), ceil(numrecordedbands/BT)); gridDim.x
// stays numBufferedFFTs because the kernel's `index` arithmetic uses it.
static const int FRINGE_TILE_THREADS = 256;

static __host__ __device__ constexpr int fringe_tile_log2(int v) {
	return (v <= 1) ? 0 : 1 + fringe_tile_log2(v >> 1);
}

// Every argument of the tiled kernel in one POD, so the 16 instantiations
// (4 tile shapes x pcal x real/complex) share one dispatch signature instead of
// repeating a 19-argument list. Passed by value, as `ms` already is.
struct FringeTileArgs {
	cuFloatComplex *dest;
	const int *sampleIndexes;
	const bool *validSamples;
	const double *bigA;
	const double *bigBred;
	int fftloop;
	int startblock;
	int numblocks;
	// Explicit rather than read back from gridDim.x: `index` below needs the
	// window count, and tying that to a grid dimension makes any later grid
	// reshuffle silently select the wrong FFT window.
	size_t numBufferedFFTs;
	size_t fftchannels;
	// Explicit, unlike the untiled kernels which read it from blockDim.x - a
	// tiled block's blockDim.x is the flat tile, not the band count.
	size_t numrecordedbands;
	struct mark5_stream ms;
	const void *packed;
	const bool *valid_frames;
	int skipped;
	int framestounpack;
	void *pcal_output;
	const int *N_pcal_bins;
	int datasamples;
	int pcal_bin_stride_length;
	bool codif;
	const unsigned char *bandflags;
};

template<int BT, int CT, bool DOPCAL, bool CPLX>
__global__ void gpu_fused_fringe_tiled(const FringeTileArgs a) {
	constexpr int BT_LOG2 = fringe_tile_log2(BT);
	constexpr int CT_LOG2 = fringe_tile_log2(CT);
	// At BT=1 the two lane decompositions coincide, so the transpose is an
	// identity: store straight to global and skip the shared round-trip and the
	// barrier. Without this the narrowest shapes pay for staging they cannot use
	// (measured 0.98x at 1 band x 64 channels, where the kernel is ~12 us).
	constexpr bool TRANSPOSE = (BT > 1);
	// Row padding, chosen so phase 1's band-strided shared writes hit 32
	// distinct banks. An 8-byte shared access is serviced in half-warps of 16
	// lanes (16 x 8 B = the 32 banks exactly), and phase 1's lane decomposition
	// is `lane = band + BT*ch`, so the 16 lanes tile the banks iff the row
	// stride in elements is congruent to 16/BT mod 16. Padding by one element
	// (the obvious choice, and what this first was) gives stride 1 mod 16 and a
	// 2-way conflict on every store - ncu measured 640k conflicts across 320k
	// store requests before this was fixed.
	constexpr int PAD = TRANSPOSE ? (16 / BT) : 0;
	__shared__ cuFloatComplex tile[BT][TRANSPOSE ? (CT + PAD) : 1];

	const size_t subloopindex = blockIdx.x;
	// Both of these depend only on blockIdx.x, so they are block-uniform and
	// returning here cannot leave part of the block stranded at the barrier.
	if (!a.validSamples[subloopindex]) return;
	const size_t index = a.fftloop * a.numBufferedFFTs + subloopindex + a.startblock;
	if (index >= a.startblock + a.numblocks) return;

	const int tid = threadIdx.x;
	const size_t bandbase = (size_t)blockIdx.z * BT;
	const size_t chbase = (size_t)blockIdx.y * CT;
	const int sample_index = a.sampleIndexes[subloopindex];

	// Phase 1 - band on the lane: decode this thread's raw sample, fold pcal,
	// apply the rotator, and leave the result in the shared tile.
	{
		const int band_local = tid & (BT - 1);
		const int ch_local = tid >> BT_LOG2;
		const size_t bandindex = bandbase + band_local;
		const size_t channelindex = chbase + ch_local;
		// Predicated, never returned from: every thread must reach the barrier.
		if (bandindex < a.numrecordedbands && channelindex < a.fftchannels) {
			const long global_sample = (long)sample_index + (long)channelindex;
			cuFloatComplex c = make_cuFloatComplex(0.f, 0.f);
			if (global_sample >= 0) {
				const int frame = (int)(global_sample / a.ms.framesamples);
				if (frame < a.framestounpack && a.valid_frames[frame]) {
					const int sample_in_frame = (int)(global_sample - (long)frame * a.ms.framesamples);
					const unsigned char *fp = (const unsigned char *)a.packed +
							(size_t)frame * a.ms.framebytes + a.ms.payloadoffset;
					if (CPLX)
						c = decode_one_complex_gpu(fp, a.ms.nbit, a.ms.nchan, a.ms.decimation,
								a.skipped, sample_in_frame, (int)bandindex, a.codif);
					else
						c = make_cuFloatComplex(decode_one_gpu(fp, a.ms.nbit, a.ms.nchan,
								a.ms.decimation, a.skipped, sample_in_frame, (int)bandindex, a.codif), 0.f);
				}
			}
			c = band_transform_gpu(c, a.bandflags[bandindex], (long)a.datasamples + global_sample, (long)channelindex);

			// [phase-cal folding (DOPCAL) cut out here, see the top of this file]

			const double bigAval = a.bigA[subloopindex * a.numrecordedbands + bandindex];
			const double bigB_reduced = a.bigBred[subloopindex * a.numrecordedbands + bandindex];
			double exponent = (bigAval * (double)channelindex + bigB_reduced);
			exponent -= int(exponent);
			cuFloatComplex cr;
			__sincosf(-GPUDECODE_TWO_PI * exponent, &cr.y, &cr.x);
			const cuFloatComplex val = cuCmulf(c, cr);
			if (TRANSPOSE) {
				tile[band_local][ch_local] = val;
			} else {
				const size_t destIndex = (subloopindex * a.fftchannels * a.numrecordedbands) +
						(bandindex * a.fftchannels) + channelindex;
				a.dest[destIndex] = val;
			}
		}
	}

	if (TRANSPOSE) {
		__syncthreads();

		// Phase 2 - channel on the lane: each warp stores one run of 32
		// consecutive channels, i.e. 256 contiguous bytes of fully-used sectors.
		const int ch_local = tid & (CT - 1);
		const int band_local = tid >> CT_LOG2;
		const size_t bandindex = bandbase + band_local;
		const size_t channelindex = chbase + ch_local;
		if (bandindex < a.numrecordedbands && channelindex < a.fftchannels) {
			const size_t destIndex = (subloopindex * a.fftchannels * a.numrecordedbands) +
					(bandindex * a.fftchannels) + channelindex;
			a.dest[destIndex] = tile[band_local][ch_local];
		}
	}
}

// The tiled path is on by default; DIFX_GPU_FRINGE_TILE=0 selects the untiled
// kernels above unchanged, so the two can be A/B'd (and the tiled path backed
// out on an architecture where it does not pay) without a rebuild.
static bool fringeTileEnabled() {
	// Function-local static: initialisation is thread-safe by the standard
	// (C++11 6.7/4), which the read-modify-written `static int cached = -1`
	// idiom used elsewhere in this tree is not - launch_fused_fringe is called
	// from every Core processing thread, one Mode per thread.
	static const bool enabled = []() {
		// Default on. Only an explicitly off-ish value disables it: an
		// `atoi(e) == 0` test would have made DIFX_GPU_FRINGE_TILE=true, =yes
		// and =<empty> all silently select the OLD path, i.e. the opposite of
		// what someone typing them means.
		const char *e = getenv("DIFX_GPU_FRINGE_TILE");
		if (e == NULL) return true;
		const bool off = (strcmp(e, "0") == 0) || (strcasecmp(e, "false") == 0) ||
				(strcasecmp(e, "no") == 0) || (strcasecmp(e, "off") == 0);
		return !off;
	}();
	return enabled;
}

// BT = the largest power of two <= 8 that DIVIDES numrecordedbands. Powers of
// two so the two index decompositions are shifts and masks; capped at 8 so CT
// never drops below 32 and a warp's store run stays 256 B; and a divisor so no
// block is left with idle band slots. The divisor condition costs nothing at the
// band counts VLBI actually uses (1, 2, 4, 8, 16, 32, 64) and stops odd counts
// paying for a partly-empty tile - measured: 3 and 6 bands lost 0-3% under a
// plain pow2_floor rule, which rounded 3 up to 4 and 6 up to 8 band slots.
static int fringeTileBT(size_t numrecordedbands) {
	if (numrecordedbands % 8 == 0) return 8;
	if (numrecordedbands % 4 == 0) return 4;
	if (numrecordedbands % 2 == 0) return 2;
	return 1;
}

template<int BT, int CT, bool DOPCAL, bool CPLX>
static void launch_fused_fringe_tiled(cudaStream_t stream, size_t numBufferedFFTs,
		const FringeTileArgs &a) {
	const dim3 grid((unsigned int)numBufferedFFTs,
			(unsigned int)((a.fftchannels + CT - 1) / CT),
			(unsigned int)((a.numrecordedbands + BT - 1) / BT));
	gpu_fused_fringe_tiled<BT, CT, DOPCAL, CPLX><<<grid, BT * CT, 0, stream>>>(a);
}

// CT = 256/BT, but never more channels than the job actually has, and never
// below 32 (a warp must still store one contiguous run). Without the clamp a
// narrow job wastes most of each block: at 1 band x 64 channels, CT=256 left 64
// of 256 threads working and the tiled path measured **0.46x** - the untiled
// geometry sizes its block to the channel count and wins outright there.
static int fringeTileCT(int bt, size_t fftchannels) {
	int ct = FRINGE_TILE_THREADS / bt;
	while (ct > 32 && (size_t)ct > fftchannels) ct >>= 1;
	return ct;
}

template<bool DOPCAL, bool CPLX>
static void dispatch_fused_fringe_tiled(cudaStream_t stream, size_t numBufferedFFTs,
		const FringeTileArgs &a) {
	const int bt = fringeTileBT(a.numrecordedbands);
	const int ct = fringeTileCT(bt, a.fftchannels);
	switch (bt) {
		case 8:
			launch_fused_fringe_tiled<8, 32, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			break;
		case 4:
			if (ct >= 64) launch_fused_fringe_tiled<4, 64, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			else          launch_fused_fringe_tiled<4, 32, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			break;
		case 2:
			if (ct >= 128)     launch_fused_fringe_tiled<2, 128, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			else if (ct >= 64) launch_fused_fringe_tiled<2, 64, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			else               launch_fused_fringe_tiled<2, 32, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			break;
		default:
			if (ct >= 256)      launch_fused_fringe_tiled<1, 256, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			else if (ct >= 128) launch_fused_fringe_tiled<1, 128, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			else if (ct >= 64)  launch_fused_fringe_tiled<1, 64, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			else                launch_fused_fringe_tiled<1, 32, DOPCAL, CPLX>(stream, numBufferedFFTs, a);
			break;
	}
}
// ---- end of copy -----------------------------------------------------------------------------
