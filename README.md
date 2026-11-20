# difx-prototyping

Quick prototyping code for performance improvements (or correctness tests) for DiFX.

Prototypes and standalone benchmarks for DiFX development are kept here, out of the main
[difx](https://github.com/difx/difx) repository. Code here is for trying ideas out and measuring
them. It is not built or installed with DiFX, and nothing in difx depends on it. When an idea
works, it is reimplemented in difx and the prototype stays here as the record of how it was
evaluated.

| directory | what |
|---|---|
| `fft-fusion/` | GPU (gpudifx): the FFT and the fractional-sample correction fused into one cuFFTDx kernel, vs cuFFT + mpifxcorr's rotation kernel; FP16 FFT input; windowed overlap-add front end |
| `pulsar-binning/` | GPU pulsar gating and binning in the XMAC (backlog: not started) |
| `phase-centres/` | GPU correlation to many phase centres at once, shifted and averaged from high-resolution visibilities (backlog: not started) |

Each directory has its own README with how to build and run it, and its results.
