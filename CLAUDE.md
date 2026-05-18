# fits-stacker

Standalone FITS frame stacker for EAA (Electronic Assisted Astronomy).
Single translation unit (`fits_stack.cpp`) plus a self-contained alignment
engine in `donuts/`.

## Build

    make                  # native NEON/SSE2 bicubic warp + MHC post-stack debayer
    make OPENCV=1         # adds OpenCV warp + bilinear debayer paths (same binary,
                          # selectable at runtime via -b flag)
    make clean

No cmake. No install step. Binary lands at `./fits_stack`.

## Run

    ./fits_stack [-o out.fits] [-o preview.png]
                 [-m mean|median|sigmaclip|wstream] [-k kappa]
                 [-j threads] [-w warmup_frames] [-b native|opencv]
                 frame1.fits frame2.fits ...

-o may be repeated; extension picks format (.fits/.fit -> FITS, .png -> PNG).

Defaults:
  -m wstream   online Welford streaming with per-pixel sigma-clip rejection
  -k 3.0       kappa for sigma clipping
  -w 5         warm-up frames before switching to online streaming
  -b native    Catmull-Rom bicubic warp + MHC post-stack debayer
  -j 0         all hardware threads
  output       stacked.fits

Inputs sorted alphabetically; first is the alignment reference.

EAA workflow -- invoke after each new frame with all frames accumulated so far:

    ./fits_stack -o live.fits -o live.png frame*.fit

Warm-up behaviour: with N < warmup frames, sigma-clips all N frames and
writes output. Full streaming kicks in once warmup count is reached.

Quick smoke test (same file three times -- should give near-zero dx/dy):

    ./fits_stack /path/to/any.fits /path/to/any.fits /path/to/any.fits

## Architecture

    fits_stack.cpp
      loadFITS()               cfitsio float read, BZERO/BSCALE applied by library
      extractGreen()           half-res green channel for DONUTS registration
      extractSubplanes()       SIMD deinterleave: Bayer -> R/G1/G2/B half-res planes
      debayerMHC_halfres()     SIMD Malvar-He-Cutler 2004 post-stack demosaic
      debayerOpenCVPostStack() reassemble Bayer from stacked subplanes, cv::cvtColor
      alignFrame4rows()        Catmull-Rom bicubic warp, NEON/SSE2 fast interior path
      alignFrame4()            dispatcher: -b native (bicubic) or -b opencv (warpAffine)
      welfordFrame()           online mean+M2 accumulation, (G1+G2)/2 rejection gate
      seedWelford4()           warm-up buffer sigmaclip seeder (anti first-frame bias)
      reduceStack4()           frame-major NaN-sentinel median/sigmaclip reduce
      main()                   pass1 (DONUTS measure) -> pass2 (align+accumulate) -> write

    donuts/
      donuts.h / donuts.cpp    4-quadrant phase-only cross-correlation solver
      pocketfft_hdronly.h      FFT engine (BSD-3-Clause, Max-Planck-Society)

## Key design decisions

**Bayer subplane stacking**: raw frames are split into R, G1, G2, B half-res
planes before warping. All four planes are stacked independently; MHC
demosaicing is applied once to the final stack. This gives MHC the benefit
of noise-reduced data and avoids demosaic artifacts accumulating across frames.

**MHC demosaicing** (Malvar-He-Cutler 2004): 5x5 linear kernels applied
post-stack at half resolution. Each output pixel averages 4 MHC estimates
(one per Bayer position in the 2x2 block). SIMD-vectorized interior path
(NEON `vmlaq_n_f32` / SSE2 `_mm_mul_ps`), scalar border path with clamping.

**SIMD subplane extraction**: `vld2q_f32` (NEON) or two `_mm_shuffle_ps`
(SSE2) deinterleave two Bayer rows into even/odd subplane pairs per iteration.

**float pixels** throughout (`using Pix = float`). Halves memory vs double;
enables 4-wide NEON/SSE2 SIMD. cfitsio reads directly to `TFLOAT`.

**Welford streaming** (default `-m wstream`): O(pixels) memory regardless of
frame count. (G1+G2)/2 rejection gate: sqrt(2) better SNR than single-channel
gate; satellites are broadband so one averaged channel suffices.

**Warm-up buffer** (`-w N`, default 5): first N accepted frames are
sigma-clipped per pixel to seed clean Welford `{mean, M2}` state. Partial
runs (N < warmup) still produce output -- sigma-clip of all frames seen.

**Frame-major pixel stacks** for median/sigmaclip: `vector<vector<Pix>>`
allocated one frame at a time. NaN marks masked border pixels.

**SIMD dispatch** at compile time: `__ARM_NEON + __aarch64__` -> NEON AArch64,
`__ARM_NEON` only -> ARMv7 NEON (Pi), `__SSE2__` -> SSE2 x86, else scalar.

**OpenCV backend** (`OPENCV=1`, `-b opencv`): both backends compiled into one
binary. OpenCV uses `cv::warpAffine` (INTER_CUBIC) for warping and
`cv::cvtColor` bilinear + INTER_AREA for post-stack demosaic. Default backend
is always native regardless of build flags.

**AffineMatrix convention**: maps destination pixel -> source pixel (inverse
warp). `warpAffine` uses `WARP_INVERSE_MAP` to match.

## Dependencies

- cfitsio (required)
- libpng (required)
- opencv4 (optional, `make OPENCV=1`)

All found via pkg-config with a homebrew prefix fallback on macOS.
