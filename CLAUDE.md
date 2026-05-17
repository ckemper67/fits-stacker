# fits-stacker

Standalone FITS frame stacker. Single translation unit (`fits_stack.cpp`) plus
a self-contained alignment engine in `donuts/`.

## Build

    make                  # native NEON/SSE2 bicubic warp + 2x2 Bayer average
    make OPENCV=1         # OpenCV bilinear debayer + warpAffine (same binary,
                          # runtime selectable via -b flag)
    make clean

No cmake. No install step. Binary lands at `./fits_stack`.

## Run

    ./fits_stack [-o out.fits] [-p out.png]
                 [-m mean|median|sigmaclip|wstream] [-k kappa]
                 [-j threads] [-w warmup_frames] [-b native|opencv]
                 frame1.fits frame2.fits ...

Inputs sorted alphabetically; first is the alignment reference.

Quick smoke test (same file three times -- should give near-zero dx/dy):

    ./fits_stack -m wstream /path/to/any.fits /path/to/any.fits /path/to/any.fits

## Architecture

    fits_stack.cpp
      loadFITS()          cfitsio float read, BZERO/BSCALE applied by library
      extractGreen()      half-res green channel for DONUTS registration
      debayer()           2x2 average (native) or OpenCV bilinear (opencv)
      alignFrame3rows()   Catmull-Rom bicubic warp, NEON/SSE2 fast interior path
      alignFrame3()       dispatcher: runtime -b flag selects native or OpenCV
      welfordFrame()      online mean+M2 accumulation with G-channel gate
      seedWelford3()      warm-up buffer sigmaclip seeder (anti first-frame bias)
      reduceStack3()      frame-major NaN-sentinel median/sigmaclip reduce
      main()              pass1 (DONUTS measure) -> pass2 (align+accumulate) -> write

    donuts/
      donuts.h / donuts.cpp    4-quadrant phase-only cross-correlation solver
      pocketfft_hdronly.h      FFT engine (BSD-3-Clause, Max-Planck-Society)

## Key design decisions

**float pixels** throughout (`using Pix = float`). Halves memory vs double;
enables 4-wide NEON/SSE2 in the bicubic kernel. cfitsio reads directly to
`TFLOAT` -- no double intermediate.

**Frame-major pixel stacks** for median/sigmaclip: `vector<vector<Pix>>`
allocated one frame at a time. NaN marks border (masked) pixels. Avoids a
large upfront allocation and gives a clear OOM error per frame.

**Welford streaming** (`-m wstream`): O(pixels) memory regardless of frame
count. G-channel-only rejection gate: satellites are broadband so one channel
suffices; avoids false rejection of red stars with high R variance.

**Warm-up buffer** (`-w N`, default 5): first N accepted frames are sigma-clipped
per pixel to seed clean Welford `{mean, M2}` state before switching to online
streaming. Prevents a satellite in frames 1-2 from biasing the baseline.

**SIMD dispatch** at compile time: `__ARM_NEON + __aarch64__` -> NEON AArch64,
`__ARM_NEON` only -> ARMv7 NEON (Pi), `__SSE2__` -> SSE2 x86, else scalar.
`hsum_neon()` / `hsum_ps()` handle the horizontal-sum difference between
AArch64 (`vaddvq_f32`) and ARMv7 (`vpadd_f32`).

**OpenCV backend** (`OPENCV=1`, `-b opencv`): runtime flag -- both paths
compiled into one binary. Default is OpenCV when available. Use `-b native` to
force the hand-rolled path. Timing breakdown printed at end of every run.

**AffineMatrix convention**: maps destination pixel -> source pixel (inverse
warp). `warpAffine` uses `WARP_INVERSE_MAP` to match.

## Dependencies

- cfitsio (required)
- libpng (required)
- opencv4 (optional, `make OPENCV=1`)

All found via pkg-config with a homebrew prefix fallback on macOS.
