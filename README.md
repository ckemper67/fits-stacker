# fits-stacker

Standalone FITS stacker using DONUTS-inspired rotation/translation alignment.

Aligns a sequence of FITS frames using 4-quadrant phase-only cross-correlation
(translation + rotation) and stacks them with a choice of combine methods.
Supports monochrome and Bayer (RGGB / GRBG / GBRG / BGGR) colour cameras.

Output is a single-plane FITS (mono) or 3-plane R/G/B FITS (colour), plus an
optional PNG with asinh stretch for quick preview.

## Dependencies

- cfitsio
- libpng

### macOS (Homebrew)

    brew install cfitsio libpng

### Debian/Ubuntu

    sudo apt install libcfitsio-dev libpng-dev

### Optional: OpenCV (better debayer and warp)

    brew install opencv          # macOS
    sudo apt install libopencv-dev   # Debian/Ubuntu

## Build

    make               # native NEON/SSE2 warp, 2x2 average debayer
    make OPENCV=1      # OpenCV bilinear warp and debayer (better quality)

Or manually (native):

    c++ -std=c++17 -O2 -march=native -I donuts \
        fits_stack.cpp donuts/donuts.cpp \
        -lcfitsio -lpng \
        -o fits_stack

With OpenCV:

    c++ -std=c++17 -O2 -march=native -I donuts -DHAVE_OPENCV \
        fits_stack.cpp donuts/donuts.cpp \
        -lcfitsio -lpng $(pkg-config --cflags --libs opencv4) \
        -o fits_stack

## Usage

    ./fits_stack [-o output.fits] [-p output.png]
                 [-m mean|median|sigmaclip|wstream] [-k kappa] [-j threads]
                 [-w N] frame1.fits frame2.fits ...

Inputs are sorted alphabetically; the first sorted frame is the alignment
reference and is always included in the stack.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o` | stacked.fits | Output FITS path |
| `-p` |  | Output PNG path (asinh stretch preview) |
| `-m` | mean | Combine method: `mean`, `median`, `sigmaclip`, `wstream` |
| `-k` | 3.0 | Sigma-clip kappa (used by `sigmaclip` and `wstream`) |
| `-j` | all cores | Thread count (0 = hardware concurrency) |
| `-w` | 5 | Warm-up frame count for `wstream` (0 = disable) |

### Combine methods

- **mean** -- simple mean with per-pixel coverage count. Fast, no rejection.
- **median** -- per-pixel median. Robust; requires O(frames * pixels) memory.
- **sigmaclip** -- per-pixel MAD sigma-clip. Most robust; same memory as median.
- **wstream** -- Welford online algorithm. O(pixels) constant memory regardless
  of frame count. Ideal for long EAA sessions. Uses a per-pixel G-channel
  rejection gate (kappa * sigma) to suppress satellite trails and airplane
  streaks. The `-w N` warm-up buffer seeds the Welford baseline from a clean
  N-frame sigmaclip pass, eliminating the first-frame vulnerability.

### Examples

    # Stack all frames in a directory, mean combine
    ./fits_stack lights/*.fits

    # Streaming mode with warm-up, save PNG preview
    ./fits_stack -m wstream -w 5 -o stack.fits -p stack.png lights/*.fits

    # Median stack, 4 threads
    ./fits_stack -m median -j 4 lights/*.fits

## Algorithm notes

- Alignment uses 4-quadrant 1-D phase-only cross-correlation to solve for
  translation (dx, dy) and rotation (dtheta). Inspired by McCormac et al. 2013,
  PASP 125, 548 (https://arxiv.org/abs/1304.2405) with modifications: 4-quadrant
  split, phase-only correlation (Kuglin & Hines 1975), and a WLS rotation solver.
- Frame-level rejection: hard bounds on rotation (5 deg) and translation
  (150 px half-res), followed by MAD sigma-clip on the measured transforms.
- Bicubic (Catmull-Rom) resampling with a NEON/SSE2 fast path for interior
  pixels. When built with `OPENCV=1`, uses `cv::warpAffine(INTER_CUBIC)` instead.
- Colour frames are Bayer-demosaiced at half resolution for alignment;
  all four Bayer patterns (RGGB, GRBG, GBRG, BGGR) are supported.
  With `OPENCV=1`, uses `cv::cvtColor` bilinear demosaic + INTER_AREA
  downsample instead of simple 2x2 cell averaging.

## Source layout

    fits_stack.cpp          -- main program
    donuts/
      donuts.h              -- public API
      donuts.cpp            -- implementation
      pocketfft_hdronly.h   -- FFT engine (C) 2010-2024 Max-Planck-Society

## License

BSD-3-Clause. See `LICENSE` and the SPDX headers in each source file.
`donuts/pocketfft_hdronly.h` is also BSD-3-Clause (Copyright 2010-2024 Max-Planck-Society).
