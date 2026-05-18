/*
 * SPDX-FileCopyrightText: 2026 Christian Kemper <ckemper@gmail.com>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone FITS stacker using the DONUTS rotation/translation solver.
 *
 * Monochrome: aligns and stacks directly.
 * Bayer (BAYERPAT=RGGB): extracts green at half resolution for DONUTS
 * registration, then extracts R, G1, G2, B subplanes (each at half-res),
 * warps all 4 subplanes, stacks them, and applies Malvar-He-Cutler 2004
 * demosaicing post-stack to produce a 3-plane FITS (R/G/B) at half resolution.
 *
 * Alignment: Catmull-Rom bicubic with clamped boundaries.
 *
 * Frame rejection (two-pass):
 *   1. Hard bounds: |dtheta| <= 5 deg, |dx|/|dy| <= 150 half-res px.
 *   2. MAD sigma-clip (3-sigma) on dx, dy, dtheta of hard-bound survivors.
 *
 * Per-pixel coverage count so zero-padded border regions don't dilute edges.
 *
 * Performance optimizations vs v1:
 *   - float pixel buffers (halves memory, enables 4-wide ARM NEON SIMD)
 *   - 3-channel fused bicubic warp: coordinate math amortized over R/G/B
 *   - ARM NEON fast path for interior pixels (no per-lane bounds clamping)
 *   - Per-row interior x-range precomputed analytically to split fast/slow paths
 *   - Thread-parallel warp, accumulate, and reduce via std::thread (-j N)
 *   - Per-thread scratch buffer eliminates per-pixel malloc in median/sigmaclip
 *   - Per-pixel median+MAD sigma-clip (robust to satellites/airplane trails)
 *   - uint8_t mask (no bit-pack overhead of vector<bool>)
 *   - Welford streaming mode (-m wstream): O(pixels) memory regardless of
 *     frame count; per-pixel online rejection gate (kappa*sigma) using
 *     Welford's numerically stable recurrence. Ideal for long EAA sessions.
 *
 * Build (from repo root):
 *   c++ -std=c++17 -O2 -march=native \
 *       -I kstars/ekos/guide/donuts \
 *       -I /opt/homebrew/include \
 *       Tests/ekos/guide/fits_stack.cpp \
 *       kstars/ekos/guide/donuts/donuts.cpp \
 *       /opt/homebrew/lib/libcfitsio.dylib \
 *       /opt/homebrew/lib/libpng.dylib \
 *       -o Tests/ekos/guide/fits_stack
 *
 * Usage:
 *   ./fits_stack [-o output.fits] [-p output.png]
 *               [-m mean|median|sigmaclip|wstream] [-k kappa] [-j threads]
 *               [-w N] frame1.fits frame2.fits ...
 *
 *   -w N  Warm-up buffer for wstream mode: collect the first N accepted frames
 *         into a sigmaclip buffer, then seed the Welford {mean, M2} state from
 *         the clean stack before switching to online streaming. Eliminates the
 *         first-frame vulnerability where a satellite baked into frame 1 or 2
 *         skews the Welford baseline. Default N=5; -w 0 disables.
 *
 *   Inputs are sorted alphabetically; the first is the reference.
 *   Default output: stacked.fits.  Default mode: mean.
 *   -j 0 (or omitted) uses std::thread::hardware_concurrency().
 */

#include "donuts.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fitsio.h>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <png.h>
#include <string>
#include <thread>
#include <vector>

#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
// Horizontal sum of 4 floats.
// AArch64 has vaddvq_f32; ARMv7 NEON needs a two-step reduction.
static inline float hsum_neon(float32x4_t v)
{
#ifdef __aarch64__
    return vaddvq_f32(v);
#else
    float32x2_t t = vadd_f32(vget_high_f32(v), vget_low_f32(v));
    return vget_lane_f32(vpadd_f32(t, t), 0);
#endif
}
#endif

#ifdef __SSE2__
#include <immintrin.h>
// Horizontal sum of 4 floats. SSE3 hadd is two instructions; SSE2-only
// fallback uses shuffles (for targets without -msse3).
static inline float hsum_ps(__m128 v)
{
#ifdef __SSE3__
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
#else
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    v           = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, v);
    v           = _mm_add_ss(v, shuf);
#endif
    return _mm_cvtss_f32(v);
}
#endif

using namespace std;

using Pix = float;

// ---------------------------------------------------------------------------
// Rejection thresholds
// ---------------------------------------------------------------------------

constexpr double MAX_ROTATION_RAD   = 5.0 * M_PI / 180.0;
constexpr double MAX_TRANSLATION_PX = 150.0;
constexpr double MAD_NSIGMA         = 3.0;

// ---------------------------------------------------------------------------
// Stacking mode
// ---------------------------------------------------------------------------

enum class StackMode { Mean, Median, SigmaClip, WelfordStream };

// ---------------------------------------------------------------------------
// Backend selection (runtime; OpenCV path only available when HAVE_OPENCV is set)
// ---------------------------------------------------------------------------

enum class Backend { Native, OpenCV };

// ---------------------------------------------------------------------------
// Thread pool: partition [0, n) into nThreads contiguous blocks.
// ---------------------------------------------------------------------------

static void parallelFor(int n, int nThreads, function<void(int, int)> fn)
{
    if (nThreads <= 1 || n <= nThreads)
    {
        fn(0, n);
        return;
    }
    vector<thread> threads;
    threads.reserve(nThreads);
    int chunk = (n + nThreads - 1) / nThreads;
    for (int t = 0; t < nThreads; ++t)
    {
        int lo = t * chunk, hi = min(lo + chunk, n);
        if (lo >= n) break;
        threads.emplace_back(fn, lo, hi);
    }
    for (auto &t : threads) t.join();
}

// ---------------------------------------------------------------------------
// FITS I/O
// ---------------------------------------------------------------------------

static vector<Pix> loadFITS(const string &path, int &w, int &h)
{
    fitsfile *fptr = nullptr;
    int s = 0;
    if (fits_open_file(&fptr, path.c_str(), READONLY, &s))
    {
        cerr << "Cannot open " << path << " (cfitsio status " << s << ")\n";
        return {};
    }
    long naxes[2] = {};
    fits_get_img_size(fptr, 2, naxes, &s);
    w = (int)naxes[0];
    h = (int)naxes[1];
    // cfitsio applies BZERO/BSCALE correctly for TFLOAT; no double intermediate needed.
    vector<Pix> buf((size_t)w * h);
    fits_read_img(fptr, TFLOAT, 1, (long)w * h, nullptr, buf.data(), nullptr, &s);
    fits_close_file(fptr, &s);
    if (s) { cerr << "Read error on " << path << " (status " << s << ")\n"; return {}; }
    return buf;
}

// Returns the trimmed BAYERPAT string, or "" if the key is absent/unreadable.
static string readBayerPat(const string &path)
{
    fitsfile *fptr = nullptr;
    int s = 0;
    if (fits_open_file(&fptr, path.c_str(), READONLY, &s)) return "";
    char value[FLEN_VALUE] = {};
    fits_read_key(fptr, TSTRING, "BAYERPAT", value, nullptr, &s);
    fits_close_file(fptr, &s);
    if (s != 0) return "";
    string pat(value);
    pat.erase(0, pat.find_first_not_of(" '\""));
    pat.erase(pat.find_last_not_of(" '\"") + 1);
    return pat;
}

// Pixel positions within a 2x2 Bayer cell, encoded as 0-3:
//   0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
// row_offset = pos >> 1,  col_offset = pos & 1
struct BayerLayout { int r, g1, g2, b; };

static bool parseBayerLayout(const string &pat, BayerLayout &bl)
{
    if      (pat == "RGGB") { bl = {0, 1, 2, 3}; return true; }
    else if (pat == "GRBG") { bl = {1, 0, 3, 2}; return true; }
    else if (pat == "GBRG") { bl = {2, 0, 3, 1}; return true; }
    else if (pat == "BGGR") { bl = {3, 1, 2, 0}; return true; }
    return false;
}

static bool writeFITS(const string &path, const vector<Pix> &pixels, int w, int h)
{
    fitsfile *fptr = nullptr;
    int s = 0;
    fits_create_file(&fptr, ("!" + path).c_str(), &s);
    long naxes[2] = { (long)w, (long)h };
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &s);
    fits_write_img(fptr, TFLOAT, 1, (long)w * h,
                   const_cast<Pix *>(pixels.data()), &s);
    fits_close_file(fptr, &s);
    if (s) { cerr << "Write error on " << path << " (status " << s << ")\n"; return false; }
    return true;
}

static bool writeFITS3(const string &path,
                       const vector<Pix> &r, const vector<Pix> &g, const vector<Pix> &b,
                       int w, int h)
{
    fitsfile *fptr = nullptr;
    int s = 0;
    fits_create_file(&fptr, ("!" + path).c_str(), &s);
    long naxes[3] = { (long)w, (long)h, 3 };
    fits_create_img(fptr, FLOAT_IMG, 3, naxes, &s);
    long npix = (long)w * h;
    fits_write_img(fptr, TFLOAT,           1, npix, const_cast<Pix *>(r.data()), &s);
    fits_write_img(fptr, TFLOAT,   npix + 1, npix, const_cast<Pix *>(g.data()), &s);
    fits_write_img(fptr, TFLOAT, 2*npix + 1, npix, const_cast<Pix *>(b.data()), &s);
    fits_close_file(fptr, &s);
    if (s) { cerr << "Write error on " << path << " (status " << s << ")\n"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// PNG output: median background subtraction + linked luminance asinh stretch
// ---------------------------------------------------------------------------

static vector<Pix> subtractMedian(vector<Pix> ch)
{
    vector<Pix> s(ch);
    auto mid = s.begin() + s.size() / 2;
    std::nth_element(s.begin(), mid, s.end());
    Pix med = *mid;
    for (Pix &v : ch) v -= med;
    return ch;
}

static vector<uint8_t> asinhStretch(const vector<Pix> &ch, double lo, double hi)
{
    double softening = 0.05 * (hi - lo);
    if (softening < 1.0) softening = 1.0;
    double norm = std::asinh((hi - lo) / softening);
    vector<uint8_t> out(ch.size());
    for (size_t i = 0; i < ch.size(); ++i)
    {
        double v = std::asinh(((double)ch[i] - lo) / softening) / norm * 255.0;
        out[i] = (uint8_t)std::max(0.0, std::min(255.0, v));
    }
    return out;
}

static bool writePNG(const string &path,
                     const vector<Pix> &r, const vector<Pix> &g, const vector<Pix> &b,
                     int w, int h)
{
    auto rn = subtractMedian(r);
    auto gn = subtractMedian(g);
    auto bn = subtractMedian(b);

    vector<Pix> lum(rn.size());
    for (size_t i = 0; i < lum.size(); ++i)
        lum[i] = 0.299f * rn[i] + 0.587f * gn[i] + 0.114f * bn[i];

    vector<Pix> slum(lum);
    std::sort(slum.begin(), slum.end());
    double lo = slum[(size_t)(0.005 * (slum.size() - 1))];
    double hi = slum[(size_t)(0.995 * (slum.size() - 1))];

    auto r8 = asinhStretch(rn, lo, hi);
    auto g8 = asinhStretch(gn, lo, hi);
    auto b8 = asinhStretch(bn, lo, hi);

    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) { cerr << "Cannot write PNG: " << path << "\n"; return false; }

    png_structp png  = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop   info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(fp); return false; }

    png_init_io(png, fp);
    png_set_IHDR(png, info, (png_uint_32)w, (png_uint_32)h,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    vector<uint8_t> row((size_t)w * 3);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            size_t i = (size_t)y * w + x;
            row[x * 3]     = r8[i];
            row[x * 3 + 1] = g8[i];
            row[x * 3 + 2] = b8[i];
        }
        png_write_row(png, row.data());
    }
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return true;
}

// Donuts::Guider takes const double*; convert float buffer for registration.
static vector<double> toDouble(const vector<Pix> &v)
{
    vector<double> d(v.size());
    for (size_t i = 0; i < v.size(); ++i) d[i] = v[i];
    return d;
}

// ---------------------------------------------------------------------------
// Bayer helpers (RGGB, XBAYROFF=0 YBAYROFF=0)
// ---------------------------------------------------------------------------

static vector<Pix> extractGreen(const vector<Pix> &bayer, int bw, int bh,
                                int &gw, int &gh, const BayerLayout &bl)
{
    gw = bw / 2;
    gh = bh / 2;
    vector<Pix> green((size_t)gw * gh);
    const int g1r = bl.g1 >> 1, g1c = bl.g1 & 1;
    const int g2r = bl.g2 >> 1, g2c = bl.g2 & 1;
    for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c)
        {
            Pix g1 = bayer[(size_t)(2*r + g1r) * bw + (2*c + g1c)];
            Pix g2 = bayer[(size_t)(2*r + g2r) * bw + (2*c + g2c)];
            green[(size_t)r * gw + c] = (g1 + g2) * 0.5f;
        }
    return green;
}

static void extractSubplanes(const vector<Pix> &bayer, int bw, int bh,
    vector<Pix> &R, vector<Pix> &G1, vector<Pix> &G2, vector<Pix> &B,
    int &w, int &h, const BayerLayout &bl)
{
    w = bw / 2; h = bh / 2;
    size_t npix = (size_t)w * h;
    R.resize(npix); G1.resize(npix); G2.resize(npix); B.resize(npix);
    const int rr  = bl.r  >> 1, rc  = bl.r  & 1;
    const int g1r = bl.g1 >> 1, g1c = bl.g1 & 1;
    const int g2r = bl.g2 >> 1, g2c = bl.g2 & 1;
    const int br  = bl.b  >> 1, bc  = bl.b  & 1;
    for (int row = 0; row < h; ++row)
        for (int col = 0; col < w; ++col)
        {
            size_t i = (size_t)row * w + col;
            R [i] = bayer[(size_t)(2*row + rr)  * bw + (2*col + rc )];
            G1[i] = bayer[(size_t)(2*row + g1r) * bw + (2*col + g1c)];
            G2[i] = bayer[(size_t)(2*row + g2r) * bw + (2*col + g2c)];
            B [i] = bayer[(size_t)(2*row + br)  * bw + (2*col + bc )];
        }
}

// OpenCV bilinear demosaic + 2x2 INTER_AREA downsample to half resolution.
// Only compiled when HAVE_OPENCV is defined.
#ifdef HAVE_OPENCV
static void debayerOpenCV(const vector<Pix> &bayer, int bw, int bh,
    vector<Pix> &r, vector<Pix> &g, vector<Pix> &b,
    int &w, int &h, const BayerLayout &bl, const string &pat)
{
    w = bw / 2; h = bh / 2;
    size_t npix = (size_t)w * h;
    r.resize(npix); g.resize(npix); b.resize(npix);

    int cvCode = -1;
    if      (pat == "RGGB") cvCode = cv::COLOR_BayerRG2BGR;
    else if (pat == "GRBG") cvCode = cv::COLOR_BayerGR2BGR;
    else if (pat == "GBRG") cvCode = cv::COLOR_BayerGB2BGR;
    else if (pat == "BGGR") cvCode = cv::COLOR_BayerBG2BGR;

    if (cvCode < 0) return;

    cv::Mat src(bh, bw, CV_32F, const_cast<Pix *>(bayer.data()));
    double minVal, maxVal;
    cv::minMaxIdx(src, &minVal, &maxVal);
    if (maxVal <= minVal) maxVal = minVal + 1.0;
    cv::Mat src16;
    src.convertTo(src16, CV_16U, 65535.0 / (maxVal - minVal),
                  -minVal * 65535.0 / (maxVal - minVal));
    cv::Mat bgr16;
    cv::cvtColor(src16, bgr16, cvCode);
    cv::Mat bgr16half;
    cv::resize(bgr16, bgr16half, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    vector<cv::Mat> chans(3);
    cv::split(bgr16half, chans);   // chans[0]=B [1]=G [2]=R
    double invScale = (maxVal - minVal) / 65535.0;
    for (int c = 0; c < 3; ++c)
        chans[c].convertTo(chans[c], CV_32F, invScale, minVal);
    std::copy((Pix*)chans[2].datastart, (Pix*)chans[2].dataend, r.begin());
    std::copy((Pix*)chans[1].datastart, (Pix*)chans[1].dataend, g.begin());
    std::copy((Pix*)chans[0].datastart, (Pix*)chans[0].dataend, b.begin());
    (void)bl;
}

// Reassemble stacked subplanes into a full-res Bayer mosaic, apply OpenCV
// bilinear demosaic, then INTER_AREA downsample back to half resolution.
static void debayerOpenCVPostStack(
    const vector<Pix> &R, const vector<Pix> &G1,
    const vector<Pix> &G2, const vector<Pix> &B,
    int w, int h, const BayerLayout &bl, const string &pat,
    vector<Pix> &outR, vector<Pix> &outG, vector<Pix> &outB)
{
    size_t npix = (size_t)w * h;
    outR.resize(npix); outG.resize(npix); outB.resize(npix);

    int cvCode = -1;
    if      (pat == "RGGB") cvCode = cv::COLOR_BayerRG2BGR;
    else if (pat == "GRBG") cvCode = cv::COLOR_BayerGR2BGR;
    else if (pat == "GBRG") cvCode = cv::COLOR_BayerGB2BGR;
    else if (pat == "BGGR") cvCode = cv::COLOR_BayerBG2BGR;
    if (cvCode < 0) return;

    // Assemble full-res Bayer from stacked half-res subplanes.
    const int bw = w * 2, bh = h * 2;
    vector<Pix> bayer((size_t)bw * bh);
    const int rr  = bl.r  >> 1, rc  = bl.r  & 1;
    const int g1r = bl.g1 >> 1, g1c = bl.g1 & 1;
    const int g2r = bl.g2 >> 1, g2c = bl.g2 & 1;
    const int br  = bl.b  >> 1, bc  = bl.b  & 1;
    for (int row = 0; row < h; ++row)
        for (int col = 0; col < w; ++col)
        {
            size_t i = (size_t)row * w + col;
            bayer[(size_t)(2*row + rr ) * bw + (2*col + rc )] = R [i];
            bayer[(size_t)(2*row + g1r) * bw + (2*col + g1c)] = G1[i];
            bayer[(size_t)(2*row + g2r) * bw + (2*col + g2c)] = G2[i];
            bayer[(size_t)(2*row + br ) * bw + (2*col + bc )] = B [i];
        }

    // Normalize to uint16 for cvtColor, demosaic, INTER_AREA downsample.
    cv::Mat src(bh, bw, CV_32F, bayer.data());
    double minVal, maxVal;
    cv::minMaxIdx(src, &minVal, &maxVal);
    if (maxVal <= minVal) maxVal = minVal + 1.0;
    cv::Mat src16;
    src.convertTo(src16, CV_16U, 65535.0 / (maxVal - minVal),
                  -minVal * 65535.0 / (maxVal - minVal));
    cv::Mat bgr16;
    cv::cvtColor(src16, bgr16, cvCode);
    cv::Mat bgr16half;
    cv::resize(bgr16, bgr16half, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    vector<cv::Mat> chans(3);
    cv::split(bgr16half, chans);   // chans[0]=B [1]=G [2]=R
    double invScale = (maxVal - minVal) / 65535.0;
    for (int c = 0; c < 3; ++c)
        chans[c].convertTo(chans[c], CV_32F, invScale, minVal);
    std::copy((Pix*)chans[2].datastart, (Pix*)chans[2].dataend, outR.begin());
    std::copy((Pix*)chans[1].datastart, (Pix*)chans[1].dataend, outG.begin());
    std::copy((Pix*)chans[0].datastart, (Pix*)chans[0].dataend, outB.begin());
}
#endif // HAVE_OPENCV

// ---------------------------------------------------------------------------
// Bicubic helpers
// ---------------------------------------------------------------------------

// Catmull-Rom weights for fractional position t in [0,1).
// w[0]=weight for ix-1, w[1]=ix+0, w[2]=ix+1, w[3]=ix+2.
static inline void cubicWeights4(float t, float w[4])
{
    float t2 = t * t, t3 = t2 * t;
    w[0] = -0.5f*t3 + t2 - 0.5f*t;
    w[1] =  1.5f*t3 - 2.5f*t2 + 1.0f;
    w[2] = -1.5f*t3 + 2.0f*t2 + 0.5f*t;
    w[3] =  0.5f*t3 - 0.5f*t2;
}

// Per-row x range where all 16 source pixels in the 4x4 kernel are in-bounds.
// Source mapping: sx(x) = dsx_dx*x + sx0,  sy(x) = dsy_dx*x + sy0  (linear in x).
// These are just the first column of the affine matrix (m.a, m.c) and the
// per-row offsets (m.b*y + m.tx, m.d*y + m.ty).
// Safe condition: ix in [1, w-3] and iy in [1, h-3].
// Returns [xlo, xhi] inclusive; xhi < xlo means the whole row needs clamping.
static pair<int,int> interiorXRange(double sx0, double sy0,
                                    double dsx_dx, double dsy_dx,
                                    int w, int h)
{
    double lo = 0.0, hi = (double)(w - 1);

    // sx constraint: dsx_dx*x + sx0 in [1, w-2).
    // Epsilon is subtracted in *source* space so it correctly shrinks the safe
    // dest-x range regardless of slope sign (negative slope inverts the mapping).
    if (std::abs(dsx_dx) > 1e-12)
    {
        double a = (1.0         - sx0) / dsx_dx;
        double b = (w-2.0-1e-9  - sx0) / dsx_dx;
        if (dsx_dx > 0) { lo = std::max(lo, a); hi = std::min(hi, b); }
        else            { lo = std::max(lo, b); hi = std::min(hi, a); }
    }
    else if (sx0 < 1.0 || sx0 >= w - 2.0) return {0, -1};

    // sy constraint: dsy_dx*x + sy0 in [1, h-2).
    if (std::abs(dsy_dx) > 1e-12)
    {
        double a = (1.0         - sy0) / dsy_dx;
        double b = (h-2.0-1e-9  - sy0) / dsy_dx;
        if (dsy_dx > 0) { lo = std::max(lo, a); hi = std::min(hi, b); }
        else            { lo = std::max(lo, b); hi = std::min(hi, a); }
    }
    else if (sy0 < 1.0 || sy0 >= h - 2.0) return {0, -1};

    int ilo = std::max(0,   (int)std::ceil(lo));
    int ihi = std::min(w-1, (int)hi);
    return (ilo <= ihi) ? make_pair(ilo, ihi) : make_pair(0, -1);
}

// ---------------------------------------------------------------------------
// Fused 3-channel bicubic warp using a precomputed AffineMatrix.
// Processes dst rows [ylo, yhi) for parallel dispatch.
// Pass t.alignmentMatrix(w, h) to undo a measured DONUTS transform.
// ---------------------------------------------------------------------------

static void alignFrame3rows(
    const vector<Pix> &srcR, const vector<Pix> &srcG, const vector<Pix> &srcB,
    int w, int h,
    const Donuts::AffineMatrix &m,
    vector<Pix> &dstR, vector<Pix> &dstG, vector<Pix> &dstB,
    vector<uint8_t> &mask,
    int ylo, int yhi)
{
    for (int y = ylo; y < yhi; ++y)
    {
        // Per-row offsets: sx(x) = m.a*x + sx0,  sy(x) = m.c*x + sy0
        const double sx0 = m.b * y + m.tx;
        const double sy0 = m.d * y + m.ty;

        auto [xInLo, xInHi] = interiorXRange(sx0, sy0, m.a, m.c, w, h);

        // Split the row into three segments to hoist the fast/slow branch out of
        // the inner loop: left border [0, xInLo), interior [xInLo, xInHi+1), right border.
        // The interior is the vast majority of pixels at typical rotation angles.

        // Helper: slow (clamped) path for a single pixel.
        auto slowPixel = [&](int x)
        {
            const double sx = m.a * x + sx0;
            const double sy = m.c * x + sy0;
            const size_t k  = (size_t)y * w + x;
            if (sx < 0.0 || sx >= w || sy < 0.0 || sy >= h) { mask[k] = 0; return; }
            const int   ix = (int)sx, iy = (int)sy;
            const float fx = (float)(sx - ix), fy = (float)(sy - iy);
            float wx[4], wy[4];
            cubicWeights4(fx, wx);
            cubicWeights4(fy, wy);
            mask[k] = 1;
            float vR = 0.0f, vG = 0.0f, vB = 0.0f;
            for (int dr = -1; dr <= 2; ++dr)
            {
                const int   py  = std::max(0, std::min(h - 1, iy + dr));
                const float wyd = wy[dr + 1];
                for (int dc = -1; dc <= 2; ++dc)
                {
                    const int    px  = std::max(0, std::min(w - 1, ix + dc));
                    const float  wdc = wx[dc + 1] * wyd;
                    const size_t s   = (size_t)py * w + px;
                    vR += wdc * srcR[s];
                    vG += wdc * srcG[s];
                    vB += wdc * srcB[s];
                }
            }
            dstR[k] = vR; dstG[k] = vG; dstB[k] = vB;
        };

        // Left border.
        for (int x = 0; x < xInLo; ++x) slowPixel(x);

        // Interior: no bounds clamping needed.
        for (int x = xInLo; x <= xInHi; ++x)
        {
            const double sx = m.a * x + sx0;
            const double sy = m.c * x + sy0;
            const size_t k  = (size_t)y * w + x;
            const int   ix = (int)sx, iy = (int)sy;
            const float fx = (float)(sx - ix), fy = (float)(sy - iy);
            float wx[4], wy[4];
            cubicWeights4(fx, wx);
            cubicWeights4(fy, wy);
            mask[k] = 1;
#ifdef __ARM_NEON
            // Accumulate 4-wide vectors over the 4 rows; one vaddvq_f32 per
            // channel at the end instead of 4 -- saves 9 horizontal reductions.
            const float32x4_t wxv = vld1q_f32(wx);
            float32x4_t accVR = vdupq_n_f32(0.0f);
            float32x4_t accVG = vdupq_n_f32(0.0f);
            float32x4_t accVB = vdupq_n_f32(0.0f);
            for (int dr = 0; dr < 4; ++dr)
            {
                const size_t      base = (size_t)(iy + dr - 1) * w + (ix - 1);
                const float32x4_t wyd  = vdupq_n_f32(wy[dr]);
                accVR = vmlaq_f32(accVR, vmulq_f32(vld1q_f32(srcR.data() + base), wxv), wyd);
                accVG = vmlaq_f32(accVG, vmulq_f32(vld1q_f32(srcG.data() + base), wxv), wyd);
                accVB = vmlaq_f32(accVB, vmulq_f32(vld1q_f32(srcB.data() + base), wxv), wyd);
            }
            dstR[k] = hsum_neon(accVR);
            dstG[k] = hsum_neon(accVG);
            dstB[k] = hsum_neon(accVB);
#elif defined(__SSE2__)
            // Same 4x4 kernel using SSE2 128-bit float vectors.
            // wxw = combined x*y weights; one _mm_loadu_ps + _mm_mul_ps per channel per row.
            const __m128 wxv = _mm_loadu_ps(wx);
            __m128 accVR = _mm_setzero_ps();
            __m128 accVG = _mm_setzero_ps();
            __m128 accVB = _mm_setzero_ps();
            for (int dr = 0; dr < 4; ++dr)
            {
                const size_t base = (size_t)(iy + dr - 1) * w + (ix - 1);
                const __m128 wxw  = _mm_mul_ps(wxv, _mm_set1_ps(wy[dr]));
                accVR = _mm_add_ps(accVR, _mm_mul_ps(_mm_loadu_ps(srcR.data() + base), wxw));
                accVG = _mm_add_ps(accVG, _mm_mul_ps(_mm_loadu_ps(srcG.data() + base), wxw));
                accVB = _mm_add_ps(accVB, _mm_mul_ps(_mm_loadu_ps(srcB.data() + base), wxw));
            }
            dstR[k] = hsum_ps(accVR);
            dstG[k] = hsum_ps(accVG);
            dstB[k] = hsum_ps(accVB);
#else
            float vR = 0.0f, vG = 0.0f, vB = 0.0f;
            for (int dr = 0; dr < 4; ++dr)
            {
                const size_t base = (size_t)(iy + dr - 1) * w + (ix - 1);
                const float  wyd  = wy[dr];
                for (int dc = 0; dc < 4; ++dc)
                {
                    const float  wdc = wx[dc] * wyd;
                    const size_t s   = base + dc;
                    vR += wdc * srcR[s];
                    vG += wdc * srcG[s];
                    vB += wdc * srcB[s];
                }
            }
            dstR[k] = vR; dstG[k] = vG; dstB[k] = vB;
#endif
        }

        // Right border.
        for (int x = xInHi + 1; x < w; ++x) slowPixel(x);
    }
}

// ---------------------------------------------------------------------------
// Warp dispatcher: OpenCV warpAffine or our NEON/SSE2 bicubic.
// Same signature; chosen at compile time via HAVE_OPENCV.
// ---------------------------------------------------------------------------

static void alignFrame3(
    const vector<Pix> &srcR, const vector<Pix> &srcG, const vector<Pix> &srcB,
    int w, int h,
    const Donuts::AffineMatrix &m,
    vector<Pix> &dstR, vector<Pix> &dstG, vector<Pix> &dstB,
    vector<uint8_t> &mask,
    int nThreads,
    Backend backend = Backend::Native)
{
#ifdef HAVE_OPENCV
    // m maps destination pixel -> source pixel (inverse warp convention).
    // WARP_INVERSE_MAP tells OpenCV to interpret M the same way.
    if (backend == Backend::OpenCV)
    {
        cv::Mat M_cv = (cv::Mat_<double>(2, 3) << m.a, m.b, m.tx, m.c, m.d, m.ty);
        cv::Size sz(w, h);
        const int flags = cv::INTER_CUBIC | cv::WARP_INVERSE_MAP;
        cv::Mat cvR(h, w, CV_32F, const_cast<Pix *>(srcR.data()));
        cv::Mat cvG(h, w, CV_32F, const_cast<Pix *>(srcG.data()));
        cv::Mat cvB(h, w, CV_32F, const_cast<Pix *>(srcB.data()));
        cv::Mat outR(h, w, CV_32F, dstR.data());
        cv::Mat outG(h, w, CV_32F, dstG.data());
        cv::Mat outB(h, w, CV_32F, dstB.data());
        cv::warpAffine(cvR, outR, M_cv, sz, flags, cv::BORDER_CONSTANT, 0);
        cv::warpAffine(cvG, outG, M_cv, sz, flags, cv::BORDER_CONSTANT, 0);
        cv::warpAffine(cvB, outB, M_cv, sz, flags, cv::BORDER_CONSTANT, 0);
        cv::Mat ones = cv::Mat::ones(h, w, CV_32F);
        cv::Mat warpedOnes;
        cv::warpAffine(ones, warpedOnes, M_cv, sz,
                       cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 0);
        const float *mp = warpedOnes.ptr<float>();
        for (int k = 0; k < w * h; ++k) mask[k] = mp[k] > 0.5f ? 1 : 0;
        return;
    }
#else
    (void)backend;
#endif
    parallelFor(h, nThreads, [&](int ylo, int yhi)
    {
        alignFrame3rows(srcR, srcG, srcB, w, h, m, dstR, dstG, dstB, mask, ylo, yhi);
    });
}

// ---------------------------------------------------------------------------
// Fused 4-channel bicubic warp (R, G1, G2, B subplanes).
// ---------------------------------------------------------------------------

static void alignFrame4rows(
    const vector<Pix> &srcR, const vector<Pix> &srcG1,
    const vector<Pix> &srcG2, const vector<Pix> &srcB,
    int w, int h,
    const Donuts::AffineMatrix &m,
    vector<Pix> &dstR, vector<Pix> &dstG1,
    vector<Pix> &dstG2, vector<Pix> &dstB,
    vector<uint8_t> &mask,
    int ylo, int yhi)
{
    for (int y = ylo; y < yhi; ++y)
    {
        const double sx0 = m.b * y + m.tx;
        const double sy0 = m.d * y + m.ty;

        auto [xInLo, xInHi] = interiorXRange(sx0, sy0, m.a, m.c, w, h);

        auto slowPixel = [&](int x)
        {
            const double sx = m.a * x + sx0;
            const double sy = m.c * x + sy0;
            const size_t k  = (size_t)y * w + x;
            if (sx < 0.0 || sx >= w || sy < 0.0 || sy >= h) { mask[k] = 0; return; }
            const int   ix = (int)sx, iy = (int)sy;
            const float fx = (float)(sx - ix), fy = (float)(sy - iy);
            float wx[4], wy[4];
            cubicWeights4(fx, wx);
            cubicWeights4(fy, wy);
            mask[k] = 1;
            float vR = 0.0f, vG1 = 0.0f, vG2 = 0.0f, vB = 0.0f;
            for (int dr = -1; dr <= 2; ++dr)
            {
                const int   py  = std::max(0, std::min(h - 1, iy + dr));
                const float wyd = wy[dr + 1];
                for (int dc = -1; dc <= 2; ++dc)
                {
                    const int    px  = std::max(0, std::min(w - 1, ix + dc));
                    const float  wdc = wx[dc + 1] * wyd;
                    const size_t s   = (size_t)py * w + px;
                    vR  += wdc * srcR [s];
                    vG1 += wdc * srcG1[s];
                    vG2 += wdc * srcG2[s];
                    vB  += wdc * srcB [s];
                }
            }
            dstR[k] = vR; dstG1[k] = vG1; dstG2[k] = vG2; dstB[k] = vB;
        };

        for (int x = 0; x < xInLo; ++x) slowPixel(x);

        for (int x = xInLo; x <= xInHi; ++x)
        {
            const double sx = m.a * x + sx0;
            const double sy = m.c * x + sy0;
            const size_t k  = (size_t)y * w + x;
            const int   ix = (int)sx, iy = (int)sy;
            const float fx = (float)(sx - ix), fy = (float)(sy - iy);
            float wx[4], wy[4];
            cubicWeights4(fx, wx);
            cubicWeights4(fy, wy);
            mask[k] = 1;
#ifdef __ARM_NEON
            const float32x4_t wxv = vld1q_f32(wx);
            float32x4_t accVR  = vdupq_n_f32(0.0f);
            float32x4_t accVG1 = vdupq_n_f32(0.0f);
            float32x4_t accVG2 = vdupq_n_f32(0.0f);
            float32x4_t accVB  = vdupq_n_f32(0.0f);
            for (int dr = 0; dr < 4; ++dr)
            {
                const size_t      base = (size_t)(iy + dr - 1) * w + (ix - 1);
                const float32x4_t wyd  = vdupq_n_f32(wy[dr]);
                accVR  = vmlaq_f32(accVR,  vmulq_f32(vld1q_f32(srcR .data() + base), wxv), wyd);
                accVG1 = vmlaq_f32(accVG1, vmulq_f32(vld1q_f32(srcG1.data() + base), wxv), wyd);
                accVG2 = vmlaq_f32(accVG2, vmulq_f32(vld1q_f32(srcG2.data() + base), wxv), wyd);
                accVB  = vmlaq_f32(accVB,  vmulq_f32(vld1q_f32(srcB .data() + base), wxv), wyd);
            }
            dstR [k] = hsum_neon(accVR);
            dstG1[k] = hsum_neon(accVG1);
            dstG2[k] = hsum_neon(accVG2);
            dstB [k] = hsum_neon(accVB);
#elif defined(__SSE2__)
            const __m128 wxv = _mm_loadu_ps(wx);
            __m128 accVR  = _mm_setzero_ps();
            __m128 accVG1 = _mm_setzero_ps();
            __m128 accVG2 = _mm_setzero_ps();
            __m128 accVB  = _mm_setzero_ps();
            for (int dr = 0; dr < 4; ++dr)
            {
                const size_t base = (size_t)(iy + dr - 1) * w + (ix - 1);
                const __m128 wxw  = _mm_mul_ps(wxv, _mm_set1_ps(wy[dr]));
                accVR  = _mm_add_ps(accVR,  _mm_mul_ps(_mm_loadu_ps(srcR .data() + base), wxw));
                accVG1 = _mm_add_ps(accVG1, _mm_mul_ps(_mm_loadu_ps(srcG1.data() + base), wxw));
                accVG2 = _mm_add_ps(accVG2, _mm_mul_ps(_mm_loadu_ps(srcG2.data() + base), wxw));
                accVB  = _mm_add_ps(accVB,  _mm_mul_ps(_mm_loadu_ps(srcB .data() + base), wxw));
            }
            dstR [k] = hsum_ps(accVR);
            dstG1[k] = hsum_ps(accVG1);
            dstG2[k] = hsum_ps(accVG2);
            dstB [k] = hsum_ps(accVB);
#else
            float vR = 0.0f, vG1 = 0.0f, vG2 = 0.0f, vB = 0.0f;
            for (int dr = 0; dr < 4; ++dr)
            {
                const size_t base = (size_t)(iy + dr - 1) * w + (ix - 1);
                const float  wyd  = wy[dr];
                for (int dc = 0; dc < 4; ++dc)
                {
                    const float  wdc = wx[dc] * wyd;
                    const size_t s   = base + dc;
                    vR  += wdc * srcR [s];
                    vG1 += wdc * srcG1[s];
                    vG2 += wdc * srcG2[s];
                    vB  += wdc * srcB [s];
                }
            }
            dstR [k] = vR; dstG1[k] = vG1; dstG2[k] = vG2; dstB[k] = vB;
#endif
        }

        for (int x = xInHi + 1; x < w; ++x) slowPixel(x);
    }
}

static void alignFrame4(
    const vector<Pix> &srcR, const vector<Pix> &srcG1,
    const vector<Pix> &srcG2, const vector<Pix> &srcB,
    int w, int h,
    const Donuts::AffineMatrix &m,
    vector<Pix> &dstR, vector<Pix> &dstG1,
    vector<Pix> &dstG2, vector<Pix> &dstB,
    vector<uint8_t> &mask,
    int nThreads,
    Backend backend = Backend::Native)
{
#ifdef HAVE_OPENCV
    if (backend == Backend::OpenCV)
    {
        cv::Mat M_cv = (cv::Mat_<double>(2, 3) << m.a, m.b, m.tx, m.c, m.d, m.ty);
        cv::Size sz(w, h);
        const int flags = cv::INTER_CUBIC | cv::WARP_INVERSE_MAP;
        cv::Mat cvR  (h, w, CV_32F, const_cast<Pix *>(srcR .data()));
        cv::Mat cvG1 (h, w, CV_32F, const_cast<Pix *>(srcG1.data()));
        cv::Mat cvG2 (h, w, CV_32F, const_cast<Pix *>(srcG2.data()));
        cv::Mat cvB  (h, w, CV_32F, const_cast<Pix *>(srcB .data()));
        cv::Mat outR (h, w, CV_32F, dstR .data());
        cv::Mat outG1(h, w, CV_32F, dstG1.data());
        cv::Mat outG2(h, w, CV_32F, dstG2.data());
        cv::Mat outB (h, w, CV_32F, dstB .data());
        cv::warpAffine(cvR,  outR,  M_cv, sz, flags, cv::BORDER_CONSTANT, 0);
        cv::warpAffine(cvG1, outG1, M_cv, sz, flags, cv::BORDER_CONSTANT, 0);
        cv::warpAffine(cvG2, outG2, M_cv, sz, flags, cv::BORDER_CONSTANT, 0);
        cv::warpAffine(cvB,  outB,  M_cv, sz, flags, cv::BORDER_CONSTANT, 0);
        cv::Mat ones = cv::Mat::ones(h, w, CV_32F);
        cv::Mat warpedOnes;
        cv::warpAffine(ones, warpedOnes, M_cv, sz,
                       cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 0);
        const float *mp = warpedOnes.ptr<float>();
        for (int k = 0; k < w * h; ++k) mask[k] = mp[k] > 0.5f ? 1 : 0;
        return;
    }
#else
    (void)backend;
#endif
    parallelFor(h, nThreads, [&](int ylo, int yhi)
    {
        alignFrame4rows(srcR, srcG1, srcG2, srcB, w, h, m,
                        dstR, dstG1, dstG2, dstB, mask, ylo, yhi);
    });
}

// ---------------------------------------------------------------------------
// Per-pixel combine helpers -- caller provides scratch[depth]; no malloc here.
// ---------------------------------------------------------------------------

// O(n) median via nth_element.
static Pix pixelMedian(const Pix *vals, Pix *scratch, int n)
{
    std::copy(vals, vals + n, scratch);
    std::nth_element(scratch, scratch + n / 2, scratch + n);
    return scratch[n / 2];
}

// Iterative median+MAD sigma-clip with in-place compaction.
//
// Uses median as the centre and 1.4826*MAD as the scale estimate.
// Mean+stddev fails on satellite/airplane trails: a single pixel at 950
// among background ~12 inflates mean to ~169 and sigma to ~383, so the
// outlier is inside the 3-sigma band (hi = 1317) and survives every pass.
// Median = 12.5 and MAD-sigma = 1.48 clips 950 immediately (hi = 16.9).
//
// devbuf must be caller-allocated with capacity >= n (same size as scratch).
static Pix pixelSigmaClip(const Pix *vals, Pix *scratch, Pix *devbuf,
                           int n, double kappa)
{
    std::copy(vals, vals + n, scratch);
    int cnt = n;
    for (int iter = 0; iter < 5 && cnt > 2; ++iter)
    {
        // Robust centre: median via nth_element (O(n) average).
        std::nth_element(scratch, scratch + cnt/2, scratch + cnt);
        double med = scratch[cnt/2];

        // Robust scale: MAD = median(|x - median|), sigma = 1.4826 * MAD.
        for (int i = 0; i < cnt; ++i) devbuf[i] = std::abs(scratch[i] - (Pix)med);
        std::nth_element(devbuf, devbuf + cnt/2, devbuf + cnt);
        double sigma = 1.4826 * devbuf[cnt/2];
        if (sigma < 1e-10) break;

        double lo = med - kappa * sigma, hi = med + kappa * sigma;
        int keep = 0;
        for (int i = 0; i < cnt; ++i)
            if (scratch[i] >= lo && scratch[i] <= hi)
                scratch[keep++] = scratch[i];
        if (keep == cnt) break;
        cnt = keep;
    }
    double sum = 0.0;
    for (int i = 0; i < cnt; ++i) sum += scratch[i];
    return cnt > 0 ? (Pix)(sum / cnt) : (Pix)vals[0];
}

// Reduce frame-major pixel stacks to final images.
// stk layout: stk[frameIndex][pixelIndex]; NaN marks masked-out pixels.
// Gathering non-NaN values per pixel is strided (one hop per frame) but this
// is a one-time pass so cache miss cost is acceptable.
// Each thread allocates scratch buffers of size nFrames -- no per-pixel malloc.
static void reduceStack4(
    const vector<vector<Pix>> &stkR,
    const vector<vector<Pix>> &stkG1,
    const vector<vector<Pix>> &stkG2,
    const vector<vector<Pix>> &stkB,
    size_t npix,
    StackMode mode, double kappa, int nThreads,
    vector<Pix> &outR, vector<Pix> &outG1,
    vector<Pix> &outG2, vector<Pix> &outB)
{
    const int nFrames = (int)stkR.size();
    outR.resize(npix); outG1.resize(npix); outG2.resize(npix); outB.resize(npix);

    parallelFor((int)npix, nThreads, [&](int klo, int khi)
    {
        vector<Pix> gR(nFrames), gG1(nFrames), gG2(nFrames), gB(nFrames);
        vector<Pix> scratch(nFrames), devbuf(nFrames);
        for (int k = klo; k < khi; ++k)
        {
            // Gather valid (non-NaN) values for this pixel across all frames.
            int n = 0;
            for (int f = 0; f < nFrames; ++f)
            {
                Pix v = stkR[f][k];
                if (std::isnan(v)) continue;
                gR [n] = v;
                gG1[n] = stkG1[f][k];
                gG2[n] = stkG2[f][k];
                gB [n] = stkB [f][k];
                ++n;
            }
            if (n == 0) { outR[k] = outG1[k] = outG2[k] = outB[k] = 0.0f; continue; }
            if (mode == StackMode::Median)
            {
                outR [k] = pixelMedian   (gR .data(), scratch.data(), n);
                outG1[k] = pixelMedian   (gG1.data(), scratch.data(), n);
                outG2[k] = pixelMedian   (gG2.data(), scratch.data(), n);
                outB [k] = pixelMedian   (gB .data(), scratch.data(), n);
            }
            else
            {
                outR [k] = pixelSigmaClip(gR .data(), scratch.data(), devbuf.data(), n, kappa);
                outG1[k] = pixelSigmaClip(gG1.data(), scratch.data(), devbuf.data(), n, kappa);
                outG2[k] = pixelSigmaClip(gG2.data(), scratch.data(), devbuf.data(), n, kappa);
                outB [k] = pixelSigmaClip(gB .data(), scratch.data(), devbuf.data(), n, kappa);
            }
        }
    });
}

// Seed Welford streaming state from a warm-up buffer of N frames.
// Eliminates the first-frame vulnerability: if a satellite trail hits frame 1
// or 2, Welford has no prior variance to reject it. Seeding runs G1-channel
// MAD sigma-clip per pixel across the N warm-up frames, then runs the Welford
// recurrence on the survivors to produce a pristine {mean, M2} baseline.
// After this call, wCount[k] >= 1 for all pixels with at least one valid sample;
// the online streaming gate fires immediately on frame N+1.
static void seedWelford4(
    const vector<vector<Pix>> &warmR,
    const vector<vector<Pix>> &warmG1,
    const vector<vector<Pix>> &warmG2,
    const vector<vector<Pix>> &warmB,
    size_t npix, double kappa, int nThreads,
    vector<Pix>      &wMeanR,  vector<Pix>      &wMeanG1, vector<Pix>      &wMeanG2, vector<Pix>      &wMeanB,
    vector<Pix>      &wM2R,    vector<Pix>      &wM2G1,   vector<Pix>      &wM2G2,   vector<Pix>      &wM2B,
    vector<uint32_t> &wCount)
{
    const int nFrames = (int)warmR.size();
    parallelFor((int)npix, nThreads, [&](int klo, int khi)
    {
        // Per-thread scratch: nFrames entries each -- no per-pixel malloc.
        vector<Pix> vG1(nFrames), vG2(nFrames), vR(nFrames), vB(nFrames);
        vector<Pix> scratch(nFrames), devbuf(nFrames);
        vector<int> survivorIdx(nFrames);
        for (int k = klo; k < khi; ++k)
        {
            // Gather non-NaN samples for this pixel.
            int n = 0;
            for (int f = 0; f < nFrames; ++f)
            {
                Pix g1v = warmG1[f][k];
                if (std::isnan(g1v)) continue;
                vG1[n] = g1v;
                vG2[n] = warmG2[f][k];
                vR [n] = warmR [f][k];
                vB [n] = warmB [f][k];
                survivorIdx[n] = n;
                ++n;
            }
            if (n == 0) { wCount[k] = 0; continue; }
            if (n == 1)
            {
                wMeanR [k] = vR [0]; wMeanG1[k] = vG1[0]; wMeanG2[k] = vG2[0]; wMeanB[k] = vB[0];
                wM2R[k] = 0.0f; wM2G1[k] = 0.0f; wM2G2[k] = 0.0f; wM2B[k] = 0.0f;
                wCount[k] = 1;
                continue;
            }

            // G1-channel MAD sigma-clip: compact scratch[] and survivorIdx[] in parallel.
            std::copy(vG1.begin(), vG1.begin() + n, scratch.begin());
            std::iota(survivorIdx.begin(), survivorIdx.begin() + n, 0);
            int cnt = n;
            for (int iter = 0; iter < 5 && cnt > 2; ++iter)
            {
                std::copy(scratch.begin(), scratch.begin() + cnt, devbuf.begin());
                std::nth_element(devbuf.begin(), devbuf.begin() + cnt/2, devbuf.begin() + cnt);
                double med = devbuf[cnt/2];
                for (int i = 0; i < cnt; ++i) devbuf[i] = std::abs(scratch[i] - (Pix)med);
                std::nth_element(devbuf.begin(), devbuf.begin() + cnt/2, devbuf.begin() + cnt);
                double sigma = 1.4826 * devbuf[cnt/2];
                if (sigma < 1e-10) break;
                double lo = med - kappa * sigma, hi = med + kappa * sigma;
                int keep = 0;
                for (int i = 0; i < cnt; ++i)
                {
                    if (scratch[i] >= (Pix)lo && scratch[i] <= (Pix)hi)
                    {
                        scratch[keep]      = scratch[i];
                        survivorIdx[keep]  = survivorIdx[i];
                        ++keep;
                    }
                }
                if (keep == cnt) break;
                cnt = keep;
            }

            // Welford recurrence over surviving {R, G1, G2, B} quads.
            float mR=0.0f, mG1=0.0f, mG2=0.0f, mB=0.0f;
            float m2R=0.0f, m2G1=0.0f, m2G2=0.0f, m2B=0.0f;
            for (int i = 0; i < cnt; ++i)
            {
                const int j = survivorIdx[i];
                const float nf = (float)(i + 1);
                float d;
                d = vR [j] - mR;  mR  += d / nf; m2R  += d * (vR [j] - mR);
                d = vG1[j] - mG1; mG1 += d / nf; m2G1 += d * (vG1[j] - mG1);
                d = vG2[j] - mG2; mG2 += d / nf; m2G2 += d * (vG2[j] - mG2);
                d = vB [j] - mB;  mB  += d / nf; m2B  += d * (vB [j] - mB);
            }
            wMeanR [k] = mR;  wMeanG1[k] = mG1; wMeanG2[k] = mG2; wMeanB[k] = mB;
            wM2R   [k] = m2R; wM2G1  [k] = m2G1; wM2G2 [k] = m2G2; wM2B[k] = m2B;
            wCount[k] = (uint32_t)cnt;
        }
    });
}

// ---------------------------------------------------------------------------
// Post-stack Malvar-He-Cutler 2004 demosaicing at half resolution.
//
// Each output pixel (oi, oj) represents a 2x2 Bayer block. We compute MHC
// interpolated R, G, B at each of the 4 full-resolution positions within that
// block (R pos, G1 pos, G2 pos, B pos), then average the 4 estimates.
// All array accesses use clamped indices to handle borders without branches.
//
// Bayer layout (RGGB assumed; handled via subplane order from extractSubplanes):
//   R [oi][oj] = bayer[2oi  ][2oj  ]
//   G1[oi][oj] = bayer[2oi  ][2oj+1]
//   G2[oi][oj] = bayer[2oi+1][2oj  ]
//   B [oi][oj] = bayer[2oi+1][2oj+1]
// ---------------------------------------------------------------------------

static void debayerMHC_halfres(
    const vector<Pix> &R, const vector<Pix> &G1,
    const vector<Pix> &G2, const vector<Pix> &B,
    int w, int h,
    vector<Pix> &outR, vector<Pix> &outG, vector<Pix> &outB,
    int nThreads)
{
    size_t npix = (size_t)w * h;
    outR.resize(npix); outG.resize(npix); outB.resize(npix);

    // Clamped accessors into each subplane.
    auto r  = [&](int i, int j) -> float { return R [std::clamp(i,0,h-1)*w + std::clamp(j,0,w-1)]; };
    auto g1 = [&](int i, int j) -> float { return G1[std::clamp(i,0,h-1)*w + std::clamp(j,0,w-1)]; };
    auto g2 = [&](int i, int j) -> float { return G2[std::clamp(i,0,h-1)*w + std::clamp(j,0,w-1)]; };
    auto b  = [&](int i, int j) -> float { return B [std::clamp(i,0,h-1)*w + std::clamp(j,0,w-1)]; };

    parallelFor(h, nThreads, [&](int ylo, int yhi)
    {
        for (int oi = ylo; oi < yhi; ++oi)
            for (int oj = 0; oj < w; ++oj)
            {
                // --- R channel: average R value from all 4 full-res positions ---

                // Position (2oi, 2oj): exact R sample.
                float rR = r(oi, oj);

                // Position (2oi, 2oj+1): G1 site -- K2 interpolates R along row.
                float rGrb = (
                     0.5f * g1(oi-1, oj  ) +
                    -1.0f * g2(oi-1, oj  ) +
                    -1.0f * g2(oi-1, oj+1) +
                    -1.0f * g1(oi,   oj-1) +
                     4.0f * r (oi,   oj  ) +
                     5.0f * g1(oi,   oj  ) +
                     4.0f * r (oi,   oj+1) +
                    -1.0f * g1(oi,   oj+1) +
                    -1.0f * g2(oi,   oj  ) +
                    -1.0f * g2(oi,   oj+1) +
                     0.5f * g1(oi+1, oj  )
                ) * (1.0f / 8.0f);

                // Position (2oi+1, 2oj): G2 site -- K3 interpolates R along column.
                float rGbr = (
                    -1.0f * g2(oi-1, oj  ) +
                    -1.0f * g1(oi,   oj-1) +
                     4.0f * r (oi,   oj  ) +
                    -1.0f * g1(oi,   oj  ) +
                     0.5f * g2(oi,   oj-1) +
                     5.0f * g2(oi,   oj  ) +
                     0.5f * g2(oi,   oj+1) +
                    -1.0f * g1(oi+1, oj-1) +
                     4.0f * r (oi+1, oj  ) +
                    -1.0f * g1(oi+1, oj  ) +
                    -1.0f * g2(oi+1, oj  )
                ) * (1.0f / 8.0f);

                // Position (2oi+1, 2oj+1): B site -- K4 interpolates R diagonally.
                float rB = (
                    -1.5f * b(oi-1, oj  ) +
                     2.0f * r(oi,   oj  ) +
                     2.0f * r(oi,   oj+1) +
                    -1.5f * b(oi,   oj-1) +
                     6.0f * b(oi,   oj  ) +
                    -1.5f * b(oi,   oj+1) +
                     2.0f * r(oi+1, oj  ) +
                     2.0f * r(oi+1, oj+1) +
                    -1.5f * b(oi+1, oj  )
                ) * (1.0f / 8.0f);

                // --- G channel: average G value from all 4 full-res positions ---

                // Position (2oi, 2oj): R site -- K1 interpolates G.
                float gR = (
                    -1.0f * r (oi-1, oj  ) +
                     2.0f * g2(oi-1, oj  ) +
                    -1.0f * r (oi,   oj-1) +
                     2.0f * g1(oi,   oj-1) +
                     4.0f * r (oi,   oj  ) +
                     2.0f * g1(oi,   oj  ) +
                    -1.0f * r (oi,   oj+1) +
                     2.0f * g2(oi,   oj  ) +
                    -1.0f * r (oi+1, oj  )
                ) * (1.0f / 8.0f);

                // Positions (2oi, 2oj+1) and (2oi+1, 2oj): exact G samples.
                float gGrb = g1(oi, oj);
                float gGbr = g2(oi, oj);

                // Position (2oi+1, 2oj+1): B site -- K1 interpolates G.
                float gB = (
                    -1.0f * b(oi-1, oj  ) +
                     2.0f * g1(oi,   oj  ) +
                    -1.0f * b(oi,   oj-1) +
                     2.0f * g2(oi,   oj  ) +
                     4.0f * b(oi,   oj  ) +
                     2.0f * g2(oi,   oj+1) +
                    -1.0f * b(oi,   oj+1) +
                     2.0f * g1(oi+1, oj  ) +
                    -1.0f * b(oi+1, oj  )
                ) * (1.0f / 8.0f);

                // --- B channel: average B value from all 4 full-res positions ---

                // Position (2oi, 2oj): R site -- K4 interpolates B diagonally.
                float bR = (
                    -1.5f * r(oi-1, oj  ) +
                     2.0f * b(oi-1, oj-1) +
                     2.0f * b(oi-1, oj  ) +
                    -1.5f * r(oi,   oj-1) +
                     6.0f * r(oi,   oj  ) +
                    -1.5f * r(oi,   oj+1) +
                     2.0f * b(oi,   oj-1) +
                     2.0f * b(oi,   oj  ) +
                    -1.5f * r(oi+1, oj  )
                ) * (1.0f / 8.0f);

                // Position (2oi, 2oj+1): G1 site -- K3 interpolates B along column.
                float bGrb = (
                    -1.0f * g1(oi-1, oj  ) +
                    -1.0f * g2(oi-1, oj  ) +
                     4.0f * b (oi-1, oj  ) +
                    -1.0f * g2(oi-1, oj+1) +
                     0.5f * g1(oi,   oj-1) +
                     5.0f * g1(oi,   oj  ) +
                     0.5f * g1(oi,   oj+1) +
                    -1.0f * g2(oi,   oj  ) +
                     4.0f * b (oi,   oj  ) +
                    -1.0f * g2(oi,   oj+1) +
                    -1.0f * g1(oi+1, oj  )
                ) * (1.0f / 8.0f);

                // Position (2oi+1, 2oj): G2 site -- K2-variant interpolates B along row.
                float bGbr = (
                     0.5f * g2(oi-1, oj  ) +
                    -1.0f * g1(oi,   oj-1) +
                    -1.0f * g1(oi,   oj  ) +
                    -1.0f * g2(oi,   oj-1) +
                     4.0f * b (oi,   oj-1) +
                     5.0f * g2(oi,   oj  ) +
                     4.0f * b (oi,   oj  ) +
                    -1.0f * g2(oi,   oj+1) +
                    -1.0f * g1(oi+1, oj-1) +
                    -1.0f * g1(oi+1, oj  ) +
                     0.5f * g2(oi+1, oj  )
                ) * (1.0f / 8.0f);

                // Position (2oi+1, 2oj+1): exact B sample.
                float bB = b(oi, oj);

                size_t idx = (size_t)oi * w + oj;
                outR[idx] = (rR + rGrb + rGbr + rB)   * 0.25f;
                outG[idx] = (gR + gGrb + gGbr + gB)   * 0.25f;
                outB[idx] = (bR + bGrb + bGbr + bB)   * 0.25f;
            }
    });
}

// ---------------------------------------------------------------------------
// Statistics helpers for per-frame MAD sigma-clip
// ---------------------------------------------------------------------------

static double medianOf(vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double madSigmaOf(const vector<double> &v, double med)
{
    vector<double> dev(v.size());
    for (size_t i = 0; i < v.size(); ++i) dev[i] = std::abs(v[i] - med);
    return 1.4826 * medianOf(dev);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0]
             << " [-o output.fits] [-p output.png]"
                " [-m mean|median|sigmaclip|wstream] [-k kappa] [-j threads]"
                " [-w N] [-b native|opencv] frame1.fits ...\n";
        return 1;
    }

    string    outputPath = "stacked.fits";
    string    pngPath;
    StackMode mode     = StackMode::Mean;
    double    kappa    = 3.0;
    int       nThreads = (int)thread::hardware_concurrency();
    if (nThreads < 1) nThreads = 1;
    int       wUpN     = 5;   // warm-up frame count for wstream (-w N); 0 = disabled
#ifdef HAVE_OPENCV
    Backend   backend  = Backend::OpenCV;
#else
    Backend   backend  = Backend::Native;
#endif

    vector<string> inputs;

    for (int i = 1; i < argc; ++i)
    {
        string a = argv[i];
        if      (a == "-o" && i+1 < argc) outputPath = argv[++i];
        else if (a == "-p" && i+1 < argc) pngPath    = argv[++i];
        else if (a == "-k" && i+1 < argc)
        {
            try { kappa = stod(argv[++i]); }
            catch (...) { cerr << "Invalid kappa: " << argv[i] << "\n"; return 1; }
        }
        else if (a == "-j" && i+1 < argc)
        {
            try { nThreads = stoi(argv[++i]); }
            catch (...) { cerr << "Invalid thread count: " << argv[i] << "\n"; return 1; }
            if (nThreads <= 0) nThreads = (int)thread::hardware_concurrency();
            if (nThreads <  1) nThreads = 1;
        }
        else if (a == "-m" && i+1 < argc)
        {
            string m = argv[++i];
            if      (m == "median")    mode = StackMode::Median;
            else if (m == "sigmaclip") mode = StackMode::SigmaClip;
            else if (m == "wstream")   mode = StackMode::WelfordStream;
            else if (m != "mean") { cerr << "Unknown mode: " << m << "\n"; return 1; }
        }
        else if (a == "-w" && i+1 < argc)
        {
            try { wUpN = stoi(argv[++i]); }
            catch (...) { cerr << "Invalid warm-up count: " << argv[i] << "\n"; return 1; }
            if (wUpN < 0) wUpN = 0;
        }
        else if (a == "-b" && i+1 < argc)
        {
            string b = argv[++i];
            if      (b == "native") backend = Backend::Native;
            else if (b == "opencv")
            {
#ifdef HAVE_OPENCV
                backend = Backend::OpenCV;
#else
                cerr << "Warning: -b opencv ignored (not compiled with HAVE_OPENCV)\n";
#endif
            }
            else { cerr << "Unknown backend: " << b << "\n"; return 1; }
        }
        else inputs.push_back(a);
    }
    if (inputs.empty()) { cerr << "No input files.\n"; return 1; }
    std::sort(inputs.begin(), inputs.end());

    const string refPat = readBayerPat(inputs[0]);
    BayerLayout refLayout{};
    const bool bayer = !refPat.empty();
    if (bayer && !parseBayerLayout(refPat, refLayout))
    {
        cerr << "Unsupported Bayer pattern \"" << refPat << "\" in " << inputs[0]
             << " -- supported: RGGB GRBG GBRG BGGR.\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // Load reference; set DONUTS reference on green channel or full frame.
    // ------------------------------------------------------------------

    int bw = 0, bh = 0;
    vector<Pix> refRaw = loadFITS(inputs[0], bw, bh);
    if (refRaw.empty()) return 1;

    const char *modeNames[] = { "mean", "median", "sigmaclip", "wstream" };
    cout << "Reference: " << inputs[0] << "  (" << bw << "x" << bh << ")";
    if (bayer) cout << "  [" << refPat << " Bayer]";
    cout << "  mode=" << modeNames[(int)mode];
    if (mode == StackMode::SigmaClip || mode == StackMode::WelfordStream)
        cout << " kappa=" << kappa;
    if (mode == StackMode::WelfordStream && wUpN > 0)
        cout << " warmup=" << wUpN;
    {
        const char *be = (backend == Backend::OpenCV) ? "opencv" : "native";
        const char *db = (backend == Backend::OpenCV) ? "opencv" : "mhc";
        cout << "  warp=" << be << " debayer=" << db;
    }
    cout << "  threads=" << nThreads << "\n";

    Donuts::Guider guider;
    if (bayer)
    {
        int gw = 0, gh = 0;
        auto refGreen = extractGreen(refRaw, bw, bh, gw, gh, refLayout);
        auto refGreenD = toDouble(refGreen);
        guider.setReference(refGreenD.data(), gw, gh);
    }
    else { auto d = toDouble(refRaw); guider.setReference(d.data(), bw, bh); }

    // ------------------------------------------------------------------
    // Pass 1: measure transforms for all non-reference frames.
    // ------------------------------------------------------------------

    struct FrameInfo { string path; Donuts::Transform t; bool accepted=true; string reason; };
    vector<FrameInfo> frames;
    frames.reserve(inputs.size() - 1);

    for (size_t i = 1; i < inputs.size(); ++i)
    {
        FrameInfo fi;
        fi.path = inputs[i];
        int fw = 0, fh = 0;
        vector<Pix> raw = loadFITS(fi.path, fw, fh);
        if (raw.empty())          { fi.accepted=false; fi.reason="load failed";   frames.push_back(fi); continue; }
        if (fw != bw || fh != bh) { fi.accepted=false; fi.reason="size mismatch"; frames.push_back(fi); continue; }
        {
            string pat = readBayerPat(fi.path);
            if (pat != refPat)
            {
                fi.accepted = false;
                fi.reason   = "Bayer pattern mismatch (ref=" + (refPat.empty() ? "mono" : refPat)
                              + " frame=" + (pat.empty() ? "mono" : pat) + ")";
                frames.push_back(fi);
                continue;
            }
        }

        if (bayer)
        {
            int gw=0, gh=0;
            auto g  = extractGreen(raw, fw, fh, gw, gh, refLayout);
            auto gd = toDouble(g);
            fi.t    = guider.measure(gd.data(), gw, gh);
        }
        else { auto d = toDouble(raw); fi.t = guider.measure(d.data(), fw, fh); }

        if      (!fi.t.valid())
            { fi.accepted=false; fi.reason="SNR < 3"; }
        else if (std::abs(fi.t.dtheta) > MAX_ROTATION_RAD)
            { fi.accepted=false; fi.reason="rotation bound"; }
        else if (std::abs(fi.t.dx) > MAX_TRANSLATION_PX || std::abs(fi.t.dy) > MAX_TRANSLATION_PX)
            { fi.accepted=false; fi.reason="translation bound"; }

        frames.push_back(fi);
    }

    // Pass 1b: MAD sigma-clip on accepted transforms.
    {
        vector<double> dxV, dyV, dtV;
        for (const auto &fi : frames)
            if (fi.accepted) { dxV.push_back(fi.t.dx); dyV.push_back(fi.t.dy); dtV.push_back(fi.t.dtheta); }
        if (dxV.size() >= 3)
        {
            double mDx=medianOf(dxV), sDx=std::max(madSigmaOf(dxV,mDx), 0.5);
            double mDy=medianOf(dyV), sDy=std::max(madSigmaOf(dyV,mDy), 0.5);
            double mDt=medianOf(dtV), sDt=std::max(madSigmaOf(dtV,mDt), 0.5*M_PI/180.0);
            for (auto &fi : frames)
                if (fi.accepted &&
                    (std::abs(fi.t.dx     - mDx) > MAD_NSIGMA * sDx ||
                     std::abs(fi.t.dy     - mDy) > MAD_NSIGMA * sDy ||
                     std::abs(fi.t.dtheta - mDt) > MAD_NSIGMA * sDt))
                    { fi.accepted=false; fi.reason="sigma-clip"; }
        }
    }

    for (size_t i = 0; i < frames.size(); ++i)
    {
        const auto &fi = frames[i];
        cout << "  [" << (i+1) << "] " << fi.path
             << "  dx=" << fi.t.dx << "  dy=" << fi.t.dy
             << "  dtheta=" << fi.t.dtheta * 180.0 / M_PI << " deg"
             << "  snr=" << fi.t.snr;
        cout << (fi.accepted ? "  -> OK\n" : ("  -> SKIP (" + fi.reason + ")\n"));
    }

    // ------------------------------------------------------------------
    // Allocate accumulators.
    // ------------------------------------------------------------------

    int dw = bayer ? bw/2 : bw;
    int dh = bayer ? bh/2 : bh;
    size_t npix = (size_t)dw * dh;

    int nAccepted = 1;
    for (const auto &fi : frames) if (fi.accepted) ++nAccepted;

    vector<Pix> accumR(npix, 0.0f), accumG1(npix, 0.0f), accumG2(npix, 0.0f), accumB(npix, 0.0f);
    vector<int> coverage(npix, 0);

    // Frame-major stacks: stkFrames*[frameIndex][pixelIndex].
    // Each accepted frame appends one vector<Pix>(npix) -- one allocation per frame,
    // no upfront bulk allocation. NaN marks masked-out pixels (border regions).
    vector<vector<Pix>> stkFramesR, stkFramesG1, stkFramesG2, stkFramesB;
    if (mode == StackMode::Median || mode == StackMode::SigmaClip)
    {
        size_t mb = 4 * npix * (size_t)nAccepted * sizeof(Pix) / (1024*1024);
        if (mb > 4096)
            cerr << "Warning: median/sigmaclip estimated ~" << mb << " MB for "
                 << nAccepted << " frames; consider -m wstream.\n";
        else
            cout << "Pixel stacks: ~" << mb << " MB estimated ("
                 << nAccepted << " frames, allocated per-frame).\n";
        stkFramesR.reserve(nAccepted);
        stkFramesG1.reserve(nAccepted);
        stkFramesG2.reserve(nAccepted);
        stkFramesB.reserve(nAccepted);
    }

    // Welford streaming state: {mean, M2, count} per pixel -- O(pixels) memory
    // regardless of frame count. mean[] is the live stacked image at all times.
    vector<Pix>      wMeanR, wMeanG1, wMeanG2, wMeanB;
    vector<Pix>      wM2R,   wM2G1,   wM2G2,   wM2B;
    vector<uint32_t> wCount;
    if (mode == StackMode::WelfordStream)
    {
        wMeanR.assign(npix, 0.0f); wMeanG1.assign(npix, 0.0f); wMeanG2.assign(npix, 0.0f); wMeanB.assign(npix, 0.0f);
        wM2R.assign(npix, 0.0f);   wM2G1.assign(npix, 0.0f);   wM2G2.assign(npix, 0.0f);   wM2B.assign(npix, 0.0f);
        wCount.assign(npix, 0u);
        size_t mb = (8 * sizeof(Pix) + sizeof(uint32_t)) * npix / (1024*1024);
        cout << "Welford streaming: " << mb << " MB (constant, frame-count-independent).\n";
    }

    const Pix kNaN = std::numeric_limits<Pix>::quiet_NaN();

    // Warm-up buffer for wstream mode: first wUpN accepted frames go here
    // for clean sigmaclip seeding of the Welford baseline.
    vector<vector<Pix>> warmR, warmG1, warmG2, warmB;
    bool warmPhase = (mode == StackMode::WelfordStream && wUpN > 0);
    int  warmCount = 0;
    if (warmPhase)
    {
        warmR.reserve(wUpN);
        warmG1.reserve(wUpN);
        warmG2.reserve(wUpN);
        warmB.reserve(wUpN);
    }

    // Threaded accumulate: pixel ranges are disjoint across threads -- no locking.
    auto accumulateFrame = [&](const vector<Pix> &r, const vector<Pix> &g1,
                               const vector<Pix> &g2, const vector<Pix> &b,
                               const vector<uint8_t> *maskPtr)
    {
        if (mode == StackMode::Mean)
        {
            parallelFor((int)npix, nThreads, [&](int klo, int khi)
            {
                for (int k = klo; k < khi; ++k)
                {
                    if (maskPtr && !(*maskPtr)[k]) continue;
                    accumR[k] += r[k]; accumG1[k] += g1[k]; accumG2[k] += g2[k]; accumB[k] += b[k];
                    coverage[k]++;
                }
            });
        }
        else
        {
            // Push a NaN-filled frame layer, then fill valid pixels in parallel.
            // push_back is serial (must complete before parallelFor captures the ptrs).
            try
            {
                stkFramesR.push_back(vector<Pix>(npix, kNaN));
                stkFramesG1.push_back(vector<Pix>(npix, kNaN));
                stkFramesG2.push_back(vector<Pix>(npix, kNaN));
                stkFramesB.push_back(vector<Pix>(npix, kNaN));
            }
            catch (const std::bad_alloc &)
            {
                cerr << "Out of memory adding frame to pixel stack (frame "
                     << stkFramesR.size() << "). Use -m wstream.\n";
                return;
            }
            Pix *fr  = stkFramesR.back().data();
            Pix *fg1 = stkFramesG1.back().data();
            Pix *fg2 = stkFramesG2.back().data();
            Pix *fb  = stkFramesB.back().data();
            parallelFor((int)npix, nThreads, [&](int klo, int khi)
            {
                for (int k = klo; k < khi; ++k)
                {
                    if (maskPtr && !(*maskPtr)[k]) continue;
                    fr[k] = r[k]; fg1[k] = g1[k]; fg2[k] = g2[k]; fb[k] = b[k];
                }
            });
        }
    };

    // Welford online update for streaming sigma-clip.
    // bootstrap=true for the reference frame: always accept, no gate.
    // bootstrap=false for subsequent frames: gate with kappa*sigma before update.
    // Gate uses (G1+G2)*0.5: satellites are broadband and both green channels
    // measure the same signal; averaging gives a cleaner gate estimate.
    // Requires n >= 2 to have a meaningful variance estimate; earlier samples
    // are always accepted to seed the statistics.
    auto welfordFrame = [&](const vector<Pix> &r, const vector<Pix> &g1,
                            const vector<Pix> &g2, const vector<Pix> &b,
                            const vector<uint8_t> *maskPtr,
                            bool bootstrap)
    {
        const float kf = (float)kappa;
        parallelFor((int)npix, nThreads, [&](int klo, int khi)
        {
            for (int k = klo; k < khi; ++k)
            {
                if (maskPtr && !(*maskPtr)[k]) continue;
                uint32_t n = wCount[k];

                // Rejection gate: use (G1+G2)*0.5 -- exact measured values,
                // better gate estimate than G1 alone.
                if (!bootstrap && n >= 2 && wM2G1[k] > 0.0f)
                {
                    float gAvg = (g1[k] + g2[k]) * 0.5f;
                    float sig = std::sqrt(wM2G1[k] / (float)(n - 1));
                    if (std::abs(gAvg - wMeanG1[k]) > kf * sig)
                        continue;
                }

                // Welford recurrence (numerically stable online mean + M2).
                ++n;
                const float nf = (float)n;
                float d;
                d = r [k] - wMeanR [k]; wMeanR [k] += d / nf; wM2R [k] += d * (r [k] - wMeanR [k]);
                d = g1[k] - wMeanG1[k]; wMeanG1[k] += d / nf; wM2G1[k] += d * (g1[k] - wMeanG1[k]);
                d = g2[k] - wMeanG2[k]; wMeanG2[k] += d / nf; wM2G2[k] += d * (g2[k] - wMeanG2[k]);
                d = b [k] - wMeanB [k]; wMeanB [k] += d / nf; wM2B [k] += d * (b [k] - wMeanB [k]);
                wCount[k] = n;
            }
        });
    };

    // Append one frame to the warm-up buffer (NaN-sentinel frame-major layout).
    auto addToWarmBuffer = [&](const vector<Pix> &r, const vector<Pix> &g1,
                               const vector<Pix> &g2, const vector<Pix> &b,
                               const vector<uint8_t> *maskPtr)
    {
        try
        {
            warmR.push_back(vector<Pix>(npix, kNaN));
            warmG1.push_back(vector<Pix>(npix, kNaN));
            warmG2.push_back(vector<Pix>(npix, kNaN));
            warmB.push_back(vector<Pix>(npix, kNaN));
        }
        catch (const std::bad_alloc &)
        {
            cerr << "Out of memory building warm-up buffer; disabling warm-up.\n";
            warmPhase = false;
            return;
        }
        Pix *fr  = warmR .back().data();
        Pix *fg1 = warmG1.back().data();
        Pix *fg2 = warmG2.back().data();
        Pix *fb  = warmB .back().data();
        parallelFor((int)npix, nThreads, [&](int klo, int khi)
        {
            for (int k = klo; k < khi; ++k)
            {
                if (maskPtr && !(*maskPtr)[k]) continue;
                fr[k] = r[k]; fg1[k] = g1[k]; fg2[k] = g2[k]; fb[k] = b[k];
            }
        });
        ++warmCount;
    };

    // Seed Welford state from warm buffer, then flush the buffer.
    auto flushWarmBuffer = [&]()
    {
        cout << "Seeding Welford from " << warmCount << " warm-up frame"
             << (warmCount == 1 ? "" : "s") << " (G1-channel MAD sigmaclip seed)...\n";
        seedWelford4(warmR, warmG1, warmG2, warmB, npix, kappa, nThreads,
                     wMeanR, wMeanG1, wMeanG2, wMeanB, wM2R, wM2G1, wM2G2, wM2B, wCount);
        warmR .clear(); warmR .shrink_to_fit();
        warmG1.clear(); warmG1.shrink_to_fit();
        warmG2.clear(); warmG2.shrink_to_fit();
        warmB .clear(); warmB .shrink_to_fit();
        warmPhase = false;
    };

    // ------------------------------------------------------------------
    // Timing accumulators.
    // ------------------------------------------------------------------

    using Clock = std::chrono::steady_clock;
    using Sec   = std::chrono::duration<double>;
    auto since  = [](Clock::time_point t0) {
        return std::chrono::duration_cast<Sec>(Clock::now() - t0).count();
    };

    double t_io = 0, t_subplane = 0, t_warp = 0, t_stack = 0, t_debayer = 0;
    Clock::time_point t0;

    // ------------------------------------------------------------------
    // Add reference frame (no transform; all pixels valid).
    // ------------------------------------------------------------------

    {
        vector<Pix> r, g1, g2, b;
        if (bayer)
        {
            t0 = Clock::now();
            int tw=0,th=0;
            extractSubplanes(refRaw, bw, bh, r, g1, g2, b, tw, th, refLayout);
            t_subplane += since(t0);
        }
        else { r = refRaw; g1 = refRaw; g2 = refRaw; b = refRaw; }

        t0 = Clock::now();
        if      (warmPhase)                        addToWarmBuffer(r, g1, g2, b, nullptr);
        else if (mode == StackMode::WelfordStream) welfordFrame(r, g1, g2, b, nullptr, true);
        else                                       accumulateFrame(r, g1, g2, b, nullptr);
        t_stack += since(t0);
    }

    // ------------------------------------------------------------------
    // Pass 2: load, align, accumulate each accepted frame.
    // ------------------------------------------------------------------

    int stackCount = 1;
    for (const auto &fi : frames)
    {
        if (!fi.accepted) continue;

        t0 = Clock::now();
        int fw=0, fh=0;
        vector<Pix> raw = loadFITS(fi.path, fw, fh);
        t_io += since(t0);
        if (raw.empty()) continue;

        vector<Pix> r, g1, g2, b;
        if (bayer)
        {
            t0 = Clock::now();
            int tw=0,th=0;
            extractSubplanes(raw, fw, fh, r, g1, g2, b, tw, th, refLayout);
            t_subplane += since(t0);
        }
        else { r = raw; g1 = raw; g2 = raw; b = raw; }

        vector<Pix>     dstR(npix), dstG1(npix), dstG2(npix), dstB(npix);
        vector<uint8_t> mask(npix, 0);

        t0 = Clock::now();
        const auto M = fi.t.alignmentMatrix(dw, dh);
        alignFrame4(r, g1, g2, b, dw, dh, M, dstR, dstG1, dstG2, dstB, mask, nThreads, backend);
        t_warp += since(t0);

        t0 = Clock::now();
        if (warmPhase)
        {
            addToWarmBuffer(dstR, dstG1, dstG2, dstB, &mask);
            if (warmCount >= wUpN) flushWarmBuffer();
        }
        else if (mode == StackMode::WelfordStream) welfordFrame(dstR, dstG1, dstG2, dstB, &mask, false);
        else                                       accumulateFrame(dstR, dstG1, dstG2, dstB, &mask);
        t_stack += since(t0);
        ++stackCount;
    }

    // Flush whatever warm frames accumulated when total accepted < wUpN.
    if (warmPhase && warmCount > 0) flushWarmBuffer();

    cout << "\nStacked " << stackCount << "/" << (int)inputs.size()
         << " frames -> " << outputPath << "\n";

    // ------------------------------------------------------------------
    // Reduce to final image.
    // ------------------------------------------------------------------

    vector<Pix> finalR(npix), finalG1(npix), finalG2(npix), finalB(npix);

    if (mode == StackMode::WelfordStream)
    {
        // mean[] is already the final stacked image -- no separate reduce pass needed.
        finalR = wMeanR; finalG1 = wMeanG1; finalG2 = wMeanG2; finalB = wMeanB;
    }
    else if (mode == StackMode::Mean)
    {
        parallelFor((int)npix, nThreads, [&](int klo, int khi)
        {
            for (int k = klo; k < khi; ++k)
            {
                float inv = coverage[k] > 0 ? 1.0f / coverage[k] : 0.0f;
                finalR [k] = accumR [k] * inv;
                finalG1[k] = accumG1[k] * inv;
                finalG2[k] = accumG2[k] * inv;
                finalB [k] = accumB [k] * inv;
            }
        });
    }
    else
    {
        reduceStack4(stkFramesR, stkFramesG1, stkFramesG2, stkFramesB, npix,
                     mode, kappa, nThreads, finalR, finalG1, finalG2, finalB);
    }

    // Post-stack debayer: opencv bilinear for opencv backend, MHC for native.
    vector<Pix> finalRout(npix), finalG(npix), finalBout(npix);
    t0 = Clock::now();
#ifdef HAVE_OPENCV
    if (backend == Backend::OpenCV)
        debayerOpenCVPostStack(finalR, finalG1, finalG2, finalB, dw, dh,
                               refLayout, refPat, finalRout, finalG, finalBout);
    else
#endif
        debayerMHC_halfres(finalR, finalG1, finalG2, finalB, dw, dh,
                           finalRout, finalG, finalBout, nThreads);
    t_debayer = since(t0);

    // ------------------------------------------------------------------
    // Print timing breakdown.
    // ------------------------------------------------------------------

    {
        double total = t_io + t_subplane + t_warp + t_stack + t_debayer;
        auto pct = [&](double t) { return total > 0 ? (int)(100.0 * t / total + 0.5) : 0; };
        cout << "Timings (s):  io="       << t_io
             << "  subplane=" << t_subplane << " (" << pct(t_subplane) << "%)"
             << "  warp="     << t_warp     << " (" << pct(t_warp)     << "%)"
             << "  stack="    << t_stack    << " (" << pct(t_stack)    << "%)"
             << "  debayer="   << t_debayer      << " (" << pct(t_debayer)      << "%)"
             << "  total="    << total      << "\n";
    }

    // ------------------------------------------------------------------
    // Write outputs.
    // ------------------------------------------------------------------

    if (bayer)
    {
        if (!writeFITS3(outputPath, finalRout, finalG, finalBout, dw, dh)) return 1;
        cout << "Output: " << dw << "x" << dh << " 3-plane FITS (R/G/B)\n";
        if (!pngPath.empty())
        {
            if (!writePNG(pngPath, finalRout, finalG, finalBout, dw, dh)) return 1;
            cout << "PNG:    " << pngPath << "\n";
        }
    }
    else
    {
        if (!writeFITS(outputPath, finalRout, dw, dh)) return 1;
        if (!pngPath.empty())
        {
            if (!writePNG(pngPath, finalRout, finalRout, finalRout, dw, dh)) return 1;
            cout << "PNG:    " << pngPath << "\n";
        }
    }

    return 0;
}
