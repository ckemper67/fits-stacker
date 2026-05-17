/*
 * SPDX-FileCopyrightText: 2026 Christian Kemper <ckemper@gmail.com>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Standalone FITS stacker using the DONUTS rotation/translation solver.
 *
 * Monochrome: aligns and stacks directly.
 * Bayer (BAYERPAT=RGGB): extracts green at half resolution for DONUTS
 * registration, then 2x2 debayers and applies the rigid-body transform to
 * each colour channel before stacking. Output is a 3-plane FITS (R/G/B) at
 * half resolution.
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

// pat is the BAYERPAT string (e.g. "RGGB"); used only by the OpenCV path.
static void debayer(const vector<Pix> &bayer, int bw, int bh,
                    vector<Pix> &r, vector<Pix> &g, vector<Pix> &b,
                    int &w, int &h, const BayerLayout &bl, const string &pat = "")
{
    w = bw / 2;
    h = bh / 2;
    size_t npix = (size_t)w * h;
    r.resize(npix); g.resize(npix); b.resize(npix);

#ifdef HAVE_OPENCV
    // OpenCV Bayer->BGR bilinear demosaic at full resolution, then INTER_AREA
    // downsample to half resolution. Gives better edge quality than 2x2 average.
    // Requires data in [0, 65535]; scales from actual min/max and back.
    // Pattern mapping: XY in COLOR_BayerXY2BGR means top-left=X, top-right=Y.
    int cvCode = -1;
    if      (pat == "RGGB") cvCode = cv::COLOR_BayerRG2BGR;
    else if (pat == "GRBG") cvCode = cv::COLOR_BayerGR2BGR;
    else if (pat == "GBRG") cvCode = cv::COLOR_BayerGB2BGR;
    else if (pat == "BGGR") cvCode = cv::COLOR_BayerBG2BGR;

    if (cvCode >= 0)
    {
        cv::Mat src(bh, bw, CV_32F, const_cast<Pix *>(bayer.data()));
        double minVal, maxVal;
        cv::minMaxIdx(src, &minVal, &maxVal);
        if (maxVal <= minVal) maxVal = minVal + 1.0;

        // Scale to uint16 for cvtColor, which doesn't support float Bayer.
        cv::Mat src16;
        src.convertTo(src16, CV_16U, 65535.0 / (maxVal - minVal),
                      -minVal * 65535.0 / (maxVal - minVal));

        cv::Mat bgr16;
        cv::cvtColor(src16, bgr16, cvCode);

        // Downsample to half resolution with INTER_AREA (correct averaging).
        cv::Mat bgr16half;
        cv::resize(bgr16, bgr16half, cv::Size(w, h), 0, 0, cv::INTER_AREA);

        // Split BGR channels and scale back to original float range.
        vector<cv::Mat> chans(3);
        cv::split(bgr16half, chans);  // chans[0]=B, [1]=G, [2]=R
        double invScale = (maxVal - minVal) / 65535.0;
        for (int chan = 0; chan < 3; ++chan)
            chans[chan].convertTo(chans[chan], CV_32F, invScale, minVal);

        // Copy from cv::Mat to our float vectors.
        std::copy((Pix*)chans[2].datastart, (Pix*)chans[2].dataend, r.begin());
        std::copy((Pix*)chans[1].datastart, (Pix*)chans[1].dataend, g.begin());
        std::copy((Pix*)chans[0].datastart, (Pix*)chans[0].dataend, b.begin());
        return;
    }
#else
    (void)pat;
#endif

    // Fallback: 2x2 cell average (fast, half-resolution).
    const int rr = bl.r  >> 1, rc  = bl.r  & 1;
    const int g1r= bl.g1 >> 1, g1c = bl.g1 & 1;
    const int g2r= bl.g2 >> 1, g2c = bl.g2 & 1;
    const int br = bl.b  >> 1, bc  = bl.b  & 1;
    for (int row = 0; row < h; ++row)
        for (int col = 0; col < w; ++col)
        {
            size_t i = (size_t)row * w + col;
            r[i] = bayer[(size_t)(2*row + rr)  * bw + (2*col + rc)];
            g[i] = (bayer[(size_t)(2*row + g1r) * bw + (2*col + g1c)] +
                    bayer[(size_t)(2*row + g2r)  * bw + (2*col + g2c)]) * 0.5f;
            b[i] = bayer[(size_t)(2*row + br)   * bw + (2*col + bc)];
        }
}

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
    int nThreads)
{
#ifdef HAVE_OPENCV
    // m maps destination pixel -> source pixel (inverse warp convention).
    // cv::warpAffine with WARP_INVERSE_MAP interprets M the same way.
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

    // Build mask: warp a ones image with nearest-neighbour to get a clean binary boundary.
    cv::Mat ones = cv::Mat::ones(h, w, CV_32F);
    cv::Mat warpedOnes;
    cv::warpAffine(ones, warpedOnes, M_cv, sz,
                   cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 0);
    const float *mp = warpedOnes.ptr<float>();
    for (int k = 0; k < w * h; ++k) mask[k] = mp[k] > 0.5f ? 1 : 0;
#else
    (void)nThreads;  // used by parallelFor inside
    parallelFor(h, nThreads, [&](int ylo, int yhi)
    {
        alignFrame3rows(srcR, srcG, srcB, w, h, m, dstR, dstG, dstB, mask, ylo, yhi);
    });
#endif
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
static void reduceStack3(
    const vector<vector<Pix>> &stkR,
    const vector<vector<Pix>> &stkG,
    const vector<vector<Pix>> &stkB,
    size_t npix,
    StackMode mode, double kappa, int nThreads,
    vector<Pix> &outR, vector<Pix> &outG, vector<Pix> &outB)
{
    const int nFrames = (int)stkR.size();
    outR.resize(npix); outG.resize(npix); outB.resize(npix);

    parallelFor((int)npix, nThreads, [&](int klo, int khi)
    {
        vector<Pix> gR(nFrames), gG(nFrames), gB(nFrames);
        vector<Pix> scratch(nFrames), devbuf(nFrames);
        for (int k = klo; k < khi; ++k)
        {
            // Gather valid (non-NaN) values for this pixel across all frames.
            int n = 0;
            for (int f = 0; f < nFrames; ++f)
            {
                Pix v = stkR[f][k];
                if (std::isnan(v)) continue;
                gR[n] = v;
                gG[n] = stkG[f][k];
                gB[n] = stkB[f][k];
                ++n;
            }
            if (n == 0) { outR[k] = outG[k] = outB[k] = 0.0f; continue; }
            if (mode == StackMode::Median)
            {
                outR[k] = pixelMedian   (gR.data(), scratch.data(), n);
                outG[k] = pixelMedian   (gG.data(), scratch.data(), n);
                outB[k] = pixelMedian   (gB.data(), scratch.data(), n);
            }
            else
            {
                outR[k] = pixelSigmaClip(gR.data(), scratch.data(), devbuf.data(), n, kappa);
                outG[k] = pixelSigmaClip(gG.data(), scratch.data(), devbuf.data(), n, kappa);
                outB[k] = pixelSigmaClip(gB.data(), scratch.data(), devbuf.data(), n, kappa);
            }
        }
    });
}

// Seed Welford streaming state from a warm-up buffer of N frames.
// Eliminates the first-frame vulnerability: if a satellite trail hits frame 1
// or 2, Welford has no prior variance to reject it. Seeding runs G-channel
// MAD sigma-clip per pixel across the N warm-up frames, then runs the Welford
// recurrence on the survivors to produce a pristine {mean, M2} baseline.
// After this call, wCount[k] >= 1 for all pixels with at least one valid sample;
// the online streaming gate fires immediately on frame N+1.
static void seedWelford3(
    const vector<vector<Pix>> &warmR,
    const vector<vector<Pix>> &warmG,
    const vector<vector<Pix>> &warmB,
    size_t npix, double kappa, int nThreads,
    vector<Pix>      &wMeanR, vector<Pix>      &wMeanG, vector<Pix>      &wMeanB,
    vector<Pix>      &wM2R,   vector<Pix>      &wM2G,   vector<Pix>      &wM2B,
    vector<uint32_t> &wCount)
{
    const int nFrames = (int)warmR.size();
    parallelFor((int)npix, nThreads, [&](int klo, int khi)
    {
        // Per-thread scratch: nFrames entries each -- no per-pixel malloc.
        vector<Pix> vG(nFrames), vR(nFrames), vB(nFrames);
        vector<Pix> scratch(nFrames), devbuf(nFrames);
        vector<int> survivorIdx(nFrames);
        for (int k = klo; k < khi; ++k)
        {
            // Gather non-NaN samples for this pixel.
            int n = 0;
            for (int f = 0; f < nFrames; ++f)
            {
                Pix gv = warmG[f][k];
                if (std::isnan(gv)) continue;
                vG[n] = gv;
                vR[n] = warmR[f][k];
                vB[n] = warmB[f][k];
                survivorIdx[n] = n;
                ++n;
            }
            if (n == 0) { wCount[k] = 0; continue; }
            if (n == 1)
            {
                wMeanR[k] = vR[0]; wMeanG[k] = vG[0]; wMeanB[k] = vB[0];
                wM2R[k] = 0.0f; wM2G[k] = 0.0f; wM2B[k] = 0.0f;
                wCount[k] = 1;
                continue;
            }

            // G-channel MAD sigma-clip: compact scratch[] and survivorIdx[] in parallel.
            std::copy(vG.begin(), vG.begin() + n, scratch.begin());
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

            // Welford recurrence over surviving {R, G, B} triplets.
            float mR=0.0f, mG=0.0f, mB=0.0f, m2R=0.0f, m2G=0.0f, m2B=0.0f;
            for (int i = 0; i < cnt; ++i)
            {
                const int j = survivorIdx[i];
                const float nf = (float)(i + 1);
                float d;
                d = vR[j] - mR; mR += d / nf; m2R += d * (vR[j] - mR);
                d = vG[j] - mG; mG += d / nf; m2G += d * (vG[j] - mG);
                d = vB[j] - mB; mB += d / nf; m2B += d * (vB[j] - mB);
            }
            wMeanR[k] = mR; wMeanG[k] = mG; wMeanB[k] = mB;
            wM2R[k]   = m2R; wM2G[k]  = m2G; wM2B[k]  = m2B;
            wCount[k] = (uint32_t)cnt;
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
                " [-w N] frame1.fits ...\n";
        return 1;
    }

    string    outputPath = "stacked.fits";
    string    pngPath;
    StackMode mode     = StackMode::Mean;
    double    kappa    = 3.0;
    int       nThreads = (int)thread::hardware_concurrency();
    if (nThreads < 1) nThreads = 1;
    int       wUpN     = 5;   // warm-up frame count for wstream (-w N); 0 = disabled

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
#ifdef HAVE_OPENCV
    cout << "  warp=opencv debayer=opencv";
#else
    cout << "  warp=native debayer=native";
#endif
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

    vector<Pix> accumR(npix, 0.0f), accumG(npix, 0.0f), accumB(npix, 0.0f);
    vector<int> coverage(npix, 0);

    // Frame-major stacks: stkFrames*[frameIndex][pixelIndex].
    // Each accepted frame appends one vector<Pix>(npix) -- one allocation per frame,
    // no upfront bulk allocation. NaN marks masked-out pixels (border regions).
    vector<vector<Pix>> stkFramesR, stkFramesG, stkFramesB;
    if (mode == StackMode::Median || mode == StackMode::SigmaClip)
    {
        size_t mb = 3 * npix * (size_t)nAccepted * sizeof(Pix) / (1024*1024);
        if (mb > 4096)
            cerr << "Warning: median/sigmaclip estimated ~" << mb << " MB for "
                 << nAccepted << " frames; consider -m wstream.\n";
        else
            cout << "Pixel stacks: ~" << mb << " MB estimated ("
                 << nAccepted << " frames, allocated per-frame).\n";
        stkFramesR.reserve(nAccepted);
        stkFramesG.reserve(nAccepted);
        stkFramesB.reserve(nAccepted);
    }

    // Welford streaming state: {mean, M2, count} per pixel -- O(pixels) memory
    // regardless of frame count. mean[] is the live stacked image at all times.
    vector<Pix>      wMeanR, wMeanG, wMeanB;
    vector<Pix>      wM2R,   wM2G,   wM2B;
    vector<uint32_t> wCount;
    if (mode == StackMode::WelfordStream)
    {
        wMeanR.assign(npix, 0.0f); wMeanG.assign(npix, 0.0f); wMeanB.assign(npix, 0.0f);
        wM2R.assign(npix, 0.0f);   wM2G.assign(npix, 0.0f);   wM2B.assign(npix, 0.0f);
        wCount.assign(npix, 0u);
        size_t mb = (6 * sizeof(Pix) + sizeof(uint32_t)) * npix / (1024*1024);
        cout << "Welford streaming: " << mb << " MB (constant, frame-count-independent).\n";
    }

    const Pix kNaN = std::numeric_limits<Pix>::quiet_NaN();

    // Warm-up buffer for wstream mode: first wUpN accepted frames go here
    // for clean sigmaclip seeding of the Welford baseline.
    vector<vector<Pix>> warmR, warmG, warmB;
    bool warmPhase = (mode == StackMode::WelfordStream && wUpN > 0);
    int  warmCount = 0;
    if (warmPhase)
    {
        warmR.reserve(wUpN);
        warmG.reserve(wUpN);
        warmB.reserve(wUpN);
    }

    // Threaded accumulate: pixel ranges are disjoint across threads -- no locking.
    auto accumulateFrame = [&](const vector<Pix> &r, const vector<Pix> &g,
                               const vector<Pix> &b, const vector<uint8_t> *maskPtr)
    {
        if (mode == StackMode::Mean)
        {
            parallelFor((int)npix, nThreads, [&](int klo, int khi)
            {
                for (int k = klo; k < khi; ++k)
                {
                    if (maskPtr && !(*maskPtr)[k]) continue;
                    accumR[k] += r[k]; accumG[k] += g[k]; accumB[k] += b[k];
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
                stkFramesG.push_back(vector<Pix>(npix, kNaN));
                stkFramesB.push_back(vector<Pix>(npix, kNaN));
            }
            catch (const std::bad_alloc &)
            {
                cerr << "Out of memory adding frame to pixel stack (frame "
                     << stkFramesR.size() << "). Use -m wstream.\n";
                return;
            }
            Pix *fr = stkFramesR.back().data();
            Pix *fg = stkFramesG.back().data();
            Pix *fb = stkFramesB.back().data();
            parallelFor((int)npix, nThreads, [&](int klo, int khi)
            {
                for (int k = klo; k < khi; ++k)
                {
                    if (maskPtr && !(*maskPtr)[k]) continue;
                    fr[k] = r[k]; fg[k] = g[k]; fb[k] = b[k];
                }
            });
        }
    };

    // Welford online update for streaming sigma-clip.
    // bootstrap=true for the reference frame: always accept, no gate.
    // bootstrap=false for subsequent frames: gate with kappa*sigma before update.
    // Gate uses the G channel only: satellites are broadband so G is sufficient,
    // and single-channel gating avoids false rejection of chromatic targets.
    // Requires n >= 2 to have a meaningful variance estimate; earlier samples
    // are always accepted to seed the statistics.
    auto welfordFrame = [&](const vector<Pix> &r, const vector<Pix> &g,
                            const vector<Pix> &b, const vector<uint8_t> *maskPtr,
                            bool bootstrap)
    {
        const float kf = (float)kappa;
        parallelFor((int)npix, nThreads, [&](int klo, int khi)
        {
            for (int k = klo; k < khi; ++k)
            {
                if (maskPtr && !(*maskPtr)[k]) continue;
                uint32_t n = wCount[k];

                // Rejection gate: use G channel only.
                // Satellite/airplane trails are broadband -- G catches them just as
                // well as per-channel. Single-channel gating avoids false rejection
                // of chromatic targets (e.g. red stars with legitimately high R variance).
                if (!bootstrap && n >= 2 && wM2G[k] > 0.0f)
                {
                    float sig = std::sqrt(wM2G[k] / (float)(n - 1));
                    if (std::abs(g[k] - wMeanG[k]) > kf * sig)
                        continue;
                }

                // Welford recurrence (numerically stable online mean + M2).
                ++n;
                const float nf = (float)n;
                float d;
                d = r[k] - wMeanR[k]; wMeanR[k] += d / nf; wM2R[k] += d * (r[k] - wMeanR[k]);
                d = g[k] - wMeanG[k]; wMeanG[k] += d / nf; wM2G[k] += d * (g[k] - wMeanG[k]);
                d = b[k] - wMeanB[k]; wMeanB[k] += d / nf; wM2B[k] += d * (b[k] - wMeanB[k]);
                wCount[k] = n;
            }
        });
    };

    // Append one frame to the warm-up buffer (NaN-sentinel frame-major layout).
    auto addToWarmBuffer = [&](const vector<Pix> &r, const vector<Pix> &g,
                               const vector<Pix> &b, const vector<uint8_t> *maskPtr)
    {
        try
        {
            warmR.push_back(vector<Pix>(npix, kNaN));
            warmG.push_back(vector<Pix>(npix, kNaN));
            warmB.push_back(vector<Pix>(npix, kNaN));
        }
        catch (const std::bad_alloc &)
        {
            cerr << "Out of memory building warm-up buffer; disabling warm-up.\n";
            warmPhase = false;
            return;
        }
        Pix *fr = warmR.back().data();
        Pix *fg = warmG.back().data();
        Pix *fb = warmB.back().data();
        parallelFor((int)npix, nThreads, [&](int klo, int khi)
        {
            for (int k = klo; k < khi; ++k)
            {
                if (maskPtr && !(*maskPtr)[k]) continue;
                fr[k] = r[k]; fg[k] = g[k]; fb[k] = b[k];
            }
        });
        ++warmCount;
    };

    // Seed Welford state from warm buffer, then flush the buffer.
    auto flushWarmBuffer = [&]()
    {
        cout << "Seeding Welford from " << warmCount << " warm-up frame"
             << (warmCount == 1 ? "" : "s") << " (G-channel MAD sigmaclip seed)...\n";
        seedWelford3(warmR, warmG, warmB, npix, kappa, nThreads,
                     wMeanR, wMeanG, wMeanB, wM2R, wM2G, wM2B, wCount);
        warmR.clear(); warmR.shrink_to_fit();
        warmG.clear(); warmG.shrink_to_fit();
        warmB.clear(); warmB.shrink_to_fit();
        warmPhase = false;
    };

    // ------------------------------------------------------------------
    // Add reference frame (no transform; all pixels valid).
    // ------------------------------------------------------------------

    {
        vector<Pix> r, g, b;
        if (bayer) { int tw=0,th=0; debayer(refRaw, bw, bh, r, g, b, tw, th, refLayout, refPat); }
        else       { r = refRaw; g = refRaw; b = refRaw; }
        if      (warmPhase)                        addToWarmBuffer(r, g, b, nullptr);
        else if (mode == StackMode::WelfordStream) welfordFrame(r, g, b, nullptr, true);
        else                                       accumulateFrame(r, g, b, nullptr);
    }

    // ------------------------------------------------------------------
    // Pass 2: load, align, accumulate each accepted frame.
    // ------------------------------------------------------------------

    int stackCount = 1;
    for (const auto &fi : frames)
    {
        if (!fi.accepted) continue;
        int fw=0, fh=0;
        vector<Pix> raw = loadFITS(fi.path, fw, fh);
        if (raw.empty()) continue;

        vector<Pix> r, g, b;
        if (bayer) { int tw=0,th=0; debayer(raw, fw, fh, r, g, b, tw, th, refLayout, refPat); }
        else       { r = raw; g = raw; b = raw; }

        vector<Pix>     dstR(npix), dstG(npix), dstB(npix);
        vector<uint8_t> mask(npix, 0);

        const auto M = fi.t.alignmentMatrix(dw, dh);
        alignFrame3(r, g, b, dw, dh, M, dstR, dstG, dstB, mask, nThreads);

        if (warmPhase)
        {
            addToWarmBuffer(dstR, dstG, dstB, &mask);
            if (warmCount >= wUpN) flushWarmBuffer();
        }
        else if (mode == StackMode::WelfordStream) welfordFrame(dstR, dstG, dstB, &mask, false);
        else                                       accumulateFrame(dstR, dstG, dstB, &mask);
        ++stackCount;
    }

    // Flush whatever warm frames accumulated when total accepted < wUpN.
    if (warmPhase && warmCount > 0) flushWarmBuffer();

    cout << "\nStacked " << stackCount << "/" << (int)inputs.size()
         << " frames -> " << outputPath << "\n";

    // ------------------------------------------------------------------
    // Reduce to final image.
    // ------------------------------------------------------------------

    vector<Pix> finalR(npix), finalG(npix), finalB(npix);

    if (mode == StackMode::WelfordStream)
    {
        // mean[] is already the final stacked image -- no separate reduce pass needed.
        finalR = wMeanR; finalG = wMeanG; finalB = wMeanB;
    }
    else if (mode == StackMode::Mean)
    {
        parallelFor((int)npix, nThreads, [&](int klo, int khi)
        {
            for (int k = klo; k < khi; ++k)
            {
                float inv = coverage[k] > 0 ? 1.0f / coverage[k] : 0.0f;
                finalR[k] = accumR[k] * inv;
                finalG[k] = accumG[k] * inv;
                finalB[k] = accumB[k] * inv;
            }
        });
    }
    else
    {
        reduceStack3(stkFramesR, stkFramesG, stkFramesB, npix,
                     mode, kappa, nThreads, finalR, finalG, finalB);
    }

    // ------------------------------------------------------------------
    // Write outputs.
    // ------------------------------------------------------------------

    if (bayer)
    {
        if (!writeFITS3(outputPath, finalR, finalG, finalB, dw, dh)) return 1;
        cout << "Output: " << dw << "x" << dh << " 3-plane FITS (R/G/B)\n";
        if (!pngPath.empty())
        {
            if (!writePNG(pngPath, finalR, finalG, finalB, dw, dh)) return 1;
            cout << "PNG:    " << pngPath << "\n";
        }
    }
    else
    {
        if (!writeFITS(outputPath, finalR, dw, dh)) return 1;
        if (!pngPath.empty())
        {
            if (!writePNG(pngPath, finalR, finalR, finalR, dw, dh)) return 1;
            cout << "PNG:    " << pngPath << "\n";
        }
    }

    return 0;
}
