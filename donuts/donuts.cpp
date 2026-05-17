/*
    SPDX-FileCopyrightText: 2026 Christian Kemper <ckemper@gmail.com>

    SPDX-License-Identifier: BSD-3-Clause
*/

#include "donuts.h"

// pocketfft is a private implementation detail; it never appears in donuts.h.
#include "pocketfft_hdronly.h"

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>
#include <cstddef>

namespace Donuts
{

// ---------------------------------------------------------------------------
// Transform::alignmentMatrix
// ---------------------------------------------------------------------------
//
// DONUTS convention: the measured (dx, dy, dtheta, scale) describe the
// transform from reference to current frame as:
//   "translate by (dx, dy), then rotate by dtheta about the image centre,
//    then scale isotropically about the centre."
//
// The alignment matrix inverts this: for each output (reference-grid) pixel
// at (xd, yd) it computes the corresponding source pixel in the current frame.
// Expanding with cx=w/2, cy=h/2, co=cos(dtheta), si=sin(dtheta), s=scale:
//
//   xs = s*co*(xd + dx - cx) - s*si*(yd + dy - cy) + cx
//   ys = s*si*(xd + dx - cx) + s*co*(yd + dy - cy) + cy
//
// which gives the 2x3 matrix below.

AffineMatrix Transform::alignmentMatrix(int w, int h) const
{
    const double cx = w * 0.5, cy = h * 0.5;
    const double co = std::cos(dtheta), si = std::sin(dtheta);
    AffineMatrix m;
    m.a  =  scale * co;
    m.b  = -scale * si;
    m.c  =  scale * si;
    m.d  =  scale * co;
    m.tx = m.a * (dx - cx) + m.b * (dy - cy) + cx;
    m.ty = m.c * (dx - cx) + m.d * (dy - cy) + cy;
    return m;
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct Quadrant
{
    std::vector<double> xProf, yProf;
    double xc { 0 };  // flux-weighted x-centroid relative to image centre
    double yc { 0 };  // flux-weighted y-centroid relative to image centre
};

struct ProfileSet
{
    Quadrant q[4];
    int width  { 0 };
    int height { 0 };
};

struct FrameStats
{
    double median { 0 };
    double stddev { 0 };
    double clip   { 0 };
};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

static FrameStats computeStats(const double *buf, int n)
{
    if (n <= 0) return {};

    // Welford one-pass mean + variance for stddev; track max for clip.
    double mean = 0, m2 = 0, maxVal = 0;
    for (int i = 0; i < n; ++i)
    {
        double delta = buf[i] - mean;
        mean += delta / (i + 1);
        m2   += delta * (buf[i] - mean);
        if (buf[i] > maxVal) maxVal = buf[i];
    }
    double stddev = (n > 1) ? std::sqrt(m2 / (n - 1)) : 0.0;

    // 1st-percentile sky background via nth_element on a subsample (capped at 64 K).
    // Using the dimmest 1% rather than the median matches McCormac 2013 and avoids
    // overestimating background in dense star fields where the median is pulled up
    // by stellar flux.
    const int SAMPLE = std::min(n, 65536);
    std::vector<double> sample(SAMPLE);
    double step = static_cast<double>(n) / SAMPLE;
    for (int i = 0; i < SAMPLE; ++i)
        sample[i] = buf[static_cast<int>(i * step)];
    auto p1 = sample.begin() + SAMPLE / 100;
    std::nth_element(sample.begin(), p1, sample.end());
    double median = *p1;

    // Clip at 95% of the frame maximum.  Using the observed max (rather than a
    // dtype constant) means the clip is set once from the reference frame and
    // reused for all subsequent measure() calls, so the same star is clipped
    // identically in both the reference and current profiles.
    return { median, stddev, maxVal * 0.95 };
}

// ---------------------------------------------------------------------------
// Algorithm helpers
// ---------------------------------------------------------------------------

static void filterSpikes(std::vector<double> &prof, double spikeRatio)
{
    if (prof.size() < 3) return;
    std::vector<double> clean = prof;
    for (std::size_t i = 1; i + 1 < prof.size(); ++i)
    {
        double avg = (prof[i - 1] + prof[i + 1]) / 2.0;
        if (prof[i] > avg * spikeRatio)
            clean[i] = avg;
    }
    prof = clean;
}

// FFT-based 1-D phase correlation with parabolic sub-pixel refinement.
// Returns {shift_pixels, correlation_SNR}.
static std::pair<double, double> correlate(
    const std::vector<double> &ref,
    const std::vector<double> &curr,
    double tukeyAlpha, double lpCutoff)
{
    const std::size_t n = ref.size();
    if (n < 4) return {0.0, 0.0};

    // Tukey window to reduce spectral leakage.
    std::vector<double> r(n), c(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        double w = 1.0;
        double t = static_cast<double>(i) / (n - 1);
        if (t < tukeyAlpha / 2.0)
            w = 0.5 * (1.0 + std::cos(M_PI * (2.0 * t / tukeyAlpha - 1.0)));
        else if (t > 1.0 - tukeyAlpha / 2.0)
            w = 0.5 * (1.0 + std::cos(M_PI * (2.0 * t / tukeyAlpha - 2.0 / tukeyAlpha + 1.0)));
        r[i] = ref[i]  * w;
        c[i] = curr[i] * w;
    }

    using namespace pocketfft;
    shape_t  shape{n};
    stride_t si{sizeof(double)}, sc{sizeof(std::complex<double>)};
    std::vector<std::complex<double>> rf(n / 2 + 1), cf(n / 2 + 1), cp(n / 2 + 1);

    r2c(shape, si, sc, 0, FORWARD, r.data(), rf.data(), 1.0);
    r2c(shape, si, sc, 0, FORWARD, c.data(), cf.data(), 1.0);

    // Phase correlation with Hann roll-off low-pass filter.
    // A rectangular cutoff produces sinc sidelobes in the correlation output
    // that can become false peaks in sparse-star quadrants.  A Hann-shaped
    // roll-off eliminates the Gibbs phenomenon while preserving the passband.
    const double k_full = rf.size() * lpCutoff * 0.9;  // flat region
    const double k_cut  = rf.size() * lpCutoff;         // zero from here

    for (std::size_t i = 0; i < rf.size(); ++i)
    {
        cp[i] = cf[i] * std::conj(rf[i]);
        double m = std::abs(cp[i]);
        double ki = static_cast<double>(i);
        double filter;
        if (ki <= k_full)
            filter = 1.0;
        else if (ki >= k_cut)
            filter = 0.0;
        else
            filter = 0.5 * (1.0 + std::cos(M_PI * (ki - k_full) / (k_cut - k_full)));

        if (m > 1e-9) cp[i] = (cp[i] / m) * filter;
    }

    std::vector<double> out(n);
    c2r(shape, sc, si, 0, BACKWARD, cp.data(), out.data(), 1.0 / n);

    auto   max_it = std::max_element(out.begin(), out.end());
    int    peak   = static_cast<int>(std::distance(out.begin(), max_it));
    double pval   = *max_it;

    double avg = 0.0, var = 0.0;
    for (double v : out) avg += v;
    avg /= n;
    for (double v : out) var += (v - avg) * (v - avg);
    double snr = (pval - avg) / (std::sqrt(var / n) + 1e-9);

    double shift = peak;
    {
        // Use wrap-around neighbors so boundary peaks (e.g. index 0 for a small
        // negative shift) are refined correctly.
        int    pm1 = (peak == 0) ? static_cast<int>(n) - 1 : peak - 1;
        int    pp1 = (peak == static_cast<int>(n) - 1) ? 0 : peak + 1;
        double y1  = out[pm1], y2 = out[peak], y3 = out[pp1];
        double den = y1 - 2.0 * y2 + y3;
        if (std::abs(den) > 1e-6)
            shift += 0.5 * (y1 - y3) / den;
    }
    if (shift > static_cast<double>(n) / 2.0) shift -= n;

    return {shift, snr};
}

// ---------------------------------------------------------------------------
// Profile building
// ---------------------------------------------------------------------------

static ProfileSet buildProfiles(
    const double *buf, int w, int h,
    const FrameStats &stats, const Config &cfg)
{
    ProfileSet p;
    p.width  = w;
    p.height = h;

    const double threshold = stats.median + cfg.sigmaThreshold * stats.stddev;
    const int    qw        = static_cast<int>(w * (0.5 + cfg.quadrantOverlap / 2.0));
    const int    qh        = static_cast<int>(h * (0.5 + cfg.quadrantOverlap / 2.0));

    struct Range { int x0, y0, x1, y1; } qr[4] = {
        {0,      0,      qw, qh},
        {w - qw, 0,      w,  qh},
        {0,      h - qh, qw, h },
        {w - qw, h - qh, w,  h }
    };

    for (int i = 0; i < 4; ++i)
    {
        p.q[i].xProf.assign(qr[i].x1 - qr[i].x0, 0.0);
        p.q[i].yProf.assign(qr[i].y1 - qr[i].y0, 0.0);
    }

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            double val = buf[y * w + x];
            if (val < threshold) continue;
            if (val > stats.clip) val = stats.clip;
            val -= stats.median;

            for (int i = 0; i < 4; ++i)
            {
                if (x >= qr[i].x0 && x < qr[i].x1 && y >= qr[i].y0 && y < qr[i].y1)
                {
                    p.q[i].xProf[x - qr[i].x0] += val;
                    p.q[i].yProf[y - qr[i].y0] += val;
                }
            }
        }
    }

    const double mx = w / 2.0, my = h / 2.0;
    for (int i = 0; i < 4; ++i)
    {
        filterSpikes(p.q[i].xProf, cfg.spikeRatio);
        filterSpikes(p.q[i].yProf, cfg.spikeRatio);

        double xFlux = 0.0, xMom = 0.0;
        for (int k = 0; k < static_cast<int>(p.q[i].xProf.size()); ++k)
        {
            xFlux += p.q[i].xProf[k];
            xMom  += (qr[i].x0 + k - mx) * p.q[i].xProf[k];
        }
        if (xFlux > 0.0) p.q[i].xc = xMom / xFlux;

        double yFlux = 0.0, yMom = 0.0;
        for (int k = 0; k < static_cast<int>(p.q[i].yProf.size()); ++k)
        {
            yFlux += p.q[i].yProf[k];
            yMom  += (qr[i].y0 + k - my) * p.q[i].yProf[k];
        }
        if (yFlux > 0.0) p.q[i].yc = yMom / yFlux;
    }
    return p;
}

// ---------------------------------------------------------------------------
// Weighted least-squares solver (3-DoF or 4-DoF)
// ---------------------------------------------------------------------------

// Gaussian elimination with partial pivoting on a 4x5 augmented matrix [A|b].
// Solutions land in column 4.  Returns false if singular.
static bool gaussElim4(double A[4][5])
{
    for (int col = 0; col < 4; ++col)
    {
        int pivot = col;
        for (int row = col + 1; row < 4; ++row)
            if (std::abs(A[row][col]) > std::abs(A[pivot][col]))
                pivot = row;
        if (std::abs(A[pivot][col]) < 1e-12) return false;
        if (pivot != col)
            for (int j = col; j < 5; ++j) std::swap(A[pivot][j], A[col][j]);

        double inv = 1.0 / A[col][col];
        for (int row = col + 1; row < 4; ++row)
        {
            double f = A[row][col] * inv;
            for (int j = col; j < 5; ++j) A[row][j] -= f * A[col][j];
        }
    }
    for (int row = 3; row >= 0; --row)
    {
        A[row][4] /= A[row][row];
        for (int r = 0; r < row; ++r) A[r][4] -= A[r][row] * A[row][4];
    }
    return true;
}

static Transform solveTransform(
    const ProfileSet &refP, const ProfileSet &currP, const Config &cfg)
{
    Transform result;

    // Accumulate normal-equation terms for unknowns [dx, dy, ds, dtheta].
    // Observation model (small-angle, isotropic scale ds = scale - 1):
    //   x-shift[i] =  dx  +  ds * xi  -  dtheta * yi   (weight wx = SNR_x^2)
    //   y-shift[i] =  dy  +  ds * yi  +  dtheta * xi   (weight wy = SNR_y^2)
    double m_xx = 0;              // sum(wx)
    double m_yy = 0;              // sum(wy)
    double m_xs = 0;              // sum(wx * xi)           -- dx-ds coupling
    double m_xt = 0;              // sum(-wx * yi)          -- dx-dtheta coupling
    double m_ys = 0;              // sum(wy * yi)           -- dy-ds coupling
    double m_yt = 0;              // sum(wy * xi)           -- dy-dtheta coupling
    double m_ss = cfg.tikhonov;  // sum(wx*xi^2 + wy*yi^2) -- ds-ds (regularised)
    double m_st = 0;              // sum(xi*yi*(wy - wx))   -- ds-dtheta coupling
    double m_tt = cfg.tikhonov;  // sum(wx*yi^2 + wy*xi^2) -- dtheta-dtheta (regularised)
    double b_x  = 0;              // sum(wx * rx)
    double b_y  = 0;              // sum(wy * ry)
    double b_s  = 0;              // sum(wx*xi*rx + wy*yi*ry)
    double b_t  = 0;              // sum(-wx*yi*rx + wy*xi*ry)

    double minSNR = std::numeric_limits<double>::max();

    for (int i = 0; i < 4; ++i)
    {
        auto rx = correlate(refP.q[i].xProf, currP.q[i].xProf,
                            cfg.tukeyAlpha, cfg.lpCutoff);
        auto ry = correlate(refP.q[i].yProf, currP.q[i].yProf,
                            cfg.tukeyAlpha, cfg.lpCutoff);

        minSNR = std::min({minSNR, rx.second, ry.second});

        const double xi = refP.q[i].xc;
        const double yi = refP.q[i].yc;
        const double wx = rx.second * rx.second;
        const double wy = ry.second * ry.second;

        m_xx += wx;
        m_xs += wx * xi;
        m_xt += -wx * yi;
        m_yy += wy;
        m_ys += wy * yi;
        m_yt += wy * xi;
        m_ss += wx * xi * xi + wy * yi * yi;
        m_st += xi * yi * (wy - wx);
        m_tt += wx * yi * yi + wy * xi * xi;
        b_x  += rx.first * wx;
        b_y  += ry.first * wy;
        b_s  += wx * xi * rx.first + wy * yi * ry.first;
        b_t  += -wx * yi * rx.first + wy * xi * ry.first;
    }

    result.snr = minSNR;

    // Lever-arm power: signal available for estimating scale and rotation.
    // When guide stars cluster near the image center (xi, yi -> 0) these DoF
    // become unobservable and approach the Tikhonov floor.  Fall back to a
    // 2-DoF (translation-only) solve to avoid noise bleeding into dx/dy.
    const double leverPower = m_ss - cfg.tikhonov;  // accumulated star signal
    if (leverPower < 1e-4 * std::max(m_xx, m_yy))
    {
        result.dx = (m_xx > 1e-9) ? b_x / m_xx : 0.0;
        result.dy = (m_yy > 1e-9) ? b_y / m_yy : 0.0;
        return result;
    }

    if (cfg.detectScale)
    {
        // 4-DoF solve: [dx, dy, ds, dtheta] from the 8 quadrant profile shifts.
        // The observation model is exact -- each quadrant's profile shift equals
        // the flux-weighted average displacement at that quadrant's centroid.
        // Normal equations (symmetric, Tikhonov absorbed into m_ss/m_tt):
        //   [ m_xx   0     m_xs  m_xt ] [dx]     [b_x]
        //   [ 0      m_yy  m_ys  m_yt ] [dy]  =  [b_y]
        //   [ m_xs   m_ys  m_ss  m_st ] [ds]     [b_s]
        //   [ m_xt   m_yt  m_st  m_tt ] [dθ]     [b_t]
        double A[4][5] = {
            { m_xx,  0,     m_xs,  m_xt,  b_x },
            { 0,     m_yy,  m_ys,  m_yt,  b_y },
            { m_xs,  m_ys,  m_ss,  m_st,  b_s },
            { m_xt,  m_yt,  m_st,  m_tt,  b_t },
        };
        if (gaussElim4(A))
        {
            result.dx     = A[0][4];
            result.dy     = A[1][4];
            result.scale  = 1.0 + A[2][4];
            result.dtheta = A[3][4];
        }
        return result;
    }

    // 3-DoF solve (Cramer's rule on [dx, dy, dtheta]).
    // The Schur complement s33 measures how much independent rotation signal
    // the quadrant lever arms provide after projecting out translation.
    const double s33 = m_tt
        - (m_xx > 1e-9 ? m_xt * m_xt / m_xx : 0.0)
        - (m_yy > 1e-9 ? m_yt * m_yt / m_yy : 0.0);

    if (s33 < 1e-4 * std::max(m_xx, m_yy))
    {
        result.dx = (m_xx > 1e-9) ? b_x / m_xx : 0.0;
        result.dy = (m_yy > 1e-9) ? b_y / m_yy : 0.0;
        return result;
    }

    const double det = m_xx * (m_yy * m_tt - m_yt * m_yt) - m_xt * (m_yy * m_xt);
    if (std::abs(det) < 1e-9) return result;

    result.dx     = (b_x * (m_yy * m_tt - m_yt * m_yt) + m_xt * (b_y * m_yt - b_t * m_yy)) / det;
    result.dy     = (b_y * (m_xx * m_tt - m_xt * m_xt) + m_yt * (b_x * m_xt - b_t * m_xx)) / det;
    result.dtheta = (m_xx * (m_yy * b_t - b_y * m_yt)  - m_xt * b_x * m_yy)                 / det;
    return result;
}

// ---------------------------------------------------------------------------
// Guider implementation
// ---------------------------------------------------------------------------

struct Guider::Impl
{
    Config     cfg;
    ProfileSet refProfiles;
    FrameStats refStats;   // locked at setReference() time; reused by measure()
    bool       hasRef { false };
};

Guider::Guider(Config cfg)
    : m_impl(std::make_unique<Impl>())
{
    m_impl->cfg = cfg;
}

Guider::~Guider() = default;

void Guider::setReference(const double *pixels, int width, int height)
{
    m_impl->refStats    = computeStats(pixels, width * height);
    m_impl->refProfiles = buildProfiles(pixels, width, height, m_impl->refStats, m_impl->cfg);
    m_impl->hasRef      = true;
}

Transform Guider::measure(const double *pixels, int width, int height)
{
    if (!m_impl->hasRef) return {};
    // Reuse reference-frame stats so threshold and clip are identical for both
    // frames -- prevents background changes from causing systematic drift.
    auto curr = buildProfiles(pixels, width, height, m_impl->refStats, m_impl->cfg);
    return solveTransform(m_impl->refProfiles, curr, m_impl->cfg);
}

void Guider::reset()
{
    m_impl->refProfiles = ProfileSet{};
    m_impl->refStats    = FrameStats{};
    m_impl->hasRef      = false;
}

bool Guider::hasReference() const
{
    return m_impl->hasRef;
}

} // namespace Donuts
