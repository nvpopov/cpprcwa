#include <cpprcwa/kbloch.h>
#include <algorithm>
#include <cmath>

namespace cpprcwa {

std::pair<Eigen::Vector2d, Eigen::Vector2d>
Lattice_Reciprocate(const Eigen::Vector2d& L1, const Eigen::Vector2d& L2) {
    double d = L1[0] * L2[1] - L1[1] * L2[0];
    Eigen::Vector2d Lk1( L2[1] / d, -L2[0] / d);
    Eigen::Vector2d Lk2(-L1[1] / d,  L1[0] / d);
    return {Lk1, Lk2};
}

namespace {

struct GEntry {
    int i, j;
    double gl2;   // |G|^2
};

// Sort by |G|^2 ascending, then by (i, j) tuple for deterministic
// degenerate-group ordering.
bool gl2_less(const GEntry& a, const GEntry& b) {
    if (a.gl2 != b.gl2) return a.gl2 < b.gl2;
    if (a.i != b.i) return a.i < b.i;
    return a.j < b.j;
}

} // namespace

static std::pair<IntMatrix, int>
Gsel_circular(int nG, const Eigen::Vector2d& Lk1, const Eigen::Vector2d& Lk2) {
    // kbloch.py:80-131
    double u  = Lk1.norm();
    double v  = Lk2.norm();
    double uv = Lk1.dot(Lk2);
    double uxv = Lk1[0] * Lk2[1] - Lk1[1] * Lk2[0];

    double uv_over_uv = (u * v > 0.0) ? uv / (u * v) : 0.0;
    double sintheta2  = std::max(0.0, 1.0 - uv_over_uv * uv_over_uv);
    double sintheta   = std::sqrt(sintheta2);

    double circ_area   = nG * std::fabs(uxv);
    double circ_radius = std::sqrt(circ_area / M_PI) + u + v;

    int u_extent = 1 + static_cast<int>(circ_radius / (u * sintheta + 1e-300));
    int v_extent = 1 + static_cast<int>(circ_radius / (v * sintheta + 1e-300));

    std::vector<GEntry> entries;
    entries.reserve((2 * u_extent + 1) * (2 * v_extent + 1));
    for (int i = -u_extent; i <= u_extent; ++i) {
        for (int j = -v_extent; j <= v_extent; ++j) {
            double gi_x = i * Lk1[0] + j * Lk2[0];
            double gi_y = i * Lk1[1] + j * Lk2[1];
            double gl2 = gi_x * gi_x + gi_y * gi_y;
            entries.push_back({i, j, gl2});
        }
    }
    std::sort(entries.begin(), entries.end(), gl2_less);

    // Degeneracy handling (kbloch.py:119-124): trim boundary entries that
    // fall inside a partially-included degenerate shell. Scan from the top:
    // find the first index i where |Gl2[i]-Gl2[i-1]| > tol and keep entries
    // 0..i-1. (Matches Python exactly, including the Gl2[-1] wrap at i=0.)
    double tol = 1e-10 * std::max(u * u, v * v);
    int total = static_cast<int>(entries.size());
    int nG_cap = std::min(nG, total);
    int i = nG_cap - 1;
    for (; i >= 0; --i) {
        int prev = i - 1;
        if (prev < 0) prev = total - 1;   // Python negative-index wrap
        double gap = std::fabs(entries[i].gl2 - entries[prev].gl2);
        if (gap > tol) break;
    }
    int nG_out = i;
    if (nG_out < 0) nG_out = 0;

    IntMatrix G(nG_out, 2);
    for (int k = 0; k < nG_out; ++k) {
        G(k, 0) = entries[k].i;
        G(k, 1) = entries[k].j;
    }
    return {G, nG_out};
}

static std::pair<IntMatrix, int>
Gsel_parallelogramic(int nG, const Eigen::Vector2d& Lk1, const Eigen::Vector2d& Lk2) {
    // kbloch.py:49-78
    double u  = Lk1.norm();
    double v  = Lk2.norm();
    double uv = Lk1.dot(Lk2);
    double uxv = Lk1[0] * Lk2[1] - Lk1[1] * Lk2[0];

    // Choose NGroot = largest odd integer <= sqrt(nG).
    int NGroot = static_cast<int>(std::floor(std::sqrt((double)nG)));
    if (NGroot % 2 == 0) NGroot -= 1;
    if (NGroot < 1) NGroot = 1;
    int M = NGroot / 2;

    std::vector<GEntry> entries;
    entries.reserve(NGroot * NGroot);
    for (int i = -M; i <= M; ++i) {
        for (int j = -M; j <= M; ++j) {
            double gi_x = i * Lk1[0] + j * Lk2[0];
            double gi_y = i * Lk1[1] + j * Lk2[1];
            double gl2 = gi_x * gi_x + gi_y * gi_y;
            entries.push_back({i, j, gl2});
        }
    }
    std::sort(entries.begin(), entries.end(), gl2_less);
    int nG_out = std::min((int)entries.size(), NGroot * NGroot);

    IntMatrix G(nG_out, 2);
    for (int k = 0; k < nG_out; ++k) {
        G(k, 0) = entries[k].i;
        G(k, 1) = entries[k].j;
    }
    (void)u; (void)v; (void)uv; (void)uxv;
    return {G, nG_out};
}

std::pair<IntMatrix, int>
Lattice_getG(int nG, const Eigen::Vector2d& Lk1, const Eigen::Vector2d& Lk2,
             int method) {
    if (method == 0) return Gsel_circular(nG, Lk1, Lk2);
    return Gsel_parallelogramic(nG, Lk1, Lk2);
}

void Lattice_SetKs(const IntMatrix& G,
                   complex kx0, complex ky0,
                   const Eigen::Vector2d& Lk1, const Eigen::Vector2d& Lk2,
                   ComplexVector& kx, ComplexVector& ky) {
    int n = G.rows();
    kx.resize(n);
    ky.resize(n);
    for (int i = 0; i < n; ++i) {
        kx(i) = kx0 + 2.0 * M_PI * (Lk1[0] * G(i, 0) + Lk2[0] * G(i, 1));
        ky(i) = ky0 + 2.0 * M_PI * (Lk1[1] * G(i, 0) + Lk2[1] * G(i, 1));
    }
}

} // namespace cpprcwa
