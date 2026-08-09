#include <cpprcwa/rcwa.h>
#include <cpprcwa/kbloch.h>
#include <cpprcwa/fft_funs.h>
#include "internal/branch_cut.h"
#include "internal/lapack_wrappers.h"
#include "internal/utils.h"
#include <cpprcwa/errors.h>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace cpprcwa {

RCWA::RCWA(const RCWAConfig& config)
    : nG_req_(config.nG),
      nG_(0),
      freq_(config.freq),
      omega_(2.0 * M_PI * config.freq),
      L1_(config.L1),
      L2_(config.L2),
      theta_(config.theta),
      phi_(config.phi),
      quasi1d_(config.quasi1d),
      report_memory_(config.report_memory) {}
RCWA::RCWA(int nG, const std::vector<double>& L1,
           const std::vector<double>& L2, complex freq, double theta,
           double phi, int verbose, bool quasi1d)
    : RCWA(RCWAConfig{nG, Eigen::Vector2d(L1[0], L1[1]),
                      Eigen::Vector2d(L2[0], L2[1]), freq, theta, phi,
                      verbose, quasi1d}) {}

RCWA::~RCWA() = default;

void RCWA::Add_LayerUniform(double thickness, complex epsilon) {
    layer_types_.push_back(LayerType::Uniform);
    thickness_.push_back(thickness);
    uniform_eps_.push_back(epsilon);
    material_idx_.push_back(uniform_eps_.size() - 1);  // uniform_idx
    grid_idx_.push_back(-1);
}

void RCWA::Add_LayerGrid(double thickness, int Nx, int Ny) {
    layer_types_.push_back(LayerType::Grid);
    thickness_.push_back(thickness);
    grid_Nxy_.emplace_back(Nx, Ny);
    material_idx_.push_back(patterned_count_);  // patterned_idx
    grid_idx_.push_back(grid_Nxy_.size() - 1);
    ++patterned_count_;
}

void RCWA::Init_Setup(double Pscale, int Gmethod) {
    Pscale_ = Pscale;
    Gmethod_ = Gmethod;
    smatrix_cache_ = SMatrixCache{};   // nG / layer structure may change

    if (layer_types_.empty() || layer_types_[0] != LayerType::Uniform)
        throw error::ConfigError("layer 0 must be Uniform (rcwa.py:89-103)");

    // Reciprocal lattice
    auto [Lk1_raw, Lk2_raw] = Lattice_Reciprocate(L1_, L2_);
    Lk1_ = Lk1_raw / Pscale;
    Lk2_ = Lk2_raw / Pscale;

    auto [Gmat, nGout] = Lattice_getG(nG_req_, Lk1_, Lk2_, Gmethod);
    G_ = std::move(Gmat);
    nG_ = nGout;

    if (quasi1d_) {
        // Restrict to the x-only harmonic row (i, j=0). Exact for y-invariant
        // structures (the y≠0 harmonics decouple and stay at zero amplitude);
        // shrinks every matrix to the pure-1D set.
        int keep = 0;
        for (int r = 0; r < nG_; ++r)
            if (G_(r, 1) == 0) ++keep;
        IntMatrix G1(keep, 2);
        int k = 0;
        for (int r = 0; r < nG_; ++r)
            if (G_(r, 1) == 0) G1.row(k++) = G_.row(r);
        G_ = std::move(G1);
        nG_ = keep;
    }

    // kx0, ky0 from layer-0 epsilon
    complex eps0 = uniform_eps_.front();
    complex kx0 = omega_ * std::sin(theta_) * std::cos(phi_) * std::sqrt(eps0);
    complex ky0 = omega_ * std::sin(theta_) * std::sin(phi_) * std::sqrt(eps0);
    Lattice_SetKs(G_, kx0, ky0, Lk1_, Lk2_, kx_, ky_);

    // The quasi-1D uniform-layer S-matrix is block-diagonal with 2×2 (Ex,Ey)
    // blocks per harmonic for ANY ky0: kp is block-diagonal (off-diagonal Ex-Ey
    // coupling ∝ kx·ky appears when ky0≠0), q is duplicated [q;q], phi=I. So
    // the per-harmonic 2×2 recursion below is exact and always usable. When
    // ky0==0 the blocks are diagonal, and the cheaper scalar recursion applies.
    quasi1d_fastpath_ = quasi1d_;
    quasi1d_diagonal_ = quasi1d_ && (std::abs(ky0) < 1e-12 * std::abs(omega_));

    // Per-layer kp/q/phi. Uniform layers SHARE storage across identical eps
    // values (they depend only on eps + the global kx/ky/omega) and share one
    // Identity phi, so periodic stacks (Mo/Si EUV multilayers) do not hold a
    // full (2nG)² copy per layer. Patterned layers are filled in
    // GridLayer_geteps().
    int nLayers = Layer_N();
    kp_list_.assign(nLayers, nullptr);
    q_list_.assign(nLayers, nullptr);
    phi_list_.assign(nLayers, nullptr);

    normalization_ = std::sqrt(eps0.real()) / std::cos(theta_);   // rcwa.py:103

    // The uniform eigensystem always yields phi = I, so one shared Identity
    // covers every uniform layer.
    struct EpsKey {
        complex eps;
        bool operator<(const EpsKey& o) const {
            if (eps.real() != o.eps.real()) return eps.real() < o.eps.real();
            return eps.imag() < o.eps.imag();
        }
    };
    std::map<EpsKey, std::pair<std::shared_ptr<const ComplexMatrix>,
                               std::shared_ptr<const ComplexVector>>> uni_cache;

    std::shared_ptr<const ComplexMatrix> I_shared;
    int uniform_idx = 0;
    for (int li = 0; li < nLayers; ++li) {
        if (layer_types_[li] != LayerType::Uniform) continue;
        complex eps = uniform_eps_[uniform_idx];
        EpsKey key{eps};
        auto it = uni_cache.find(key);
        if (it == uni_cache.end()) {
            auto kp = std::make_shared<ComplexMatrix>();
            auto q  = std::make_shared<ComplexVector>();
            ComplexMatrix phi;   // discarded — always I for uniform layers
            MakeKPMatrix_uniform(omega_, kx_, ky_, eps, *kp);
            SolveLayerEigensystem_uniform(omega_, kx_, ky_, eps, *q, phi);
            it = uni_cache.emplace(key, std::make_pair(kp, q)).first;
        }
        kp_list_[li] = it->second.first;
        q_list_[li]  = it->second.second;
        if (!I_shared)
            I_shared = std::make_shared<ComplexMatrix>(ComplexMatrix::Identity(2 * nG_, 2 * nG_));
        phi_list_[li] = I_shared;
        ++uniform_idx;
    }

    if (report_memory_) PrintMemoryReport();
}

void RCWA::PrintMemoryReport() const {    const int64_t cplx = static_cast<int64_t>(sizeof(complex));      // 16
    const int64_t n    = nG_;
    const int64_t n2   = 2 * n;
    const int64_t full_mat = n2 * n2 * cplx;   // bytes per 2nG×2nG complex matrix

    // ── Persistent storage (RCWA object lifetime) ──
    // Uniform layers DEDUPLICATE kp/q across identical ε values and all share
    // one Identity phi; only patterned layers own a full (kp, phi) each.
    int64_t persistent = 0;
    int nUniform = 0;
    struct EpsCmp {
        bool operator()(const complex& a, const complex& b) const {
            if (a.real() != b.real()) return a.real() < b.real();
            return a.imag() < b.imag();
        }
    };
    std::set<complex, EpsCmp> distinct_eps;
    for (int li = 0; li < Layer_N(); ++li) {
        if (layer_types_[li] == LayerType::Grid) {
            persistent += 2 * full_mat + n2 * cplx;   // kp + phi, q (own)
            persistent += n * n * cplx + full_mat;    // patterned epinv + eps2
        } else {
            ++nUniform;
            distinct_eps.insert(uniform_eps_[material_idx_[li]]);
        }
    }
    persistent += static_cast<int64_t>(distinct_eps.size()) * full_mat;  // one kp per distinct ε
    persistent += static_cast<int64_t>(distinct_eps.size()) * n2 * cplx; // one q per distinct ε
    if (nUniform > 0) persistent += full_mat;          // single shared Identity phi
    persistent += n * 2 * 4;                          // G_ (nG×2 int)
    persistent += 2 * n * cplx;                       // kx_, ky_
    persistent += 2 * n2 * cplx;                      // a0_, bN_

    // ── Uniform-pair T-matrix cache ──
    // Distinct (eps_l, eps_lp1) across consecutive uniform-uniform interfaces;
    // each distinct pair caches T11 + T12 (2 full matrices).
    int64_t pair_cache = 0;
    {
        struct EpsPairKey {
            complex el, elp;
            bool operator<(const EpsPairKey& o) const {
                if (el.real() != o.el.real()) return el.real() < o.el.real();
                if (el.imag() != o.el.imag()) return el.imag() < o.el.imag();
                if (elp.real() != o.elp.real()) return elp.real() < o.elp.real();
                return elp.imag() < o.elp.imag();
            }
        };
        std::set<EpsPairKey> seen;
        for (int li = 0; li + 1 < Layer_N(); ++li) {
            if (layer_types_[li] != LayerType::Uniform ||
                layer_types_[li + 1] != LayerType::Uniform)
                continue;
            seen.insert({uniform_eps_[material_idx_[li]],
                         uniform_eps_[material_idx_[li + 1]]});
        }
        pair_cache = 2 * full_mat * static_cast<int64_t>(seen.size());
    }

    // ── Transient peaks ──
    // GetSMatrix: one star-product step keeps ~20 live 2nG×2nG matrices
    // (4 accumulated S + 2 phase d + 2 interface T + 8 star temporaries + a
    // few inverse/product workspaces). Bounds the quasi-1D path too, whose
    // all-uniform suffix is O(n) per harmonic (the grid-interface prefix still
    // runs the general step at the reduced nG).
    const int64_t smatrix_peak = 20 * full_mat;
    // Volume_integral (block-wise, PLAN.md §10.4): never materializes 4nG×4nG
    // or 3nG×3nG matrices. Peak is ~7 live 2nG×2nG matrices (Faxy, Mxy, C/D,
    // A/B, Maa/Mab, W_A/W_B, qi/qj) + the nG×2nG Faz.
    const int64_t volume_peak =
        7 * (2 * n) * (2 * n) * cplx
        + (n) * (2 * n) * cplx;

    auto human = [](int64_t b) {
        char buf[64];
        if (b >= (int64_t)1 << 30)
            std::snprintf(buf, sizeof buf, "%.2f GiB", b / (double)((int64_t)1 << 30));
        else if (b >= (int64_t)1 << 20)
            std::snprintf(buf, sizeof buf, "%.2f MiB", b / (double)((int64_t)1 << 20));
        else if (b >= (int64_t)1 << 10)
            std::snprintf(buf, sizeof buf, "%.2f KiB", b / (double)((int64_t)1 << 10));
        else
            std::snprintf(buf, sizeof buf, "%lld B", (long long)b);
        return std::string(buf);
    };

    std::printf("Memory report (nG=%d, %d layers%s):\n",
                (int)n, Layer_N(), quasi1d_ ? ", quasi-1D" : "");
    std::printf("  persistent layer storage  : %14s  (%lld B)\n",
                human(persistent).c_str(), (long long)persistent);
    std::printf("  uniform-pair T cache      : %14s  (%lld B)\n",
                human(pair_cache).c_str(), (long long)pair_cache);
    std::printf("  RT_Solve transient peak   : %14s  (%lld B)\n",
                human(smatrix_peak).c_str(), (long long)smatrix_peak);
    std::printf("  Volume_integral transient : %14s  (%lld B)  [if used]\n",
                human(volume_peak).c_str(), (long long)volume_peak);
    std::printf("  estimated peak RSS (RT)   : %14s  (%lld B)\n",
                human(persistent + pair_cache + smatrix_peak).c_str(),
                (long long)(persistent + pair_cache + smatrix_peak));
    std::printf("  estimated peak RSS (VolIn): %14s  (%lld B)\n",
                human(persistent + pair_cache + volume_peak).c_str(),
                (long long)(persistent + pair_cache + volume_peak));
}

void RCWA::MakeExcitationPlanewave(const PlaneWaveExcitation& exc) {
    direction_ = exc.direction;
    int n2 = 2 * nG_;
    a0_ = ComplexVector::Zero(n2);
    bN_ = ComplexVector::Zero(n2);
    int idx = exc.order;
    if (idx < 0 || idx >= nG_) throw error::ConfigError("excitation order out of range");

    // Polarization basis mapping (rcwa.py:125-153). The (p, s) amplitudes are
    // projected onto the (x, y) Fourier-amplitude basis — NOT a raw a0[order]=p_amp.
    double ct = std::cos(theta_), cp = std::cos(phi_);
    double sp = std::sin(phi_);
    complex eip = std::polar(1.0, exc.p_phase);
    complex eis = std::polar(1.0, exc.s_phase);
    complex a_x = -exc.s_amp * ct * cp * eis - exc.p_amp * sp * eip;
    complex a_y = -exc.s_amp * ct * sp * eis + exc.p_amp * cp * eip;

    if (exc.direction == Direction::Forward) {
        a0_(idx)        = a_x;
        a0_(idx + nG_)  = a_y;
    } else {
        bN_(idx)        = a_x;
        bN_(idx + nG_)  = a_y;
    }
}

void RCWA::GridLayer_geteps(const std::vector<complex>& ep_all_isotropic) {
    int grid_idx = 0;
    size_t offset = 0;
    smatrix_cache_ = SMatrixCache{};   // patterned-layer eps changed
    for (int li = 0; li < Layer_N(); ++li) {
        if (layer_types_[li] != LayerType::Grid) continue;
        auto [Nx, Ny] = grid_Nxy_[grid_idx];
        double dN = 1.0 / ((double)Nx * Ny);
        std::vector<complex> slice(ep_all_isotropic.begin() + offset,
                                   ep_all_isotropic.begin() + offset + (size_t)Nx * Ny);
        auto t_e0 = std::chrono::steady_clock::now();
        auto result = Epsilon_fft(dN, slice, Nx, Ny, G_);
        auto t_e1 = std::chrono::steady_clock::now();
        patterned_epinv_.push_back(result.epsinv);
        patterned_ep2_.push_back(result.eps2);
        // Fill kp/q/phi for this layer (patterned layers own their matrices).
        auto kp = std::make_shared<ComplexMatrix>();
        auto q  = std::make_shared<ComplexVector>();
        auto phi = std::make_shared<ComplexMatrix>();
        MakeKPMatrix_patterned(omega_, kx_, ky_, result.epsinv, result.eps2, *kp);
        auto t_e2 = std::chrono::steady_clock::now();
        SolveLayerEigensystem_patterned(omega_, kx_, ky_, *kp, result.eps2,
                                        *q, *phi);
        kp_list_[li] = kp;
        q_list_[li]  = q;
        phi_list_[li] = phi;
        auto t_e3 = std::chrono::steady_clock::now();
        if (std::getenv("CPPRCWA_TIMING")) {
            using ms = std::chrono::duration<double, std::milli>;
            std::fprintf(stderr, "[grid] Epsilon_fft=%.1f MakeKP=%.1f eig=%.1f ms\n",
                         ms(t_e1-t_e0).count(), ms(t_e2-t_e1).count(), ms(t_e3-t_e2).count());
        }
        offset += (size_t)Nx * Ny;
        ++grid_idx;
    }
}

void RCWA::GridLayer_geteps(const std::vector<std::vector<complex>>& ep_all_anisotropic) {
    (void)ep_all_anisotropic;
    throw error::NotImplementedError("anisotropic GridLayer_geteps multi-layer path");
}

// ── KP / eigensystem ────────────────────────────────────────────────────────

void RCWA::MakeKPMatrix_uniform(complex omega, const ComplexVector& kx, const ComplexVector& ky,
                                complex eps, ComplexMatrix& kp) {
    int nG = kx.size();
    ComplexMatrix I = ComplexMatrix::Identity(2 * nG, 2 * nG);
    // Jk = vstack(diag(-ky), diag(kx))  -> (2nG, nG). MUST be zero-initialized:
    // only the diagonal is set, and the off-diagonal garbage would pollute the
    // matrix product Jk * Jk.transpose() (reads every column of Jk).
    ComplexMatrix Jk = ComplexMatrix::Zero(2 * nG, nG);
    Jk.topRows(nG).diagonal()    = -ky;
    Jk.bottomRows(nG).diagonal() =  kx;
    complex epinv = complex(1.0, 0.0) / eps;
    kp = omega * omega * I - epinv * Jk * Jk.transpose();
}

void RCWA::MakeKPMatrix_patterned(complex omega, const ComplexVector& kx, const ComplexVector& ky,
                                  const ComplexMatrix& epinv, const ComplexMatrix& ep2,
                                  ComplexMatrix& kp) {
    (void)ep2;  // ep2 unused in kp (matches rcwa.py MakeKPMatrix)
    int nG = kx.size();
    ComplexMatrix I = ComplexMatrix::Identity(2 * nG, 2 * nG);
    ComplexMatrix Jk = ComplexMatrix::Zero(2 * nG, nG);
    Jk.topRows(nG).diagonal()    = -ky;
    Jk.bottomRows(nG).diagonal() =  kx;
    // kp = omega^2 * I - (Jk * epinv) * Jk^T
    kp = omega * omega * I - (Jk * epinv) * Jk.transpose();
}

void RCWA::SolveLayerEigensystem_uniform(complex omega, const ComplexVector& kx, const ComplexVector& ky,
                                         complex eps, ComplexVector& q, ComplexMatrix& phi) {
    int nG = kx.size();
    ComplexVector q_raw(nG);
    for (int i = 0; i < nG; ++i) {
        q_raw(i) = std::sqrt(eps * omega * omega - kx(i) * kx(i) - ky(i) * ky(i));
    }
    q_raw = apply_branch_cut(q_raw);
    // Duplicate: q = [q; q], phi = I_(2nG)
    q.resize(2 * nG);
    q.head(nG) = q_raw;
    q.tail(nG) = q_raw;
    phi = ComplexMatrix::Identity(2 * nG, 2 * nG);
}

void RCWA::SolveLayerEigensystem_patterned(complex omega, const ComplexVector& kx, const ComplexVector& ky,
                                           const ComplexMatrix& kp, const ComplexMatrix& ep2,
                                           ComplexVector& q, ComplexMatrix& phi) {
    (void)omega;  // unused (matches rcwa.py SolveLayerEigensystem)
    int nG = kx.size();
    int n2 = 2 * nG;
    // k = vstack(diag(kx), diag(ky))  -> (2nG, nG); MUST be zero-initialized.
    // kkT = k * k^T then has cross terms diag(kx*ky) in the off-diagonal blocks
    // (same trap as the Jk construction in MakeKPMatrix_*).
    ComplexMatrix k_mat = ComplexMatrix::Zero(2 * nG, nG);
    k_mat.topRows(nG).diagonal()    = kx;
    k_mat.bottomRows(nG).diagonal() = ky;
    auto tt0 = std::chrono::steady_clock::now();
    ComplexMatrix kkT = k_mat * k_mat.transpose();
    ComplexMatrix M = ep2 * kp - kkT;
    auto tt1 = std::chrono::steady_clock::now();

    // Block-diagonal M detection (quasi-1D gratings at normal incidence,
    // ky0=0): M = diag(C·ω²−diag(kx²), C·(ω²I−diag(kx)epinv diag(kx))) — the
    // off-diagonal nG×nG blocks vanish exactly, decoupling the 2nG eigenproblem
    // into two nG ones (4× fewer zgeev flops). Falls back to the full 2nG
    // solve otherwise.
    //
    // Block-LOWER-TRIANGULAR M detection (quasi-1D gratings at OBLIQUE
    // incidence, ky0≠0): with the 1D harmonic set (ky constant) and epinv=C⁻¹,
    // the top-right block M12 = ky0·(C·epinv−I)·diag(kx) = 0 exactly, but M21 =
    // ky0·(C·diag(kx)·C⁻¹−diag(kx)) ≠ 0. det(M−λI) = det(M11−λI)·det(M22−λI),
    // so the eigenvalues still split into two nG problems. The eigenvectors are
    // coupled: phi = [[phA, 0],[V, phC]] with V = −phC·((phC⁻¹·M21·phA) ⊘ D),
    // D[i,j] = λC[i]−λA[j]. Verified identical to the full 2nG solve to ~1e-14
    // (numpy + C++).
    double offTR = 0.0, offBL = 0.0, on = 0.0;
    for (int i = 0; i < nG; ++i) {
        for (int j = 0; j < nG; ++j) {
            offTR = std::max(offTR, std::abs(M(i, nG + j)));
            offBL = std::max(offBL, std::abs(M(nG + i, j)));
            on    = std::max(on,    std::abs(M(i, j)));
            on    = std::max(on,    std::abs(M(nG + i, nG + j)));
        }
    }
    q.resize(n2);
    phi.resize(n2, n2);
    const double tol = 1e-10 * on + 1e-30;
    const bool block_diag = (std::max(offTR, offBL) <= tol);
    const bool block_tri  = (!block_diag && offTR <= tol &&
                            !std::getenv("CPPRCWA_NO_BLOCKTRI"));

    bool solved = false;
    if (block_diag) {
        ComplexVector ev1(nG), ev2(nG);
        ComplexMatrix ph1(nG, nG), ph2(nG, nG);
        ComplexMatrix M1 = M.topLeftCorner(nG, nG);
        ComplexMatrix M2 = M.bottomRightCorner(nG, nG);
        internal::zgeev(nG, M1.data(), (int)M1.outerStride(), ev1,
                        ph1.data(), (int)ph1.outerStride());
        internal::zgeev(nG, M2.data(), (int)M2.outerStride(), ev2,
                        ph2.data(), (int)ph2.outerStride());
        q.head(nG) = ev1;
        q.tail(nG) = ev2;
        phi.setZero(n2, n2);
        phi.topLeftCorner(nG, nG)     = ph1;
        phi.bottomRightCorner(nG, nG) = ph2;
        solved = true;
    } else if (block_tri) {
        ComplexVector lamA(nG), lamC(nG);
        ComplexMatrix phA(nG, nG), phC(nG, nG);
        ComplexMatrix M11 = M.topLeftCorner(nG, nG);
        ComplexMatrix M22 = M.bottomRightCorner(nG, nG);
        internal::zgeev(nG, M11.data(), (int)M11.outerStride(), lamA,
                        phA.data(), (int)phA.outerStride());
        internal::zgeev(nG, M22.data(), (int)M22.outerStride(), lamC,
                        phC.data(), (int)phC.outerStride());
        // X = phC⁻¹ · M21 · phA  (solve phC·X = M21·phA; LU overwrites a copy)
        ComplexMatrix Rhs = M.bottomLeftCorner(nG, nG) * phA;
        ComplexMatrix phC_lu = phC;
        std::vector<int> piv = internal::zgetrf_factor(
            nG, phC_lu.data(), (int)phC_lu.outerStride());
        internal::zgetrs_solve(nG, nG, phC_lu.data(), (int)phC_lu.outerStride(),
                               piv.data(), Rhs.data(), (int)Rhs.outerStride());
        // D[i,j] = λC[i] − λA[j]; guard against TE/TM degeneracy (D ≈ 0).
        ComplexMatrix D = lamC.replicate(1, nG)
                          - lamA.transpose().replicate(nG, 1);
        const double lam_scale = std::max(lamA.cwiseAbs().maxCoeff(),
                                          lamC.cwiseAbs().maxCoeff());
        if (D.cwiseAbs().minCoeff() < 1e-12 * lam_scale + 1e-300) {
            // ill-conditioned coupling — fall back to the full 2nG solve.
            ComplexVector evals(n2);
            internal::zgeev(n2, M.data(), (int)M.outerStride(), evals,
                            phi.data(), (int)phi.outerStride());
            q = evals;
            solved = true;
        } else {
            // V = −phC · (X ⊘ D)
            ComplexMatrix W = Rhs.cwiseQuotient(D);
            ComplexMatrix V = -(phC * W);
            q.head(nG) = lamA;
            q.tail(nG) = lamC;
            phi.setZero(n2, n2);
            phi.topLeftCorner(nG, nG)     = phA;
            phi.bottomLeftCorner(nG, nG)  = V;
            phi.bottomRightCorner(nG, nG) = phC;
            solved = true;
        }
    }
    if (!solved) {
        ComplexVector evals(n2);
        internal::zgeev(n2, M.data(), (int)M.outerStride(), evals,
                        phi.data(), (int)phi.outerStride());
        q = evals;
    }
    auto tt2 = std::chrono::steady_clock::now();
    if (std::getenv("CPPRCWA_TIMING")) {
        using ms = std::chrono::duration<double, std::milli>;
        std::fprintf(stderr, "[eig] buildM=%.1f zgeev=%.1f ms%s\n",
                     ms(tt1-tt0).count(), ms(tt2-tt1).count(),
                     block_diag ? " (block-diag)" : (block_tri ? " (block-tri)" : ""));
    }
    // q = sqrt(evals) with branch cut
    for (int i = 0; i < n2; ++i) q(i) = std::sqrt(q(i));
    q = apply_branch_cut(q);
}

// ── S-matrix (Phase 5 — stub for now) ───────────────────────────────────────

// Redheffer combination of S_L = S(indi, m) and S_R = S(m, indj), which SHARE
// the boundary layer m (the "overlapping cascade" — identical to the formula
// used by the quasi-1D fast path and validated to ~1e-13 vs grcwa). Returns
// S(indi, indj). Used by the periodic uniform-suffix exponentiation.
namespace {
void redheffer_cascade(const ComplexMatrix& L11, const ComplexMatrix& L12,
                       const ComplexMatrix& L21, const ComplexMatrix& L22,
                       const ComplexMatrix& R11, const ComplexMatrix& R12,
                       const ComplexMatrix& R21, const ComplexMatrix& R22,
                       ComplexMatrix& S11, ComplexMatrix& S12,
                       ComplexMatrix& S21, ComplexMatrix& S22) {
    int n2 = (int)L11.rows();
    ComplexMatrix M = internal::zinverse(
        ComplexMatrix::Identity(n2, n2) - L12 * R21);
    ComplexMatrix ML11 = M * L11;
    ComplexMatrix ML12R22 = M * (L12 * R22);
    S11 = R11 * ML11;
    S12 = R11 * ML12R22 + R12;
    S21 = L21 + L22 * (R21 * ML11);
    S22 = L22 * (R21 * ML12R22) + L22 * R22;
}
} // namespace

void RCWA::GetSMatrix(int indi, int indj,
                      ComplexMatrix& S11, ComplexMatrix& S12,
                      ComplexMatrix& S21, ComplexMatrix& S22) {
    int n2 = 2 * nG_;
    // Memoize the most recent (non-trivial) range: RT_Solve then field
    // reconstruction both ask for S(0, Layer_N-1).
    if (smatrix_cache_.indi == indi && smatrix_cache_.indj == indj) {
        S11 = smatrix_cache_.S11;
        S12 = smatrix_cache_.S12;
        S21 = smatrix_cache_.S21;
        S22 = smatrix_cache_.S22;
        return;
    }
    if (indi == indj) {
        S11 = ComplexMatrix::Identity(n2, n2);
        S22 = ComplexMatrix::Identity(n2, n2);
        S12 = ComplexMatrix::Zero(n2, n2);
        S21 = ComplexMatrix::Zero(n2, n2);
        return;
    }

    // One Redheffer star-product step: append the interface (l, l+1) to S.
    auto step = [&](int l, ComplexMatrix& s11, ComplexMatrix& s12,
                    ComplexMatrix& s21, ComplexMatrix& s22) {
        int lp1 = l + 1;
        const bool uniform_pair =
            layer_types_[l] == LayerType::Uniform &&
            layer_types_[lp1] == LayerType::Uniform;

        ComplexMatrix T11, T12;
        // Phase factors are DIAGONAL: keep them as vectors and scale with
        // .asDiagonal() (O(n²) row/column scaling) instead of dense matrices
        // (which would make d1·s12 a full O(n³) product).
        ComplexVector d1 = (complex(0,1) * q(l)   * thickness_[l]  ).array().exp().matrix();
        ComplexVector d2 = (complex(0,1) * q(lp1) * thickness_[lp1]).array().exp().matrix();

        if (uniform_pair) {
            // phi = I for uniform layers → Q = I. Cache T11/T12 per (eps_l, eps_lp1).
            UniformPairKey key{uniform_eps_[material_idx_[l]],
                               uniform_eps_[material_idx_[lp1]]};
            auto it = uniform_pair_cache_.find(key);
            if (it == uniform_pair_cache_.end()) {
                ComplexMatrix kp_l_inv = internal::zinverse(kp(l));   // = inv(kp_l·phi_l)
                ComplexVector qinv = q(lp1).cwiseInverse();
                ComplexMatrix P = q(l).asDiagonal() * kp_l_inv
                                  * kp(lp1) * qinv.asDiagonal();
                UniformPairCache c;
                c.T11 = 0.5 * (ComplexMatrix::Identity(n2, n2) + P);
                c.T12 = 0.5 * (ComplexMatrix::Identity(n2, n2) - P);
                it = uniform_pair_cache_.emplace(key, std::move(c)).first;
            }
            T11 = it->second.T11;
            T12 = it->second.T12;
        } else {
            ComplexMatrix Q = internal::zinverse(ph(l)) * ph(lp1);
            ComplexMatrix kpphi_l_inv = internal::zinverse(kp(l) * ph(l));
            ComplexVector qinv = q(lp1).cwiseInverse();
            ComplexMatrix P = q(l).asDiagonal() * kpphi_l_inv
                              * kp(lp1) * ph(lp1) * qinv.asDiagonal();
            T11 = 0.5 * (Q + P);
            T12 = 0.5 * (Q - P);
        }

        // Redheffer star product (S-update, sequential, cannot cache).
        // Common subexpressions d1·S12 and S22·T12 are computed once; the
        // inv(P1m) products are replaced by one LU factorization + two
        // back-substitutions.
        ComplexMatrix d1S12 = d1.asDiagonal() * s12;             // O(n²) (d1 diagonal)
        ComplexMatrix P1m = T11 - d1S12 * T12;                   // 1 full matmul
        std::vector<int> piv = internal::zgetrf_factor(n2, P1m.data(),
                                                       static_cast<int>(P1m.outerStride()));

        ComplexMatrix new_S11 = d1.asDiagonal() * s11;           // O(n²)
        internal::zgetrs_solve(n2, n2, P1m.data(), static_cast<int>(P1m.outerStride()),
                               piv.data(), new_S11.data(),
                               static_cast<int>(new_S11.outerStride()));

        ComplexMatrix P2 = d1S12 * T11 - T12;                    // 1 full matmul
        ComplexMatrix new_S12 = P2 * d2.asDiagonal();            // O(n²)
        internal::zgetrs_solve(n2, n2, P1m.data(), static_cast<int>(P1m.outerStride()),
                               piv.data(), new_S12.data(),
                               static_cast<int>(new_S12.outerStride()));

        ComplexMatrix S22T12 = s22 * T12;                        // 1 full matmul
        ComplexMatrix new_S21 = s21 + S22T12 * new_S11;          // 1 full matmul
        ComplexMatrix new_S22 = s22 * T11 * d2.asDiagonal() + S22T12 * new_S12;  // 1 full matmul + O(n²)
        s11 = std::move(new_S11);
        s12 = std::move(new_S12);
        s21 = std::move(new_S21);
        s22 = std::move(new_S22);
    };

    // Block-diagonality check: are the off-diagonal nG×nG blocks of M ~zero?
    // (True at ky0=0 for quasi-1D gratings and D4-symmetric patterns.)
    auto is_block_diag = [&](const ComplexMatrix& M) {
        double off = 0.0, on = 0.0;
        for (int i = 0; i < nG_; ++i)
            for (int j = 0; j < nG_; ++j) {
                off = std::max(off, std::abs(M(i, nG_ + j)));
                off = std::max(off, std::abs(M(nG_ + i, j)));
                on  = std::max(on,  std::abs(M(i, j)));
                on  = std::max(on,  std::abs(M(nG_ + i, nG_ + j)));
            }
        return off <= 1e-10 * on + 1e-30;
    };

    // Same Redheffer star step, but for ONE polarization block (p∈{0,1}) of the
    // block-diagonal (kp, phi, q) at ky0=0 — operates on nG×nG blocks (4× fewer
    // flops). Used by the TE/TM-decoupled quasi-1D path.
    auto bstep = [&](int l, int p, ComplexMatrix& s11, ComplexMatrix& s12,
                     ComplexMatrix& s21, ComplexMatrix& s22) {
        int lp1 = l + 1;
        int nb = nG_;
        ComplexVector d1 = (complex(0,1) * q(l).segment(p*nb, nb) * thickness_[l]).array().exp().matrix();
        ComplexVector d2 = (complex(0,1) * q(lp1).segment(p*nb, nb) * thickness_[lp1]).array().exp().matrix();
        ComplexMatrix phil   = ph(l).block(p*nb, p*nb, nb, nb);
        ComplexMatrix philp1 = ph(lp1).block(p*nb, p*nb, nb, nb);
        ComplexMatrix kpl    = kp(l).block(p*nb, p*nb, nb, nb);
        ComplexMatrix kplp1  = kp(lp1).block(p*nb, p*nb, nb, nb);
        // General interface (no uniform-pair cache in the block path).
        ComplexMatrix Q = internal::zinverse(phil) * philp1;
        ComplexMatrix kpphi_l_inv = internal::zinverse(kpl * phil);
        ComplexVector qinv = q(lp1).segment(p*nb, nb).cwiseInverse();
        ComplexMatrix P = q(l).segment(p*nb, nb).asDiagonal() * kpphi_l_inv
                          * kplp1 * philp1 * qinv.asDiagonal();
        ComplexMatrix T11 = 0.5 * (Q + P);
        ComplexMatrix T12 = 0.5 * (Q - P);
        ComplexMatrix d1S12 = d1.asDiagonal() * s12;
        ComplexMatrix P1m = T11 - d1S12 * T12;
        std::vector<int> piv = internal::zgetrf_factor(nb, P1m.data(),
                                                       (int)P1m.outerStride());
        ComplexMatrix new_S11 = d1.asDiagonal() * s11;
        internal::zgetrs_solve(nb, nb, P1m.data(), (int)P1m.outerStride(),
                               piv.data(), new_S11.data(), (int)new_S11.outerStride());
        ComplexMatrix P2 = d1S12 * T11 - T12;
        ComplexMatrix new_S12 = P2 * d2.asDiagonal();
        internal::zgetrs_solve(nb, nb, P1m.data(), (int)P1m.outerStride(),
                               piv.data(), new_S12.data(), (int)new_S12.outerStride());
        ComplexMatrix S22T12 = s22 * T12;
        ComplexMatrix new_S21 = s21 + S22T12 * new_S11;
        ComplexMatrix new_S22 = s22 * T11 * d2.asDiagonal() + S22T12 * new_S12;
        s11 = std::move(new_S11);
        s12 = std::move(new_S12);
        s21 = std::move(new_S21);
        s22 = std::move(new_S22);
    };

    // ── quasi-1D fast path ──
    // With the 1D harmonic set (j==0), every uniform-layer kp/q/phi is
    // block-diagonal with 2×2 (Ex,Ey) blocks per harmonic (exact for any ky0).
    // So the S-matrix of an all-uniform layer range is block-diagonal 2×2, and
    // the Redheffer recursion factorizes per harmonic. Split the stack at the
    // start of the trailing uniform suffix: compute the (grid + interface)
    // prefix with the general sequential loop, the uniform suffix with a 2×2
    // per-harmonic recursion, then assemble with the overlapping-cascade
    // formula.
    if (quasi1d_fastpath_) {
        int m = indj;
        while (m >= indi && layer_types_[m] == LayerType::Uniform) --m;
        ++m;   // layers [m, indj] are all uniform (if m <= indj)
        if (m <= indj) {
            auto tt0 = std::chrono::steady_clock::now();

            // ── TE/TM block-decoupled path (ky0=0) ──
            // When quasi1d_diagonal_ (ky0==0) AND every prefix layer's kp/phi
            // is block-diagonal (2× nG blocks), the whole S-matrix is
            // block-diagonal: the prefix interfaces and the cascade decouple
            // into two independent nG×nG (TE and TM) problems. The uniform
            // suffix is scalar per harmonic (both blocks in one pass).
            bool prefix_block_diag = quasi1d_diagonal_;
            if (prefix_block_diag) {
                for (int l = indi; l < m; ++l)
                    if (!is_block_diag(kp(l)) || !is_block_diag(ph(l))) {
                        prefix_block_diag = false;
                        break;
                    }
            }
            if (prefix_block_diag) {
                ComplexVector r11 = ComplexVector::Ones(n2);
                ComplexVector r22 = ComplexVector::Ones(n2);
                ComplexVector r12 = ComplexVector::Zero(n2);
                ComplexVector r21 = ComplexVector::Zero(n2);
                for (int l = m; l < indj; ++l) {
                    int lp1 = l + 1;
                    const ComplexVector& ql = q(l);
                    const ComplexVector& qb = q(lp1);
                    for (int h = 0; h < n2; ++h) {
                        complex d1 = std::exp(complex(0, 1) * ql(h) * thickness_[l]);
                        complex d2 = std::exp(complex(0, 1) * qb(h) * thickness_[lp1]);
                        complex kl = kp(l)(h, h), kb = kp(lp1)(h, h);
                        complex P = (ql(h) / kl) * (kb / qb(h));
                        complex T11 = 0.5 * (complex(1, 0) + P);
                        complex T12 = 0.5 * (complex(1, 0) - P);
                        complex P1m = T11 - d1 * r12(h) * T12;
                        complex ns11 = d1 * r11(h) / P1m;
                        complex ns12 = (d1 * r12(h) * T11 - T12) * d2 / P1m;
                        complex ns21 = r21(h) + r22(h) * T12 * ns11;
                        complex ns22 = r22(h) * (T11 * d2 + T12 * ns12);
                        r11(h) = ns11; r12(h) = ns12; r21(h) = ns21; r22(h) = ns22;
                    }
                }
                auto tt1 = std::chrono::steady_clock::now();
                ComplexMatrix S11r = ComplexMatrix::Zero(n2, n2);
                ComplexMatrix S22r = ComplexMatrix::Zero(n2, n2);
                ComplexMatrix S12r = ComplexMatrix::Zero(n2, n2);
                ComplexMatrix S21r = ComplexMatrix::Zero(n2, n2);
                for (int p = 0; p < 2; ++p) {
                    const int nb = nG_;
                    ComplexMatrix L11 = ComplexMatrix::Identity(nb, nb);
                    ComplexMatrix L22 = ComplexMatrix::Identity(nb, nb);
                    ComplexMatrix L12 = ComplexMatrix::Zero(nb, nb);
                    ComplexMatrix L21 = ComplexMatrix::Zero(nb, nb);
                    for (int l = indi; l < m; ++l) bstep(l, p, L11, L12, L21, L22);
                    ComplexMatrix R11 = r11.segment(p * nb, nb).asDiagonal();
                    ComplexMatrix R22 = r22.segment(p * nb, nb).asDiagonal();
                    ComplexMatrix R12 = r12.segment(p * nb, nb).asDiagonal();
                    ComplexMatrix R21 = r21.segment(p * nb, nb).asDiagonal();
                    ComplexMatrix Sp11, Sp12, Sp21, Sp22;
                    redheffer_cascade(L11, L12, L21, L22, R11, R12, R21, R22,
                                      Sp11, Sp12, Sp21, Sp22);
                    S11r.block(p * nb, p * nb, nb, nb) = Sp11;
                    S12r.block(p * nb, p * nb, nb, nb) = Sp12;
                    S21r.block(p * nb, p * nb, nb, nb) = Sp21;
                    S22r.block(p * nb, p * nb, nb, nb) = Sp22;
                }
                S11 = std::move(S11r); S12 = std::move(S12r);
                S21 = std::move(S21r); S22 = std::move(S22r);
                if (std::getenv("CPPRCWA_TIMING")) {
                    using ms = std::chrono::duration<double, std::milli>;
                    std::fprintf(stderr, "[S] TE/TM-decoupled: suffix=%.1f prefix+cascade=%.1f ms\n",
                                 ms(tt1-tt0).count(),
                                 ms(std::chrono::steady_clock::now()-tt1).count());
                }
                smatrix_cache_ = {indi, indj, S11, S12, S21, S22};
                return;
            }

            // ── General quasi-1D path (full 2nG prefix, per-harmonic suffix) ──
            // L = S(indi, m) — general prefix (may contain patterned layers).
            ComplexMatrix L11 = ComplexMatrix::Identity(n2, n2);
            ComplexMatrix L22 = ComplexMatrix::Identity(n2, n2);
            ComplexMatrix L12 = ComplexMatrix::Zero(n2, n2);
            ComplexMatrix L21 = ComplexMatrix::Zero(n2, n2);
            for (int l = indi; l < m; ++l) {
                auto ts0 = std::chrono::steady_clock::now();
                step(l, L11, L12, L21, L22);
                if (std::getenv("CPPRCWA_TIMING")) {
                    using ms = std::chrono::duration<double, std::milli>;
                    std::fprintf(stderr, "[prefix] layer %d->%d: %.1f ms\n", l, l+1,
                                 ms(std::chrono::steady_clock::now()-ts0).count());
                }
            }
            auto tt1 = std::chrono::steady_clock::now();

            // R = S(m, indj) — all-uniform suffix. For each harmonic h the
            // 2×2 block couples (Ex_h, Ey_h) only, so run a per-harmonic
            // recursion instead of the full (2nG)² matrices. When ky0==0 the
            // 2×2 blocks are diagonal and the cheaper scalar recursion applies.
            ComplexMatrix R11, R12, R21, R22;
            if (quasi1d_diagonal_) {
                ComplexVector r11 = ComplexVector::Ones(n2);
                ComplexVector r22 = ComplexVector::Ones(n2);
                ComplexVector r12 = ComplexVector::Zero(n2);
                ComplexVector r21 = ComplexVector::Zero(n2);
                for (int l = m; l < indj; ++l) {
                    int lp1 = l + 1;
                    const ComplexVector& ql = q(l);
                    const ComplexVector& qb = q(lp1);
                    for (int h = 0; h < n2; ++h) {
                        complex d1 = std::exp(complex(0, 1) * ql(h) * thickness_[l]);
                        complex d2 = std::exp(complex(0, 1) * qb(h) * thickness_[lp1]);
                        complex kl = kp(l)(h, h), kb = kp(lp1)(h, h);
                        complex P = (ql(h) / kl) * (kb / qb(h));
                        complex T11 = 0.5 * (complex(1, 0) + P);
                        complex T12 = 0.5 * (complex(1, 0) - P);
                        complex P1m = T11 - d1 * r12(h) * T12;
                        complex ns11 = d1 * r11(h) / P1m;
                        complex ns12 = (d1 * r12(h) * T11 - T12) * d2 / P1m;
                        complex ns21 = r21(h) + r22(h) * T12 * ns11;
                        complex ns22 = r22(h) * (T11 * d2 + T12 * ns12);
                        r11(h) = ns11; r12(h) = ns12; r21(h) = ns21; r22(h) = ns22;
                    }
                }
                R11 = r11.asDiagonal();
                R12 = r12.asDiagonal();
                R21 = r21.asDiagonal();
                R22 = r22.asDiagonal();
            } else {
                using M2 = Eigen::Matrix<complex, 2, 2>;
                std::vector<M2> r11(nG_), r12(nG_), r21(nG_), r22(nG_);
                const M2 I2 = M2::Identity();
                const M2 Z2 = M2::Zero();
                for (int h = 0; h < nG_; ++h) {
                    r11[h] = I2; r22[h] = I2; r12[h] = Z2; r21[h] = Z2;
                }
                for (int l = m; l < indj; ++l) {
                    int lp1 = l + 1;
                    const ComplexVector& ql = q(l);
                    const ComplexVector& qb = q(lp1);
                    const ComplexMatrix& kpl = kp(l);
                    const ComplexMatrix& kpb = kp(lp1);
                    const double tl = thickness_[l], tb = thickness_[lp1];
                    for (int h = 0; h < nG_; ++h) {
                        complex d1 = std::exp(complex(0, 1) * ql(h) * tl);
                        complex d2 = std::exp(complex(0, 1) * qb(h) * tb);
                        M2 Kl, Kb;
                        Kl << kpl(h, h),       kpl(h, nG_ + h),
                              kpl(nG_ + h, h),  kpl(nG_ + h, nG_ + h);
                        Kb << kpb(h, h),       kpb(h, nG_ + h),
                              kpb(nG_ + h, h),  kpb(nG_ + h, nG_ + h);
                        // P = q_l·kp_l⁻¹·kp_lp1·q_lp1⁻¹ (2×2), q scalar per harmonic.
                        M2 P = (ql(h) / qb(h)) * Kl.inverse() * Kb;
                        M2 T11 = 0.5 * (I2 + P);
                        M2 T12 = 0.5 * (I2 - P);
                        M2 P1m = T11 - d1 * r12[h] * T12;
                        M2 P1m_inv = P1m.inverse();
                        M2 ns11 = d1 * r11[h] * P1m_inv;
                        M2 ns12 = (d1 * r12[h] * T11 - T12) * d2 * P1m_inv;
                        M2 ns21 = r21[h] + r22[h] * T12 * ns11;
                        M2 ns22 = r22[h] * (T11 * d2 + T12 * ns12);
                        r11[h] = ns11; r12[h] = ns12; r21[h] = ns21; r22[h] = ns22;
                    }
                }
                // Scatter the per-harmonic 2×2 blocks into block-diagonal (2nG)²
                // matrices (ordering [Ex_0..Ex_{nG-1}, Ey_0..Ey_{nG-1}]).
                auto scatter = [&](const std::vector<M2>& v) {
                    ComplexMatrix out = ComplexMatrix::Zero(n2, n2);
                    for (int h = 0; h < nG_; ++h) {
                        out(h, h)             = v[h](0, 0);
                        out(h, nG_ + h)       = v[h](0, 1);
                        out(nG_ + h, h)       = v[h](1, 0);
                        out(nG_ + h, nG_ + h) = v[h](1, 1);
                    }
                    return out;
                };
                R11 = scatter(r11);
                R12 = scatter(r12);
                R21 = scatter(r21);
                R22 = scatter(r22);
            }
            auto tt2 = std::chrono::steady_clock::now();

            // Overlapping cascade: L = S(indi, m), R = S(m, indj) share layer m.
            // M = inv(I - L12·R21); S11=R11·M·L11, S12=R11·M·L12·R22+R12,
            // S21=L21+L22·R21·M·L11, S22=L22·R21·M·L12·R22+L22·R22.
            ComplexMatrix M = internal::zinverse(
                ComplexMatrix::Identity(n2, n2) - L12 * R21);
            ComplexMatrix ML11 = M * L11;
            ComplexMatrix ML12R22 = M * (L12 * R22);
            S11 = R11 * ML11;
            S12 = R11 * ML12R22 + R12;
            S21 = L21 + L22 * (R21 * ML11);
            S22 = L22 * (R21 * ML12R22) + L22 * R22;
            if (std::getenv("CPPRCWA_TIMING")) {
                using ms = std::chrono::duration<double, std::milli>;
                std::fprintf(stderr, "[S] prefix=%.1f suffix=%.1f cascade=%.1f ms\n",
                             ms(tt1-tt0).count(), ms(tt2-tt1).count(),
                             ms(std::chrono::steady_clock::now()-tt2).count());
            }
            smatrix_cache_ = {indi, indj, S11, S12, S21, S22};
            return;
        }
    }

    // ── General sequential path (with periodic uniform-suffix acceleration) ──
    S11 = ComplexMatrix::Identity(n2, n2);
    S22 = ComplexMatrix::Identity(n2, n2);
    S12 = ComplexMatrix::Zero(n2, n2);
    S21 = ComplexMatrix::Zero(n2, n2);

    auto g0 = std::chrono::steady_clock::now();
    auto build_seq = [&](int a, int b, ComplexMatrix& s11, ComplexMatrix& s12,
                         ComplexMatrix& s21, ComplexMatrix& s22) {
        s11 = ComplexMatrix::Identity(n2, n2);
        s22 = ComplexMatrix::Identity(n2, n2);
        s12 = ComplexMatrix::Zero(n2, n2);
        s21 = ComplexMatrix::Zero(n2, n2);
        for (int l = a; l < b; ++l) step(l, s11, s12, s21, s22);
    };

    // Trailing all-uniform suffix [m, indj).
    int m = indj;
    while (m >= indi && layer_types_[m] == LayerType::Uniform) --m;
    ++m;

    bool periodic_done = false;
    if (m <= indj) {
        auto layer_eq = [&](int a, int b) {
            return layer_types_[a] == LayerType::Uniform &&
                   layer_types_[b] == LayerType::Uniform &&
                   thickness_[a] == thickness_[b] &&
                   uniform_eps_[material_idx_[a]] == uniform_eps_[material_idx_[b]];
        };
        // Longest periodic run [s, s+R·L): period L, R repeats (e.g. 40×(Mo,Si)).
        int best_s = -1, best_L = 0, best_R = 0;
        for (int s = m; s < indj; ++s) {
            for (int L = 2; L <= (indj - s) / 2; ++L) {
                int r = 1;
                while (s + (r + 1) * L <= indj) {
                    bool ok = true;
                    for (int k = 0; k < L; ++k)
                        if (!layer_eq(s + r * L + k, s + k)) { ok = false; break; }
                    if (!ok) break;
                    ++r;
                }
                if (r >= 3 && r * L > best_R * best_L) { best_s = s; best_L = L; best_R = r; }
            }
        }
        if (best_R >= 3) {
            // One period: S(best_s, best_s+L). The period tiles best_R times
            // (layers best_s..best_s+best_R·L), but period^R would end with a
            // PHANTOM reference layer (the period's next starting layer), which
            // is wrong when the run's last boundary differs (e.g. the Si
            // substrate after the final Mo/Si period). So exponentiate
            // period^(R-1) and cascade with the LAST period + its real
            // boundary interface S(s+(R-1)L, s+R·L) computed directly.
            ComplexMatrix P11, P12, P21, P22;
            build_seq(best_s, best_s + best_L, P11, P12, P21, P22);
            ComplexMatrix C11 = ComplexMatrix::Identity(n2, n2);
            ComplexMatrix C22 = ComplexMatrix::Identity(n2, n2);
            ComplexMatrix C12 = ComplexMatrix::Zero(n2, n2);
            ComplexMatrix C21 = ComplexMatrix::Zero(n2, n2);
            ComplexMatrix B11 = P11, B12 = P12, B21 = P21, B22 = P22;
            int rem = best_R - 1;          // period^(best_R - 1)
            while (rem > 0) {
                if (rem & 1) {
                    ComplexMatrix n11, n12, n21, n22;
                    redheffer_cascade(C11, C12, C21, C22, B11, B12, B21, B22,
                                      n11, n12, n21, n22);
                    C11 = std::move(n11); C12 = std::move(n12);
                    C21 = std::move(n21); C22 = std::move(n22);
                }
                rem >>= 1;
                if (rem > 0) {
                    ComplexMatrix b11, b12, b21, b22;
                    redheffer_cascade(B11, B12, B21, B22, B11, B12, B21, B22,
                                      b11, b12, b21, b22);
                    B11 = std::move(b11); B12 = std::move(b12);
                    B21 = std::move(b21); B22 = std::move(b22);
                }
            }
            // Last period + boundary: S(s+(R-1)L, s+R·L).
            ComplexMatrix T11c, T12c, T21c, T22c, C11o, C12o, C21o, C22o;
            build_seq(best_s + (best_R - 1) * best_L,
                      best_s + best_R * best_L, T11c, T12c, T21c, T22c);
            redheffer_cascade(C11, C12, C21, C22, T11c, T12c, T21c, T22c,
                              C11o, C12o, C21o, C22o);

            // Assemble: prefix+head [indi, best_s) → core → tail.
            ComplexMatrix Scur11 = ComplexMatrix::Identity(n2, n2);
            ComplexMatrix Scur22 = ComplexMatrix::Identity(n2, n2);
            ComplexMatrix Scur12 = ComplexMatrix::Zero(n2, n2);
            ComplexMatrix Scur21 = ComplexMatrix::Zero(n2, n2);
            if (best_s > indi) {
                ComplexMatrix H11, H12, H21, H22;
                build_seq(indi, best_s, H11, H12, H21, H22);
                Scur11 = std::move(H11); Scur12 = std::move(H12);
                Scur21 = std::move(H21); Scur22 = std::move(H22);
            }
            {
                ComplexMatrix n11, n12, n21, n22;
                redheffer_cascade(Scur11, Scur12, Scur21, Scur22,
                                  C11o, C12o, C21o, C22o, n11, n12, n21, n22);
                Scur11 = std::move(n11); Scur12 = std::move(n12);
                Scur21 = std::move(n21); Scur22 = std::move(n22);
            }
            if (best_s + best_R * best_L < indj) {
                ComplexMatrix T11b, T12b, T21b, T22b, n11, n12, n21, n22;
                build_seq(best_s + best_R * best_L, indj, T11b, T12b, T21b, T22b);
                redheffer_cascade(Scur11, Scur12, Scur21, Scur22,
                                  T11b, T12b, T21b, T22b, n11, n12, n21, n22);
                Scur11 = std::move(n11); Scur12 = std::move(n12);
                Scur21 = std::move(n21); Scur22 = std::move(n22);
            }
            S11 = std::move(Scur11); S12 = std::move(Scur12);
            S21 = std::move(Scur21); S22 = std::move(Scur22);
            periodic_done = true;
        }
    }

    if (!periodic_done) {
        for (int l = indi; l < indj; ++l)
            step(l, S11, S12, S21, S22);
    }
    if (std::getenv("CPPRCWA_TIMING")) {
        using ms = std::chrono::duration<double, std::milli>;
        std::fprintf(stderr, "[S] general total %.1f ms%s\n",
                     ms(std::chrono::steady_clock::now()-g0).count(),
                     periodic_done ? " (periodic)" : "");
    }
    smatrix_cache_ = {indi, indj, S11, S12, S21, S22};
}

void RCWA::SolveExterior(const ComplexVector& a0, const ComplexVector& bN,
                         ComplexVector& aN, ComplexVector& b0) {
    int NL = Layer_N();
    ComplexMatrix S11, S12, S21, S22;
    GetSMatrix(0, NL - 1, S11, S12, S21, S22);
    // [aN; b0] = S * [a0; bN]
    aN = S11 * a0 + S12 * bN;
    b0 = S21 * a0 + S22 * bN;
}

void RCWA::SolveInterior(int which_layer, const ComplexVector& a0, const ComplexVector& bN,
                         ComplexVector& ai, ComplexVector& bi) {
    ComplexMatrix L11, L12, L21, L22;
    GetSMatrix(0, which_layer, L11, L12, L21, L22);
    ComplexMatrix R11, R12, R21, R22;
    GetSMatrix(which_layer, Layer_N() - 1, R11, R12, R21, R22);
    // ai = (I - L12*R21)^-1 (L11*a0 + L12*R22*bN)
    int n2 = 2 * nG_;
    ComplexMatrix M = ComplexMatrix::Identity(n2, n2) - L12 * R21;
    ComplexVector rhs = L11 * a0 + L12 * R22 * bN;
    ai = internal::zinverse(M) * rhs;
    bi = R21 * ai + R22 * bN;
}

void RCWA::TranslateAmplitudes(const ComplexVector& q, double thickness, double dz,
                               const ComplexVector& ai, const ComplexVector& bi,
                               ComplexVector& aim, ComplexVector& bim) {
    aim.resizeLike(ai);
    aim = ai.array() * (complex(0,1) * q * dz).array().exp();
    bim.resizeLike(bi);
    bim = bi.array() * (complex(0,1) * q * (thickness - dz)).array().exp();
}

// ── Public solves (Phase 5/6/7 — partial) ───────────────────────────────────

// Helper: compute (forward, backward) Poynting flux scalars for a given
// layer's (ai, bi) amplitudes. Matches grcwa.rcwa.GetZPoyntingFlux byorder=0.
static std::pair<double, double>
poynting_flux(complex omega, const ComplexMatrix& kp, const ComplexMatrix& phi,
              const ComplexVector& q, const ComplexVector& ai, const ComplexVector& bi) {
    int n2 = ai.size();
    int n  = n2 / 2;
    // A = kp · phi · diag(1/(omega·q))
    ComplexVector inv_omega_q = (complex(1,0) / (omega * q.array())).matrix();
    ComplexMatrix A = kp * phi * inv_omega_q.asDiagonal();
    ComplexVector pa = phi * ai;
    ComplexVector pb = phi * bi;
    ComplexVector Aa = A * ai;
    ComplexVector Ab = A * bi;
    // diff = 0.5 * (conj(pb)*Aa - conj(Ab)*pa)
    ComplexVector diff = 0.5 * (pb.conjugate().cwiseProduct(Aa)
                                - Ab.conjugate().cwiseProduct(pa));
    ComplexVector fwd_xy = Aa.conjugate().cwiseProduct(pa).real() + diff;
    ComplexVector bwd_xy = -Ab.conjugate().cwiseProduct(pb).real() + diff.conjugate();
    // forward = sum(fwd_xy[:n] + fwd_xy[n:])
    complex fwd = (fwd_xy.head(n).sum() + fwd_xy.tail(n).sum());
    complex bwd = (bwd_xy.head(n).sum() + bwd_xy.tail(n).sum());
    return {fwd.real(), bwd.real()};
}

RTResult RCWA::RT_Solve(bool normalize, bool byorder) {
    int NL = Layer_N();
    ComplexVector aN, b0;
    SolveExterior(a0_, bN_, aN, b0);

    // Layer 0: incident (a0) and reflected (b0)
    auto [fi, bi] = poynting_flux(omega_, kp(0), ph(0), q(0), a0_, b0);
    // Last layer: transmitted (aN) and backward-excited (bN)
    auto [fe, be] = poynting_flux(omega_, kp(NL-1), ph(NL-1), q(NL-1), aN, bN_);

    RTResult result;
    if (direction_ == Direction::Forward) {
        result.R = -bi;
        result.T =  fe;
    } else {
        result.R =  fe;
        result.T = -bi;
    }
    if (normalize) {
        result.R *= normalization_;
        result.T *= normalization_;
    }
    (void)byorder;
    return result;
}

std::pair<ComplexVector, ComplexVector>
RCWA::GetAmplitudes_noTranslate(int which_layer) {
    ComplexVector ai, bi;
    SolveInterior(which_layer, a0_, bN_, ai, bi);
    return {ai, bi};
}

std::pair<ComplexVector, ComplexVector>
RCWA::GetAmplitudes(int which_layer, double z_offset) {
    auto [ai, bi] = GetAmplitudes_noTranslate(which_layer);
    ComplexVector aim, bim;
    TranslateAmplitudes(q(which_layer), thickness_[which_layer], z_offset,
                        ai, bi, aim, bim);
    return {aim, bim};
}

FieldFourier
RCWA::field_from_amplitudes(int which_layer,
                            const ComplexVector& ai,
                            const ComplexVector& bi) const {
    // rcwa.py Solve_FieldFourier body (282-319)
    const ComplexVector& qloc = q(which_layer);
    bool is_uniform = (layer_types_[which_layer] == LayerType::Uniform);

    // hx, hy in Fourier space
    ComplexVector fhxy = ph(which_layer) * (ai + bi);
    ComplexVector fhx = fhxy.head(nG_);
    ComplexVector fhy = fhxy.tail(nG_);

    // ex, ey in Fourier space (fey = -fexy[:nG], fex = fexy[nG:])
    ComplexVector tmp1 = (ai - bi).cwiseQuotient((omega_ * qloc.array()).matrix());
    ComplexVector tmp2 = ph(which_layer) * tmp1;
    ComplexVector fexy = kp(which_layer) * tmp2;    ComplexVector fey = -fexy.head(nG_);
    ComplexVector fex = fexy.tail(nG_);

    // hz
    ComplexVector fhz = (kx_.cwiseProduct(fey) - ky_.cwiseProduct(fex)) / omega_;

    // ez
    ComplexVector fez = (ky_.cwiseProduct(fhx) - kx_.cwiseProduct(fhy)) / omega_;
    if (is_uniform) {
        fez /= uniform_eps_[material_idx_[which_layer]];
    } else {
        fez = patterned_epinv_[material_idx_[which_layer]] * fez;
    }

    FieldFourier f;
    f.ex = fex; f.ey = fey; f.ez = fez;
    f.hx = fhx; f.hy = fhy; f.hz = fhz;
    return f;
}

std::vector<FieldFourier>
RCWA::Solve_FieldFourier(int which_layer, const std::vector<double>& z_offsets) {
    // rcwa.py:282-319
    auto [ai0, bi0] = GetAmplitudes_noTranslate(which_layer);
    const ComplexVector& qloc = q(which_layer);
    double thickness = thickness_[which_layer];

    std::vector<FieldFourier> out;
    out.reserve(z_offsets.size());
    for (double zoff : z_offsets) {
        ComplexVector aim, bim;
        TranslateAmplitudes(qloc, thickness, zoff, ai0, bi0, aim, bim);
        out.push_back(field_from_amplitudes(which_layer, aim, bim));
    }
    return out;
}

std::vector<FieldFourier>
RCWA::Solve_FieldFourier(int which_layer, double z_offset) {
    return Solve_FieldFourier(which_layer, std::vector<double>{z_offset});
}

std::vector<FieldGrid>
RCWA::Solve_FieldOnGrid(int which_layer, const std::vector<double>& z_offsets,
                        std::optional<std::array<int,2>> Nxy) {
    // rcwa.py:321-348
    int Nx, Ny;
    if (Nxy) {
        Nx = (*Nxy)[0];
        Ny = (*Nxy)[1];
    } else {
        Nx = grid_Nxy_[grid_idx_[which_layer]].first;
        Ny = grid_Nxy_[grid_idx_[which_layer]].second;
    }
    double dN = 1.0 / ((double)Nx * Ny);
    auto fehl = Solve_FieldFourier(which_layer, z_offsets);

    std::vector<FieldGrid> out;
    out.reserve(fehl.size());
    for (const auto& feh : fehl) {
        FieldGrid g;
        g.ex = get_ifft(dN, Nx, Ny, feh.ex, G_);
        g.ey = get_ifft(dN, Nx, Ny, feh.ey, G_);
        g.ez = get_ifft(dN, Nx, Ny, feh.ez, G_);
        g.hx = get_ifft(dN, Nx, Ny, feh.hx, G_);
        g.hy = get_ifft(dN, Nx, Ny, feh.hy, G_);
        g.hz = get_ifft(dN, Nx, Ny, feh.hz, G_);
        out.push_back(std::move(g));
    }
    return out;
}

std::vector<FieldGrid>
RCWA::Solve_FieldOnGrid(int which_layer, double z_offset,
                        std::optional<std::array<int,2>> Nxy) {
    return Solve_FieldOnGrid(which_layer, std::vector<double>{z_offset}, Nxy);
}

FieldFourier RCWA::ForwardPropagatedFieldFourier(int which_layer, double z_offset) {
    // Field from the forward amplitudes ai alone (bi = 0), at depth z_offset.
    auto [ai, bi] = GetAmplitudes_noTranslate(which_layer);
    ComplexVector aim, bim;
    TranslateAmplitudes(q(which_layer), thickness_[which_layer], z_offset,
                        ai, bi, aim, bim);
    return field_from_amplitudes(which_layer, aim, ComplexVector::Zero(2 * nG_));
}

FieldFourier RCWA::BackwardPropagatedFieldFourier(int which_layer, double z_offset) {
    // Field from the backward amplitudes bi alone (ai = 0), at depth z_offset.
    // Layer 0 at z=0 → the reflected field in air.
    auto [ai, bi] = GetAmplitudes_noTranslate(which_layer);
    ComplexVector aim, bim;
    TranslateAmplitudes(q(which_layer), thickness_[which_layer], z_offset,
                        ai, bi, aim, bim);
    return field_from_amplitudes(which_layer, ComplexVector::Zero(2 * nG_), bim);
}

GridMatrix RCWA::Return_eps(int which_layer, int Nx, int Ny, const std::string& component) {
    // rcwa.py:192-216
    if (layer_types_[which_layer] == LayerType::Uniform) {
        complex ep = uniform_eps_[material_idx_[which_layer]];
        return GridMatrix::Constant(Nx, Ny, ep);
    }
    // patterned layer
    const ComplexMatrix& eps2 = patterned_ep2_[material_idx_[which_layer]];
    ComplexVector epk_row;
    if (component == "zz") {
        ComplexMatrix epk = internal::zinverse(patterned_epinv_[material_idx_[which_layer]]);
        epk_row = epk.row(0);
    } else if (component == "xx") {
        epk_row = eps2.block(0, 0, nG_, nG_).row(0);
    } else if (component == "xy") {
        epk_row = eps2.block(0, nG_, nG_, nG_).row(0);
    } else if (component == "yx") {
        epk_row = eps2.block(nG_, 0, nG_, nG_).row(0);
    } else if (component == "yy") {
        epk_row = eps2.block(nG_, nG_, nG_, nG_).row(0);
    } else {
        throw error::ConfigError("Return_eps: unknown component '" + component + "'");
    }
    double dN = 1.0 / ((double)Nx * Ny);
    return get_ifft(dN, Nx, Ny, epk_row, G_);
}

// ── Post-processing (Phase 7) ───────────────────────────────────────────────

// Matrix_zintegral (rcwa.py:607-639): Mt = [[Maa, Mab], [Mab, Maa]] where the
// blocks Maa, Mab are each (2nG×2nG). Returned as two blocks so the full
// 4nG×4nG Mt is never materialized (PLAN.md §10.4). The diagonal `shift`
// stability term is applied only to qij = qj - conj(qi), not qij2 (rcwa.py:628).
namespace {
struct ZIntegralBlocks { ComplexMatrix Maa, Mab; };
ZIntegralBlocks matrix_zintegral_blocks(const ComplexVector& q, double thickness,
                                        double shift = 1e-12) {
    int n2 = q.size();
    ComplexMatrix qi = q.replicate(1, n2);              // qi[i,j] = q[i]
    ComplexMatrix qj = q.transpose().replicate(n2, 1);  // qj[i,j] = q[j]
    complex I(0,1);

    ComplexMatrix qij = qj - qi.conjugate();
    qij.diagonal().array() += shift;
    ComplexMatrix Maa = ((qij * thickness * I).array().exp() - 1.0).matrix()
                        .cwiseQuotient(I * qij);

    ComplexMatrix qij2 = qj + qi.conjugate();
    ComplexMatrix Mab = (((qj * thickness * I).array().exp()
                        - (-qi.conjugate() * thickness * I).array().exp()).matrix())
                        .cwiseQuotient(I * qij2);

    return {std::move(Maa), std::move(Mab)};
}
} // namespace

complex RCWA::Volume_integral(int which_layer,
                              const ComplexMatrix& Mx,
                              const ComplexMatrix& My,
                              const ComplexMatrix& Mz,
                              bool normalize) {
    // rcwa.py:350-395, computed block-wise (PLAN.md §10.4).
    //
    // The original computes  val = trace(abM · F†·Mtotal·F)  with
    //   abM    = outer(ab, conj(ab)) ⊙ Mt           (4nG×4nG, ~1 GiB @ nG=1000)
    //   F      = [[Faxy, -Faxy], [Faz, Faz]]        (3nG×4nG)
    //   Mtotal = block_diag(Mx, My, Mz)             (3nG×3nG)
    //   Mt     = [[Maa, Mab], [Mab, Maa]]           (4nG×4nG)
    // The block structure of F and Mt collapses the trace onto 2nG×2nG matrices:
    //   Mxy = block_diag(Mx, My)
    //   A   = Faxy†·Mxy·Faxy + Faz†·Mz·Faz
    //   B   = -Faxy†·Mxy·Faxy + Faz†·Mz·Faz     (= A with the first term negated)
    //   T   = F†·Mtotal·F = [[A, B], [B, A]]
    // trace(abM·T) = Σ_{P,Q} a_Pᵀ (Mt_PQ ∘ T_QPᵀ) conj(a_Q)  (a_L=ai, a_R=bi):
    // val = aiᵀ(Maa∘Aᵀ)conj(ai) + biᵀ(Maa∘Aᵀ)conj(bi)
    //     + aiᵀ(Mab∘Bᵀ)conj(bi) + biᵀ(Mab∘Bᵀ)conj(ai)
    const ComplexMatrix& kpm = kp(which_layer);
    const ComplexVector&  qm  = q(which_layer);
    const ComplexMatrix&  phim = ph(which_layer);
    bool is_uniform = (layer_types_[which_layer] == LayerType::Uniform);

    ComplexMatrix epinv_mat;
    if (is_uniform) {
        complex epinv = complex(1.0, 0.0) / uniform_eps_[material_idx_[which_layer]];
        epinv_mat = ComplexMatrix::Identity(nG_, nG_) * epinv;
    } else {
        epinv_mat = patterned_epinv_[material_idx_[which_layer]];
    }

    ComplexVector ai, bi;
    SolveInterior(which_layer, a0_, bN_, ai, bi);

    const int n2 = 2 * nG_;

    // Faxy (2nG×2nG) and Faz (nG×2nG) — F = [[Faxy, -Faxy], [Faz, Faz]]
    ComplexMatrix Faxy = kpm * phim * (complex(1,0) / (omega_ * qm.array())).matrix().asDiagonal();
    ComplexMatrix Faz = ComplexMatrix::Zero(nG_, 2 * nG_);
    {
        ComplexMatrix Faz1 = (complex(1,0)/omega_) * epinv_mat * ky_.asDiagonal();
        ComplexMatrix Faz2 = -(complex(1,0)/omega_) * epinv_mat * kx_.asDiagonal();
        Faz.leftCols(nG_)  = Faz1;
        Faz.rightCols(nG_) = Faz2;
        Faz = Faz * phim;
    }

    // A = C + D, B = D - C with C = Faxy†·Mxy·Faxy, D = Faz†·Mz·Faz
    ComplexMatrix A, B;
    {
        ComplexMatrix Mxy = ComplexMatrix::Zero(n2, n2);
        Mxy.topLeftCorner(nG_, nG_)     = Mx;
        Mxy.bottomRightCorner(nG_, nG_) = My;
        ComplexMatrix C = Faxy.adjoint() * (Mxy * Faxy);
        ComplexMatrix D = Faz.adjoint() * (Mz * Faz);
        A = C + D;
        B = D - C;
    }

    // W_A = Maa∘Aᵀ, W_B = Mab∘Bᵀ (2nG×2nG each) — Mt blocks from z-integral.
    ComplexMatrix W_A, W_B;
    {
        auto [Maa, Mab] = matrix_zintegral_blocks(qm, thickness_[which_layer]);
        W_A = Maa.cwiseProduct(A.transpose());
        W_B = Mab.cwiseProduct(B.transpose());
    }

    // Scalar: f(x,y,W) = xᵀ·(W·conj(y)) summed element-wise (no conjugation on x).
    ComplexVector cai = ai.conjugate();
    ComplexVector cbi = bi.conjugate();
    complex val = (ai.array() * (W_A * cai).array()).sum()
                + (bi.array() * (W_A * cbi).array()).sum()
                + (ai.array() * (W_B * cbi).array()).sum()
                + (bi.array() * (W_B * cai).array()).sum();
    if (normalize) val *= normalization_;
    return val;
}

std::array<double, 3> RCWA::Solve_ZStressTensorIntegral(int which_layer) {
    // rcwa.py:397-435
    auto ff = Solve_FieldFourier(which_layer, 0.0);
    const ComplexVector& ex = ff[0].ex;
    const ComplexVector& ey = ff[0].ey;
    const ComplexVector& ez = ff[0].ez;
    const ComplexVector& hx = ff[0].hx;
    const ComplexVector& hy = ff[0].hy;
    const ComplexVector& hz = ff[0].hz;

    // Dz = (ky*hx - kx*hy)/omega
    ComplexVector dz = (ky_.cwiseProduct(hx) - kx_.cwiseProduct(hy)) / omega_;

    ComplexVector dx, dy;
    if (layer_types_[which_layer] == LayerType::Uniform) {
        complex ep = uniform_eps_[material_idx_[which_layer]];
        dx = ex * ep;
        dy = ey * ep;
    } else {
        // exy = hstack(-ey, ex); dxy = eps2 * exy; dx = dxy[nG:]; dy = -dxy[:nG]
        ComplexVector exy(2 * nG_);
        exy.head(nG_) = -ey;
        exy.tail(nG_) =  ex;
        ComplexVector dxy = patterned_ep2_[material_idx_[which_layer]] * exy;
        dx = dxy.tail(nG_);
        dy = -dxy.head(nG_);
    }

    complex Tx = (ex.cwiseProduct(dz.conjugate()) + hx.cwiseProduct(hz.conjugate())).sum();
    complex Ty = (ey.cwiseProduct(dz.conjugate()) + hy.cwiseProduct(hz.conjugate())).sum();
    complex Tz = 0.5 * (ez.cwiseProduct(dz.conjugate())
                        + hz.cwiseProduct(hz.conjugate())
                        - ey.cwiseProduct(dy.conjugate())
                        - ex.cwiseProduct(dx.conjugate())
                        - hx.cwiseAbs2() - hy.cwiseAbs2()).sum();
    return {Tx.real(), Ty.real(), Tz.real()};
}

} // namespace cpprcwa
