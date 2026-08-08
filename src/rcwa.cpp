#include <cpprcwa/rcwa.h>
#include <cpprcwa/kbloch.h>
#include <cpprcwa/fft_funs.h>
#include "internal/branch_cut.h"
#include "internal/lapack_wrappers.h"
#include "internal/utils.h"
#include <cpprcwa/errors.h>
#include <cmath>

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
      quasi1d_(config.quasi1d) {}

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

    // Per-layer kp/q/phi for uniform layers. Patterned layers are filled
    // in GridLayer_geteps().
    int nLayers = Layer_N();
    kp_list_.assign(nLayers, ComplexMatrix());
    q_list_.assign(nLayers, ComplexVector());
    phi_list_.assign(nLayers, ComplexMatrix());

    normalization_ = std::sqrt(eps0.real()) / std::cos(theta_);   // rcwa.py:103

    // Uniform layers share kp/q/phi across identical eps values (they depend
    // only on eps + the global kx/ky/omega). Periodic stacks (Mo/Si EUV
    // multilayers) repeat a few eps values dozens of times — caching turns 80
    // eigen/matrix setups into ~5.
    struct EpsKey {
        complex eps;
        bool operator<(const EpsKey& o) const {
            if (eps.real() != o.eps.real()) return eps.real() < o.eps.real();
            return eps.imag() < o.eps.imag();
        }
    };
    std::map<EpsKey, std::pair<ComplexMatrix, ComplexVector>> uni_cache;

    int uniform_idx = 0;
    for (int li = 0; li < nLayers; ++li) {
        if (layer_types_[li] != LayerType::Uniform) continue;
        complex eps = uniform_eps_[uniform_idx];
        EpsKey key{eps};
        auto it = uni_cache.find(key);
        if (it == uni_cache.end()) {
            MakeKPMatrix_uniform(omega_, kx_, ky_, eps, kp_list_[li]);
            SolveLayerEigensystem_uniform(omega_, kx_, ky_, eps,
                                          q_list_[li], phi_list_[li]);
            it = uni_cache.emplace(key, std::make_pair(kp_list_[li], q_list_[li])).first;
        } else {
            kp_list_[li] = it->second.first;
            q_list_[li]  = it->second.second;
            phi_list_[li] = ComplexMatrix::Identity(2 * nG_, 2 * nG_);
        }
        ++uniform_idx;
    }
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
    for (int li = 0; li < Layer_N(); ++li) {
        if (layer_types_[li] != LayerType::Grid) continue;
        auto [Nx, Ny] = grid_Nxy_[grid_idx];
        double dN = 1.0 / ((double)Nx * Ny);
        std::vector<complex> slice(ep_all_isotropic.begin() + offset,
                                   ep_all_isotropic.begin() + offset + (size_t)Nx * Ny);
        auto result = Epsilon_fft(dN, slice, Nx, Ny, G_);
        patterned_epinv_.push_back(result.epsinv);
        patterned_ep2_.push_back(result.eps2);
        // Fill kp/q/phi for this layer
        MakeKPMatrix_patterned(omega_, kx_, ky_, result.epsinv, result.eps2, kp_list_[li]);
        SolveLayerEigensystem_patterned(omega_, kx_, ky_, kp_list_[li], result.eps2,
                                        q_list_[li], phi_list_[li]);
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
    ComplexMatrix kkT = k_mat * k_mat.transpose();
    ComplexMatrix M = ep2 * kp - kkT;
    ComplexVector evals(n2);
    phi.resize(n2, n2);
    internal::zgeev(n2, M.data(), (int)M.outerStride(), evals,
                    phi.data(), (int)phi.outerStride());
    // q = sqrt(evals) with branch cut
    for (int i = 0; i < n2; ++i) evals(i) = std::sqrt(evals(i));
    q = apply_branch_cut(evals);
}

// ── S-matrix (Phase 5 — stub for now) ───────────────────────────────────────

void RCWA::GetSMatrix(int indi, int indj,
                      ComplexMatrix& S11, ComplexMatrix& S12,
                      ComplexMatrix& S21, ComplexMatrix& S22) {
    int n2 = 2 * nG_;
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
        ComplexMatrix d1 = (complex(0,1) * q_list_[l]   * thickness_[l]  ).array().exp().matrix().asDiagonal();
        ComplexMatrix d2 = (complex(0,1) * q_list_[lp1] * thickness_[lp1]).array().exp().matrix().asDiagonal();

        if (uniform_pair) {
            // phi = I for uniform layers → Q = I. Cache T11/T12 per (eps_l, eps_lp1).
            UniformPairKey key{uniform_eps_[material_idx_[l]],
                               uniform_eps_[material_idx_[lp1]]};
            auto it = uniform_pair_cache_.find(key);
            if (it == uniform_pair_cache_.end()) {
                ComplexMatrix kp_l_inv = internal::zinverse(kp_list_[l]);   // = inv(kp_l·phi_l)
                ComplexVector qinv = q_list_[lp1].cwiseInverse();
                ComplexMatrix P = q_list_[l].asDiagonal() * kp_l_inv
                                  * kp_list_[lp1] * qinv.asDiagonal();
                UniformPairCache c;
                c.T11 = 0.5 * (ComplexMatrix::Identity(n2, n2) + P);
                c.T12 = 0.5 * (ComplexMatrix::Identity(n2, n2) - P);
                it = uniform_pair_cache_.emplace(key, std::move(c)).first;
            }
            T11 = it->second.T11;
            T12 = it->second.T12;
        } else {
            ComplexMatrix Q = internal::zinverse(phi_list_[l]) * phi_list_[lp1];
            ComplexMatrix kpphi_l_inv = internal::zinverse(kp_list_[l] * phi_list_[l]);
            ComplexVector qinv = q_list_[lp1].cwiseInverse();
            ComplexMatrix P = q_list_[l].asDiagonal() * kpphi_l_inv
                              * kp_list_[lp1] * phi_list_[lp1] * qinv.asDiagonal();
            T11 = 0.5 * (Q + P);
            T12 = 0.5 * (Q - P);
        }

        // Redheffer star product (S-update, sequential, cannot cache).
        // Common subexpressions d1·S12 and S22·T12 are computed once; the
        // inv(P1m) products are replaced by one LU factorization + two
        // back-substitutions.
        ComplexMatrix d1S12 = d1 * s12;                      // O(n²) (d1 diagonal)
        ComplexMatrix P1m = T11 - d1S12 * T12;               // 1 full matmul
        std::vector<int> piv = internal::zgetrf_factor(n2, P1m.data(),
                                                       static_cast<int>(P1m.outerStride()));

        ComplexMatrix new_S11 = d1 * s11;                    // O(n²)
        internal::zgetrs_solve(n2, n2, P1m.data(), static_cast<int>(P1m.outerStride()),
                               piv.data(), new_S11.data(),
                               static_cast<int>(new_S11.outerStride()));

        ComplexMatrix P2 = d1S12 * T11 - T12;                // 1 full matmul
        ComplexMatrix new_S12 = P2 * d2;                     // O(n²)
        internal::zgetrs_solve(n2, n2, P1m.data(), static_cast<int>(P1m.outerStride()),
                               piv.data(), new_S12.data(),
                               static_cast<int>(new_S12.outerStride()));

        ComplexMatrix S22T12 = s22 * T12;                    // 1 full matmul
        ComplexMatrix new_S21 = s21 + S22T12 * new_S11;      // 1 full matmul
        ComplexMatrix new_S22 = s22 * T11 * d2 + S22T12 * new_S12;  // 2 full matmuls
        s11 = std::move(new_S11);
        s12 = std::move(new_S12);
        s21 = std::move(new_S21);
        s22 = std::move(new_S22);
    };

    // ── quasi-1D fast path ──
    // With the 1D harmonic set (j==0) every uniform-layer kp/q/phi is diagonal,
    // so the S-matrix of an all-uniform layer range is exactly diagonal. Split
    // the stack at the start of the trailing uniform suffix: compute the (grid
    // + interface) prefix with the general sequential loop, the uniform suffix
    // with a scalar (per-harmonic) recursion, then assemble with the
    // overlapping-cascade formula.
    if (quasi1d_) {
        int m = indj;
        while (m >= indi && layer_types_[m] == LayerType::Uniform) --m;
        ++m;   // layers [m, indj] are all uniform (if m <= indj)
        if (m <= indj) {
            // L = S(indi, m) — general prefix (may contain patterned layers).
            ComplexMatrix L11 = ComplexMatrix::Identity(n2, n2);
            ComplexMatrix L22 = ComplexMatrix::Identity(n2, n2);
            ComplexMatrix L12 = ComplexMatrix::Zero(n2, n2);
            ComplexMatrix L21 = ComplexMatrix::Zero(n2, n2);
            for (int l = indi; l < m; ++l) step(l, L11, L12, L21, L22);

            // R = S(m, indj) — all-uniform suffix, diagonal. Scalar recursion
            // over the 2nG harmonics.
            ComplexVector r11 = ComplexVector::Ones(n2);
            ComplexVector r22 = ComplexVector::Ones(n2);
            ComplexVector r12 = ComplexVector::Zero(n2);
            ComplexVector r21 = ComplexVector::Zero(n2);
            for (int l = m; l < indj; ++l) {
                int lp1 = l + 1;
                const ComplexVector& ql = q_list_[l];
                const ComplexVector& qb = q_list_[lp1];
                for (int h = 0; h < n2; ++h) {
                    complex d1 = std::exp(complex(0, 1) * ql(h) * thickness_[l]);
                    complex d2 = std::exp(complex(0, 1) * qb(h) * thickness_[lp1]);
                    complex kl = kp_list_[l](h, h), kb = kp_list_[lp1](h, h);
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
            ComplexMatrix R11 = r11.asDiagonal();
            ComplexMatrix R12 = r12.asDiagonal();
            ComplexMatrix R21 = r21.asDiagonal();
            ComplexMatrix R22 = r22.asDiagonal();

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
            return;
        }
    }

    // ── General sequential path ──
    S11 = ComplexMatrix::Identity(n2, n2);
    S22 = ComplexMatrix::Identity(n2, n2);
    S12 = ComplexMatrix::Zero(n2, n2);
    S21 = ComplexMatrix::Zero(n2, n2);
    for (int l = indi; l < indj; ++l)
        step(l, S11, S12, S21, S22);
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
    auto [fi, bi] = poynting_flux(omega_, kp_list_[0], phi_list_[0], q_list_[0], a0_, b0);
    // Last layer: transmitted (aN) and backward-excited (bN)
    auto [fe, be] = poynting_flux(omega_, kp_list_[NL-1], phi_list_[NL-1], q_list_[NL-1], aN, bN_);

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
    TranslateAmplitudes(q_list_[which_layer], thickness_[which_layer], z_offset,
                        ai, bi, aim, bim);
    return {aim, bim};
}

FieldFourier
RCWA::field_from_amplitudes(int which_layer,
                            const ComplexVector& ai,
                            const ComplexVector& bi) const {
    // rcwa.py Solve_FieldFourier body (282-319)
    const ComplexVector& q = q_list_[which_layer];
    bool is_uniform = (layer_types_[which_layer] == LayerType::Uniform);

    // hx, hy in Fourier space
    ComplexVector fhxy = phi_list_[which_layer] * (ai + bi);
    ComplexVector fhx = fhxy.head(nG_);
    ComplexVector fhy = fhxy.tail(nG_);

    // ex, ey in Fourier space (fey = -fexy[:nG], fex = fexy[nG:])
    ComplexVector tmp1 = (ai - bi).cwiseQuotient((omega_ * q.array()).matrix());
    ComplexVector tmp2 = phi_list_[which_layer] * tmp1;
    ComplexVector fexy = kp_list_[which_layer] * tmp2;
    ComplexVector fey = -fexy.head(nG_);
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
    const ComplexVector& q = q_list_[which_layer];
    double thickness = thickness_[which_layer];

    std::vector<FieldFourier> out;
    out.reserve(z_offsets.size());
    for (double zoff : z_offsets) {
        ComplexVector aim, bim;
        TranslateAmplitudes(q, thickness, zoff, ai0, bi0, aim, bim);
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
    TranslateAmplitudes(q_list_[which_layer], thickness_[which_layer], z_offset,
                        ai, bi, aim, bim);
    return field_from_amplitudes(which_layer, aim, ComplexVector::Zero(2 * nG_));
}

FieldFourier RCWA::BackwardPropagatedFieldFourier(int which_layer, double z_offset) {
    // Field from the backward amplitudes bi alone (ai = 0), at depth z_offset.
    // Layer 0 at z=0 → the reflected field in air.
    auto [ai, bi] = GetAmplitudes_noTranslate(which_layer);
    ComplexVector aim, bim;
    TranslateAmplitudes(q_list_[which_layer], thickness_[which_layer], z_offset,
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

// Matrix_zintegral (rcwa.py:607-639): generates 4nG×4nG matrix for the z-integral.
namespace {
ComplexMatrix Matrix_zintegral(const ComplexVector& q, double thickness, double shift = 1e-12) {
    int nG2 = q.size();
    ComplexMatrix qi = q.replicate(1, nG2);              // qi[i,j] = q[i]
    ComplexMatrix qj = q.transpose().replicate(nG2, 1);  // qj[i,j] = q[j]
    complex I(0,1);

    // qij = qj - conj(qi) + shift*I on the diagonal (stability term)
    ComplexMatrix qij = qj - qi.conjugate();
    qij.diagonal().array() += shift;
    ComplexMatrix Maa = ((qij * thickness * I).array().exp() - 1.0).matrix()
                        .cwiseQuotient(I * qij);

    ComplexMatrix qij2 = qj + qi.conjugate();
    ComplexMatrix Mab = (((qj * thickness * I).array().exp()
                        - (-qi.conjugate() * thickness * I).array().exp()).matrix())
                        .cwiseQuotient(I * qij2);

    ComplexMatrix Mt(2 * nG2, 2 * nG2);
    Mt.topLeftCorner(nG2, nG2)     = Maa;
    Mt.topRightCorner(nG2, nG2)    = Mab;
    Mt.bottomLeftCorner(nG2, nG2)  = Mab;
    Mt.bottomRightCorner(nG2, nG2) = Maa;
    return Mt;
}
} // namespace

complex RCWA::Volume_integral(int which_layer,
                              const ComplexMatrix& Mx,
                              const ComplexMatrix& My,
                              const ComplexMatrix& Mz,
                              bool normalize) {
    // rcwa.py:350-395
    const ComplexMatrix& kp  = kp_list_[which_layer];
    const ComplexVector&  q  = q_list_[which_layer];
    const ComplexMatrix&  phi = phi_list_[which_layer];
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

    // ab = hstack(ai, bi) (4nG); abMatrix = outer(ab, conj(ab)) (4nG×4nG)
    ComplexVector ab(4 * nG_);
    ab.head(2 * nG_) = ai;
    ab.tail(2 * nG_) = bi;
    ComplexMatrix abMatrix = ab * ab.conjugate().transpose();
    ComplexMatrix Mt = Matrix_zintegral(q, thickness_[which_layer]);
    ComplexMatrix abM = abMatrix.cwiseProduct(Mt);

    // F (3nG × 4nG)
    ComplexVector inv_omega_q = (complex(1,0) / (omega_ * q.array())).matrix();
    ComplexMatrix Faxy = kp * phi * inv_omega_q.asDiagonal();      // 2nG × 2nG
    ComplexMatrix Faz1 = (complex(1,0)/omega_) * epinv_mat * ky_.asDiagonal();
    ComplexMatrix Faz2 = -(complex(1,0)/omega_) * epinv_mat * kx_.asDiagonal();
    ComplexMatrix Faz = ComplexMatrix::Zero(nG_, 2 * nG_);
    Faz.leftCols(nG_)  = Faz1;
    Faz.rightCols(nG_) = Faz2;
    Faz = Faz * phi;                                               // nG × 2nG

    ComplexMatrix F(3 * nG_, 4 * nG_);
    F.topRows(2 * nG_).leftCols(2 * nG_)  =  Faxy;
    F.topRows(2 * nG_).rightCols(2 * nG_) = -Faxy;
    F.bottomRows(nG_).leftCols(2 * nG_)   = Faz;
    F.bottomRows(nG_).rightCols(2 * nG_)  = Faz;

    // Mtotal = block_diag(Mx, My, Mz) (3nG × 3nG)
    ComplexMatrix Mtotal = ComplexMatrix::Zero(3 * nG_, 3 * nG_);
    Mtotal.topLeftCorner(nG_, nG_)      = Mx;
    Mtotal.block(nG_, nG_, nG_, nG_)    = My;
    Mtotal.bottomRightCorner(nG_, nG_)  = Mz;

    // val = trace(abM @ (F^dagger Mtotal F))
    complex val = (abM * (F.adjoint() * Mtotal * F)).trace();
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
