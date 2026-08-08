#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>
#include <optional>
#include <cpprcwa/types.h>

namespace cpprcwa {

class RCWA {
public:
    explicit RCWA(const RCWAConfig& config);
    ~RCWA();

    RCWA(const RCWA&) = delete;
    RCWA& operator=(const RCWA&) = delete;

    // ── Layer specification (call in order, BEFORE Init_Setup) ──
    void Add_LayerUniform(double thickness, complex epsilon);
    void Add_LayerGrid(double thickness, int Nx, int Ny);

    // ── Initialization (must call before any solve) ──
    // Pscale: lattice period scale (multiplies period). Gmethod: 0=circ, 1=para.
    void Init_Setup(double Pscale = 1.0, int Gmethod = 0);

    // ── Excitation ──
    void MakeExcitationPlanewave(const PlaneWaveExcitation& exc);

    // ── Material for patterned layers ──
    // ep_all: flat concatenation of all patterned layer grids in order.
    void GridLayer_geteps(const std::vector<complex>& ep_all_isotropic);
    void GridLayer_geteps(const std::vector<std::vector<complex>>& ep_all_anisotropic);

    // ── Main solve ──
    RTResult RT_Solve(bool normalize = false, bool byorder = false);

    // ── Amplitudes ──
    std::pair<ComplexVector, ComplexVector>
    GetAmplitudes(int which_layer, double z_offset);

    std::pair<ComplexVector, ComplexVector>
    GetAmplitudes_noTranslate(int which_layer);

    // ── Fields ──
    std::vector<FieldFourier>
    Solve_FieldFourier(int which_layer, const std::vector<double>& z_offsets);
    std::vector<FieldFourier>
    Solve_FieldFourier(int which_layer, double z_offset);

    std::vector<FieldGrid>
    Solve_FieldOnGrid(int which_layer, const std::vector<double>& z_offsets,
                      std::optional<std::array<int,2>> Nxy = std::nullopt);
    std::vector<FieldGrid>
    Solve_FieldOnGrid(int which_layer, double z_offset,
                      std::optional<std::array<int,2>> Nxy = std::nullopt);

    GridMatrix Return_eps(int which_layer, int Nx, int Ny,
                          const std::string& component);

    // Volume integral of M·|E|² for absorbed power.
    // Mx, My, Mz are nG×nG convolution matrices (e.g. real part of epinv).
    complex Volume_integral(int which_layer,
                            const ComplexMatrix& Mx,
                            const ComplexMatrix& My,
                            const ComplexMatrix& Mz,
                            bool normalize = false);

    // Maxwell stress tensor integral → (2Fx, 2Fy, 2Fz)
    std::array<double, 3> Solve_ZStressTensorIntegral(int which_layer);

    // Field contribution from the forward-propagating (ai) amplitudes alone,
    // evaluated in `which_layer` at depth `z_offset` (0 = front interface).
    FieldFourier ForwardPropagatedFieldFourier(int which_layer, double z_offset);

    // Field contribution from the backward-propagating (bi) amplitudes alone,
    // evaluated in `which_layer` at depth `z_offset` (0 = front interface).
    // For layer 0 at z=0 this is the reflected field in air.
    FieldFourier BackwardPropagatedFieldFourier(int which_layer, double z_offset);

    // ── Accessors ──
    int nG() const { return nG_; }
    int Layer_N() const { return static_cast<int>(layer_types_.size()); }
    const IntMatrix& G() const { return G_; }
    const ComplexVector& kx() const { return kx_; }
    const ComplexVector& ky() const { return ky_; }
    const std::vector<double>& thickness_list() const { return thickness_; }
    const std::vector<ComplexVector>& q_list() const { return q_list_; }
    const std::vector<ComplexMatrix>& phi_list() const { return phi_list_; }
    const std::vector<ComplexMatrix>& kp_list() const { return kp_list_; }
    double normalization() const { return normalization_; }
    const std::vector<ComplexMatrix>& patterned_epinv_list() const { return patterned_epinv_; }
    const std::vector<ComplexMatrix>& patterned_ep2_list() const { return patterned_ep2_; }

private:
    // ── Configuration & problem ──
    int nG_req_;                      // requested nG
    int nG_;                           // actual nG from getG
    complex freq_, omega_;            // omega = 2*pi*freq (both complex)
    Eigen::Vector2d L1_, L2_;
    double theta_, phi_;
    double normalization_ = 1.0;
    Direction direction_ = Direction::Forward;

    // ── Init_Setup params ──
    double Pscale_ = 1.0;
    int Gmethod_ = 0;

    // ── Layer storage ──
    std::vector<LayerType> layer_types_;
    std::vector<double>    thickness_;
    std::vector<complex>   uniform_eps_;
    std::vector<std::pair<int,int>> grid_Nxy_;  // (Nx,Ny) per grid layer
    std::vector<int> material_idx_;   // per layer: uniform_idx (uniform) or patterned_idx (grid/fourier)
    std::vector<int> grid_idx_;       // per layer: grid layer counter (only for Grid layers)
    int patterned_count_ = 0;         // total patterned layers added

    // ── Computed layer data (length = Layer_N) ──
    std::vector<ComplexMatrix> kp_list_;
    std::vector<ComplexVector> q_list_;
    std::vector<ComplexMatrix> phi_list_;

    // ── Per-patterned-layer (indexed by patterned_count, NOT Layer_N) ──
    std::vector<ComplexMatrix> patterned_epinv_;
    std::vector<ComplexMatrix> patterned_ep2_;

    // ── Reciprocal lattice & k-vectors ──
    IntMatrix G_;
    Eigen::Vector2d Lk1_, Lk2_;
    ComplexVector kx_, ky_;

    // ── Excitation ──
    ComplexVector a0_, bN_;

    // ── Internal helpers ──
    void MakeKPMatrix_uniform(complex omega, const ComplexVector& kx, const ComplexVector& ky,
                              complex eps, ComplexMatrix& kp);
    void MakeKPMatrix_patterned(complex omega, const ComplexVector& kx, const ComplexVector& ky,
                                const ComplexMatrix& epinv, const ComplexMatrix& ep2,
                                ComplexMatrix& kp);
    void SolveLayerEigensystem_uniform(complex omega, const ComplexVector& kx, const ComplexVector& ky,
                                       complex eps, ComplexVector& q, ComplexMatrix& phi);
    void SolveLayerEigensystem_patterned(complex omega, const ComplexVector& kx, const ComplexVector& ky,
                                         const ComplexMatrix& kp, const ComplexMatrix& ep2,
                                         ComplexVector& q, ComplexMatrix& phi);

    void GetSMatrix(int indi, int indj,
                    ComplexMatrix& S11, ComplexMatrix& S12,
                    ComplexMatrix& S21, ComplexMatrix& S22);

    void SolveExterior(const ComplexVector& a0, const ComplexVector& bN,
                       ComplexVector& aN, ComplexVector& b0);
    void SolveInterior(int which_layer, const ComplexVector& a0, const ComplexVector& bN,
                       ComplexVector& ai, ComplexVector& bi);
    static void TranslateAmplitudes(const ComplexVector& q, double thickness, double dz,
                                    const ComplexVector& ai, const ComplexVector& bi,
                                    ComplexVector& aim, ComplexVector& bim);

    // Field Fourier coefficients in `which_layer` from arbitrary (untranslated)
    // amplitudes ai, bi (rcwa.py Solve_FieldFourier body).
    FieldFourier field_from_amplitudes(int which_layer,
                                       const ComplexVector& ai,
                                       const ComplexVector& bi) const;
};

} // namespace cpprcwa
