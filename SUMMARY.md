# cpprcwa — Project Summary

A high-performance C++17 port of the **grcwa** Python library (Rigorous
Coupled-Wave Analysis for photonic structures), providing the forward RCWA
solver with validation against grcwa/S4 reference values.

This document describes the directory layout, every public/private function,
and the mathematical logic behind each module. It complements `PLAN.md`
(porting plan) and `README.md` (build/usage).

---

## 1. Directory Structure

```
cpprcwa/
├── PLAN.md                     # Porting plan (phases, risks, status §1.3)
├── README.md                   # Build + quick-start guide
├── CMakeLists.txt              # Root build (lib + tests + examples + install)
├── cmake/
│   └── FindFFTW3.cmake         # FFTW3 find-module (no upstream CMake config)
├── include/cpprcwa/            # Public headers (installed)
│   ├── cpprcwa.h               # Aggregator: include only this
│   ├── types.h                 # Aliases, RCWAConfig, layer/excitation/result structs
│   ├── errors.h                # Exception hierarchy
│   ├── kbloch.h                # Reciprocal-lattice math
│   ├── fft_funs.h              # FFTW plan cache + FFT convolution ops
│   └── rcwa.h                  # RCWA class (public API + private state/helpers)
├── src/                        # Implementations
│   ├── kbloch.cpp
│   ├── fft_funs.cpp
│   ├── rcwa.cpp
│   └── internal/
│       ├── branch_cut.{h,cpp}  # sqrt branch-cut helper
│       ├── lapack_wrappers.{h,cpp}
│       └── utils.{h,cpp}       # diag/eye/vstack/hstack helpers
├── tests/
│   ├── CMakeLists.txt          # Catch2 v3 (system or FetchContent)
│   ├── test_main.cpp
│   ├── test_kbloch.cpp
│   ├── test_fft_funs.cpp
│   ├── test_rcwa.cpp           # incl. S4 golden test, fields, post-processing
│   └── golden/                 # Python-generated reference outputs
├── examples/
│   ├── ex1_square_lattice.cpp  # Square lattice of holes (port of ex1.py)
│   ├── ex2_two_layers.cpp      # Two patterned layers, oblique incidence (ex2.py)
│   ├── ex4_hexagonal.cpp       # Hexagonal lattice, non-orthogonal coords (ex4.py)
│   └── ex_euv_multilayer.cpp   # EUV Mo/Si multilayer mirror (Ru cap) reflectivity
├── benchmarks/
│   └── bench_full_rt.cpp       # End-to-end RT_Solve timing
└── scripts/
    ├── generate_golden.py      # Run grcwa, save golden .txt files
    └── compare_results.py      # Diff C++ scalar vs golden
```

---

## 2. Common Types (`include/cpprcwa/types.h`)

```cpp
using complex       = std::complex<double>;
using ComplexVector = Eigen::VectorXcd;      // dynamic complex vector
using ComplexMatrix = Eigen::MatrixXcd;      // dynamic complex matrix (column-major)
using RealVector    = Eigen::VectorXd;
using IntMatrix     = Eigen::MatrixXi;       // G is (nG, 2)
using GridMatrix    = Eigen::Matrix<complex, Dynamic, Dynamic, RowMajor>;
```

**`GridMatrix` (row-major)** exists specifically for FFTW interop: FFTW
expects row-major `(Nx, Ny)` buffers. All real-space grids
(`FieldGrid`, `Solve_FieldOnGrid`, `Return_eps`) use it; all Fourier-space
objects use column-major `ComplexMatrix`. The index convention is
`buf[i*Ny + j]` ↔ `(i = x = G_x, j = y = G_y)`, matching NumPy `fft2`.

### Configuration
```cpp
struct RCWAConfig {
    int nG;                  // requested truncation (actual may be smaller)
    Eigen::Vector2d L1, L2;  // direct lattice vectors
    complex freq;            // COMPLEX — caller folds in Qabs loss (§6.1)
    double theta = 0, phi = 0;  // polar / azimuthal incidence angles
    int verbose  = 1;
};
```
`Qabs`, `Pscale`, `Gmethod` are **not** in the config: `Qabs` is folded into
`freq` by the caller; `Pscale`/`Gmethod` are `Init_Setup()` arguments
(matching the Python keyword args).

### Layers / excitation / results
```cpp
enum class LayerType : int { Uniform, Grid, Fourier };   // Fourier = stub (throws)
struct LayerUniform { double thickness; complex epsilon; };
struct LayerGrid    { double thickness; int Nx, Ny; };

enum class Direction : int { Forward, Backward };
struct PlaneWaveExcitation {
    double p_amp = 0, p_phase = 0, s_amp = 0, s_phase = 0;
    int order = 0;               // which G-index is the incident wave
    Direction direction = Direction::Forward;
};

struct FieldFourier { ComplexVector ex, ey, ez, hx, hy, hz; };   // len nG each
struct FieldGrid    { GridMatrix  ex, ey, ez, hx, hy, hz; };     // (Nx, Ny) each
struct RTResult {
    double R = 0, T = 0;
    ComplexVector R_per_order, T_per_order;   // populated if byorder=true
};
```

### Exceptions (`include/cpprcwa/errors.h`)
`error::Error` (base) → `ConfigError`, `NotImplementedError`,
`LapackError(routine, info)`, `SingularMatrixError(info, layer)`. LAPACK
failures throw typed exceptions instead of silently producing NaNs.

---

## 3. Module: `kbloch` — Reciprocal Lattice

File: `include/cpprcwa/kbloch.h`, `src/kbloch.cpp`. Pure math, no external
libraries.

### `Lattice_Reciprocate(L1, L2) → (Lk1, Lk2)`
```
d   = L1[0]*L2[1] - L1[1]*L2[0]          // 2D scalar cross product
Lk1 = [ L2[1]/d, -L2[0]/d ]
Lk2 = [-L1[1]/d,  L1[0]/d ]
```
The returned vectors omit the `2π` factor (it is applied later in
`Lattice_SetKs`).

### `Lattice_getG(nG, Lk1, Lk2, method=0) → (G, nG_out)`
Selects the integer `(i, j)` pairs defining G-vectors `G = i·Lk1 + j·Lk2`,
truncated to ≈`nG` points. Two methods:

**Circular (`method=0`)** — preserves rotational symmetry:
1. `u=|Lk1|, v=|Lk2|, uv=Lk1·Lk2, uxv=Lk1×Lk2`
2. Estimate a k-space disc: `circ_area = nG·|uxv|`,
   `circ_radius = √(circ_area/π) + u + v`
3. Compute extents `u_extent, v_extent` and generate a
   `(2·u_extent+1)×(2·v_extent+1)` grid of `(i,j)`.
4. Sort by `|G|² = (i·u)² + (j·v)² + 2·i·j·uv` (ties broken by `(i,j)` for
   determinism), take the first `nG` entries.
5. **Degeneracy trim** (mirrors `kbloch.py:119-124`): scan from the top with
   `tol = 1e-10·max(u²,v²)`, find the first index `i` where
   `|gl2[i] − gl2[i−1]| > tol` and keep entries `0..i−1`. This keeps partial
   degenerate shells from breaking symmetry — so `nG_out ≤ nG`.

**Parallelogramic (`method=1`)**:
1. `NGroot` = largest odd integer `≤ √nG`; `M = NGroot/2`.
2. Generate the `NGroot×NGroot` grid `-M..M`, sort by `|G|²`, keep the first
   `NGroot²`. Always returns exactly `NGroot²` points.

### `Lattice_SetKs(G, kx0, ky0, Lk1, Lk2, kx, ky)`
```
kx[i] = kx0 + 2π·(Lk1[0]·G(i,0) + Lk2[0]·G(i,1))
ky[i] = ky0 + 2π·(Lk1[1]·G(i,0) + Lk2[1]·G(i,1))
```
`kx0, ky0` are **complex** because `omega` is complex (`omega·sinθ·cosφ·√ε₀`).

---

## 4. Module: `fft_funs` — FFT Convolution Matrices

File: `include/cpprcwa/fft_funs.h`, `src/fft_funs.cpp`. Uses FFTW3.

### `FFTWPlanCache`
FFTW plans cost ~10–100 ms each, so one `(forward, backward)` plan pair is
cached per `(Nx, Ny)` in a mutex-protected `std::unordered_map`. Plans are
created with `FFTW_MEASURE`. A single `static` cache instance is shared by all
calls in the translation unit. Reusing the *same* plan from multiple threads
requires FFTW threads support (see `PLAN.md §6.11`).

### FFT convention / normalization
- Inputs are treated as row-major `(Nx, Ny)` via `GridMatrix` (see §2).
- `fftw_plan_dft_2d(Nx, Ny, ...)` with `FFTW_FORWARD` matches NumPy `fft2`
  convention: `sfft[k] = Σₙ a[n]·exp(−2πi·k·n/N)`.
- FFTW does **not** normalize; NumPy divides by `Nx·Ny` on the inverse. The
  `dN = 1/(Nx·Ny)` factor from the Python code is applied manually.

### `get_conv(dN, s_in_flat, Nx, Ny, G) → ComplexMatrix (nG×nG)`
Builds the Toeplitz-like convolution matrix:
1. FFT the flattened grid, scale by `dN`.
2. `C[i,j] = s_fft[(G[i,0]−G[j,0]) mod Nx, (G[i,1]−G[j,1]) mod Ny]`
   (negative differences wrapped with the double-`%` idiom to match NumPy).

### `get_fft(dN, s_in_flat, Nx, Ny, G) → ComplexVector (nG)`
`fft2(s_in)·dN`, then pick coefficients at `(G[i,0], G[i,1])` (wrapped).

### `get_ifft(dN, Nx, Ny, s_in, G) → GridMatrix (Nx×Ny)`
1. Place `s_in[i]` at `(G[i,0] mod Nx, G[i,1] mod Ny)` in a zeroed grid.
2. IFFT (FFTW `BACKWARD`, unnormalized sum).
3. Scale by `1/(Nx·Ny·dN)`. Because `dN = 1/(Nx·Ny)`, this factor is **1**:
   the result is exactly the raw inverse transform — matching NumPy's
   `ifft2(s0)/dN` (the `1/N` from `ifft2` cancels the `/dN`).

### `Epsilon_fft(dN, eps_grid, Nx, Ny, G) → EpsilonFftResult`
`EpsilonFftResult { ComplexMatrix epsinv; ComplexMatrix eps2; }`.

**Isotropic** (single flat `eps_grid`):
```
eps_fft = get_conv(dN, eps, ...)         // nG × nG
epsinv  = inv(eps_fft)                    // LAPACK zgetrf + zgetri
eps2    = block_diag(eps_fft, eps_fft)    // 2nG × 2nG
```
**Anisotropic** (3 flat arrays `[epsx, epsy, epsz]`):
```
epsinv = inv(get_conv(epsz))
eps2   = block_diag(get_conv(epsx), get_conv(epsy))
```
The `epsinv`/`eps2` are stored per patterned layer and drive both `kp`
construction and field/`D = εE` reconstruction.

---

## 5. Module: `rcwa` — Main Engine

File: `include/cpprcwa/rcwa.h`, `src/rcwa.cpp`. The `RCWA` class owns all
layer/material state and the S-matrix machinery.

### 5.1 Public API and workflow

Typical usage order:
```
RCWA solver(cfg);
solver.Add_LayerUniform(...);        // layer 0 MUST be uniform (§6.2)
solver.Add_LayerGrid(...);
solver.Init_Setup(Pscale, Gmethod);  // reciprocal lattice + uniform eigensystems
solver.GridLayer_geteps(ep_all);     // patterned-layer eps → kp/q/phi
solver.MakeExcitationPlanewave(exc);
RTResult rt = solver.RT_Solve(normalize, byorder);
```

**Layer specification**
- `Add_LayerUniform(thickness, epsilon)` — appends a uniform layer; records
  `material_idx_` = uniform index.
- `Add_LayerGrid(thickness, Nx, Ny)` — appends a patterned layer; records
  `material_idx_` = patterned index and `grid_idx_` = grid counter. Layer 0
  must be uniform (asserted in `Init_Setup`).

**`Init_Setup(Pscale = 1.0, Gmethod = 0)`**
1. Assert layer 0 is uniform; compute `ω = 2π·freq` (complex).
2. `Lk1, Lk2 = Lattice_Reciprocate(L1, L2)`, then divide by `Pscale`.
3. `G, nG = Lattice_getG(nG_req, Lk1, Lk2, Gmethod)`.
4. `kx0 = ω·sinθ·cosφ·√ε₀`, `ky0 = ω·sinθ·sinφ·√ε₀` from the first uniform
   layer's ε; then `Lattice_SetKs` fills `kx_, ky_`.
5. `normalization_ = √ε₀ / cosθ` (used when `normalize=1`).
6. For every uniform layer: build `kp`, solve the uniform eigensystem, store
   into `kp_list_`, `q_list_`, `phi_list_`. Patterned entries are filled
   later by `GridLayer_geteps`.

**`MakeExcitationPlanewave(exc)`**
Projects the `(p, s)` amplitudes onto the `(x, y)` Fourier-amplitude basis
(mirrors `rcwa.py:125-153`):
```
a_x = -s_amp·cosθ·cosφ·e^{i·s_phase} − p_amp·sinφ·e^{i·p_phase}
a_y = -s_amp·cosθ·sinφ·e^{i·s_phase} + p_amp·cosφ·e^{i·p_phase}
a0[order] = a_x;  a0[order+nG] = a_y       // forward
bN[order] = a_x;  bN[order+nG] = a_y       // backward
```
This is **not** `a0[order]=p_amp` — the trig factors are essential for
oblique-incidence energy conservation.

**`GridLayer_geteps(ep_all)`**
Takes a flat concatenation of all patterned-layer grids (`[layer0(N0²),
layer1(N1²), …]`). For each grid layer: `Epsilon_fft` → `patterned_epinv_` /
`patterned_ep2_`, then `MakeKPMatrix_patterned` + `SolveLayerEigensystem_patterned`
to fill `kp_list_`, `q_list_`, `phi_list_`. An anisotropic overload exists but
throws `NotImplementedError` for the multi-layer path.

### 5.2 Private helpers (layer physics)

**`MakeKPMatrix_uniform(ω, kx, ky, ε, kp)`**
```
Jk    = vstack(diag(-ky), diag(kx))        // (2nG × nG), zero-initialized
kp    = ω²·I − (1/ε)·(Jk·Jkᵀ)
```
`Jk·Jkᵀ` yields the `2×2` block form
`[[diag(ky²), diag(-ky·kx)], [diag(-kx·ky), diag(kx²)]]` — the cross terms
come from the rectangular `(2nG × nG)` shape. **Must** be zero-initialized;
only the diagonal is set.

**`MakeKPMatrix_patterned(ω, kx, ky, epinv, ep2, kp)`**
```
Jk  = vstack(diag(-ky), diag(kx))          // (2nG × nG), zero-initialized
kp  = ω²·I − (Jk·epinv)·Jkᵀ
```
`ep2` is intentionally unused here (matches the Python `MakeKPMatrix`).

**`SolveLayerEigensystem_uniform(ω, kx, ky, ε, q, phi)`**
```
q_raw[i] = √(ε·ω² − kx[i]² − ky[i]²)        // branch cut applied (§6.6)
q        = [q_raw; q_raw]                    // 2nG
phi      = I_(2nG)
```

**`SolveLayerEigensystem_patterned(ω, kx, ky, kp, ep2, q, phi)`**
```
k    = vstack(diag(kx), diag(ky))           // (2nG × nG), zero-initialized
kkT  = k·kᵀ                                  // cross terms diag(kx·ky)
M    = ep2·kp − kkT                          // non-Hermitian, 2nG × 2nG
[evals, evecs] = zgeev(M)                    // LAPACK, columns = right eigenvectors
q    = apply_branch_cut(√evals)
phi  = evecs
```
The eigenvalues/vectors are in arbitrary order; the S-matrix is
order-invariant, but `q[i] ↔ phi[:,i]` pairing must be preserved.

### 5.3 S-matrix algorithm

**`GetSMatrix(indi, indj, S11, S12, S21, S22)`** — Redheffer star product
over layers `[indi, indj)`, each block `2nG × 2nG`:
```
initialize S11=S22=I, S12=S21=0
for l in [indi, indj):
    lp1 = l+1
    Q   = inv(phi[l]) · phi[lp1]
    P   = diag(q[l]) · inv(kp[l]·phi[l]) · kp[lp1]·phi[lp1] · diag(1/q[lp1])
    T11 = ½(Q+P),  T12 = ½(Q−P)
    d1  = diag(exp(i·q[l]·t[l])),  d2 = diag(exp(i·q[lp1]·t[lp1]))
    P1  = inv(T11 − d1·S12·T12)
    S11 = P1 · d1 · S11
    S12 = P1 · (d1·S12·T11 − T12) · d2
    S21 = S21 + S22·T12·S11
    S22 = S22·T11·d2 + S22·T12·S12
```
Every matrix inversion goes through `zinverse` (LAPACK `zgetrf`+`zgetri`).
Phase factors use **element-wise** `.array().exp()` (never a matrix
exponential).

**`SolveExterior(a0, bN, aN, b0)`**
```
S = GetSMatrix(0, Layer_N−1)
[aN; b0] = S · [a0; bN]
```

**`SolveInterior(which_layer, a0, bN, ai, bi)`**
```
L = GetSMatrix(0, which_layer);  R = GetSMatrix(which_layer, Layer_N−1)
ai = inv(I − L12·R21) · (L11·a0 + L12·R22·bN)
bi = R21·ai + R22·bN
```

**`TranslateAmplitudes(q, thickness, dz, ai, bi, aim, bim)`**
```
aim = ai · exp(i·q·dz)
bim = bi · exp(i·q·(thickness − dz))
```

### 5.4 Poynting flux / `RT_Solve`

Module-local `poynting_flux(ω, kp, phi, q, ai, bi)` implements
`GetZPoyntingFlux`:
```
A    = kp·phi·diag(1/(ω·q))
pa   = phi·ai,  pb = phi·bi
Aa   = A·ai,    Ab = A·bi
diff = ½·(conj(pb)∘Aa − conj(Ab)∘pa)                 // element-wise
fwd  = Σ[ Re(conj(Aa)∘pa) + diff ]                    // head(n)+tail(n)
bwd  = Σ[ −Re(conj(Ab)∘pb) + conj(diff) ]
```

**`RT_Solve(normalize, byorder)`**
```
aN, b0 = SolveExterior(a0, bN)
(fi, bi) = flux(a0, b0, layer 0)
(fe, be) = flux(aN, bN, layer Layer_N−1)
forward:  R = −bi,  T = fe
backward: R = fe,   T = −bi
if normalize: R *= normalization_,  T *= normalization_
```
`normalization_ = √ε₀/cosθ`. `byorder` is not yet wired into the per-order
arrays (accepted but currently unused — see §8).

### 5.5 Amplitudes and fields

**`GetAmplitudes_noTranslate(which_layer)`** → `SolveInterior` at that layer.
Layer 0 / last layer are handled uniformly by `SolveInterior` (equivalent to
the Python special cases since the S-matrix is defined identically).

**`GetAmplitudes(which_layer, z_offset)`** → translates the interior
amplitudes to a `z`-offset inside the layer.

**`Solve_FieldFourier(which_layer, z_offsets)`** — per `z`:
```
ai, bi = translated amplitudes
fhxy = phi·(ai + bi);          fhx = fhxy[0:nG],  fhy = fhxy[nG:2nG]
fexy = kp·phi·((ai−bi)/(ω·q)); fey = −fexy[0:nG], fex = fexy[nG:2nG]
fhz  = (kx∘fey − ky∘fex)/ω
fez  = (ky∘fhx − kx∘fhy)/ω
      uniform:  fez /= ε_uniform
      patterned: fez = epinv · fez
```
The `fey = −fexy[:nG]` sign is **intentional** (amplitude-vector ordering maps
`(E_y, −E_x)`); see `PLAN.md §6.9`.

**`Solve_FieldOnGrid(which_layer, z_offsets, Nxy)`** — runs
`Solve_FieldFourier`, then `get_ifft` each of the six components to an
`(Nx, Ny)` `GridMatrix`. `Nxy` defaults to the layer's stored grid size.

### 5.6 Post-processing

**`Return_eps(which_layer, Nx, Ny, component)`**
- Uniform: returns a constant `ε·ones(Nx, Ny)`.
- Patterned: takes **row 0** of the relevant `eps2` block (`xx/xy/yx/yy`) or
  `inv(epsinv)` (`zz`) and reconstructs real space via `get_ifft`.
  (This mirrors `rcwa.py` exactly — row 0 of the convolution matrix holds the
  needed Fourier coefficients.)

**`Volume_integral(which_layer, Mx, My, Mz, normalize)`**
Computes `1/A·∫V (Mx·|Ex|² + My·|Ey|² + Mz·|Ez|²)` for absorption:
```
ab   = [ai; bi]                       // 4nG, from SolveInterior
abM  = outer(ab, conj(ab)) ∘ Mt        // ∘ = Hadamard, Mt = Matrix_zintegral
Faxy = kp·phi·diag(1/(ω·q))            // 2nG × 2nG
Faz  = [ (1/ω)·epinv·diag(ky), −(1/ω)·epinv·diag(kx) ] · phi   // nG × 2nG
F    = [[Faxy, −Faxy], [Faz, Faz]]     // 3nG × 4nG
Mtotal = block_diag(Mx, My, Mz)        // 3nG × 3nG
val   = trace(abM · (F† · Mtotal · F))
if normalize: val *= normalization_
```
`Matrix_zintegral(q, thickness, shift=1e-12)` builds the `4nG×4nG` z-integral
matrix from the `Maa`/`Mab` formulas with the asymmetric diagonal `shift`
stability term (only on `qij = qj − conj(qi)`, per `PLAN.md §6.8`).

**`Solve_ZStressTensorIntegral(which_layer)`** — returns `(2Fx, 2Fy, 2Fz)`:
```
dz = (ky∘hx − kx∘hy)/ω                     # = D_z
uniform:   dx = ε·ex,  dy = ε·ey
patterned: dxy = eps2·[−ey; ex];  dx = dxy[nG:],  dy = −dxy[0:nG]
Tx = Re Σ(ex·conj(dz) + hx·conj(hz))
Ty = Re Σ(ey·conj(dz) + hy·conj(hz))
Tz = Re ½ Σ(ez·conj(dz) + hz·conj(hz) − ey·conj(dy) − ex·conj(dx) − |hx|² − |hy|²)
```

---

## 6. Internal helpers

### `internal/lapack_wrappers.{h,cpp}`
Direct Fortran LAPACK bindings (`extern "C"`), throwing on `info ≠ 0`:
- `zgetri_inplace(n, A, lda, layer)` — `zgetrf` + workspace-query `zgetri`.
- `zinverse(A, layer)` — copy + `zgetri_inplace` (used everywhere an inverse
  is needed, incl. the S-matrix star product).
- `zgesv(n, nrhs, A, B, ...)` — `zgetrf` + `zgetrs` (reserved).
- `zgeev(n, A, w, VR, ...)` — full eigendecomposition of the non-Hermitian
  `M` matrix; `VR` columns are right eigenvectors.

### `internal/branch_cut.{h,cpp}`
`apply_branch_cut(q)` forces `Im(q) ≥ 0` element-wise (evanescent waves decay
in +z): `q[i].imag() < 0 ⇒ q[i] = −q[i]`.

### `internal/utils.{h,cpp}`
`diag`, `eye`, `makeDiag`, `blockDiag`, `vstack`, `hstack` — Eigen helpers
mirroring NumPy's `np.diag`/`np.eye`/`np.block`/`np.vstack`/`np.hstack`.

---

## 7. Validation Summary

All validated against grcwa (Python) / S4:

| Case | C++ | grcwa | match |
|---|---|---|---|
| S4 golden p-pol `T` | 0.853173 | 0.852499* | within `1e-3` ✓ |
| S4 golden s-pol `T` | 0.839654 | 0.839005* | within `1e-3` ✓ |
| ex1 (square holes) `R,T` | 0.132415, 0.867585 | 0.13241493, 0.86758507 | ✓ |
| ex2 (two layers) `R,T` | 0.132599, 0.867401 | 0.13259894, 0.86740106 | ✓ |
| ex4 (hexagonal) `R,T` | 0.164450, 0.835550 | 0.16445004, 0.83554996 | ✓ |
| EUV mirror @13.5 nm `R` | 0.599171 | 0.599171 | exact ✓ |
| EUV mirror @13.7 nm `R` | 0.647593 | 0.647593 | exact ✓ |
| `FieldFourier` component norms | §test_rcwa | exact to ~1e-12 | ✓ |
| `Volume_integral(real(epinv))` | 0.196376 | 0.196392 | ~8e-5 rel ✓ |
| `Solve_ZStressTensorIntegral(0)` | −0.139217, −0.050671, −1.09948 | identical | ✓ |

\* S4 reference value embedded in the golden test; grcwa itself computes
0.853173 / 0.839654, which cpprcwa reproduces exactly.

Additional invariants tested: `R+T = cos(θ)` for lossless oblique uniform
stacks (energy conservation), `nG_out ≤ nG`, degenerate-G symmetry, FFT
DC-component equality, and `Layer 0 must be uniform`.

### 7.1 EUV multilayer mirror example

`examples/ex_euv_multilayer.cpp` computes the reflectivity of an EUV Mo/Si
multilayer mirror (validated against grcwa via `scripts/compare_euv.py`):

- **Stack** (from the incident side): vacuum / Ru cap 2.5 nm /
  40× [Mo 2.8 nm, Si 4.2 nm] (period 7.0 nm, Γ≈0.4) / Si substrate.
- **Normalization**: `freq = 1/λ` with thicknesses in nm, so the phase
  across a layer is `2π·n·t/λ`. λ = 13.5 nm → `freq = 0.074074`.
- **Truncation**: a planar stack has no lateral pattern, so only the specular
  order propagates. Requesting `nG=2` yields `nG_out=1` via the circular
  truncation's shell trimming (grcwa-compatible).
- **Result**: R = 0.599171 at 13.5 nm (identical to grcwa); a wavelength scan
  (`--scan`) shows the Bragg peak at ≈13.7 nm (R = 0.647593) because the
  geometric period of 7.0 nm combined with these optical constants peaks
  slightly above the 13.5 nm design wavelength.

---

## 8. Known Gaps / Notes- **`RT_Solve(byorder=true)`** — per-order `R_per_order`/`T_per_order` arrays
  are declared but not yet populated (scalar path is correct and validated).
- **Anisotropic `GridLayer_geteps`** — the multi-layer anisotropic path throws
  `NotImplementedError`; no example exercises it.
- **`Add_LayerFourier`** — not implemented (stub); throws if called.
- **GPU backend (Phase 9)** — deferred; see `PLAN.md` for the `zgeev`
  feasibility spike that should precede it.
- **OpenMP layer parallelism** — deferred; FFTW thread-safety caveats in
  `PLAN.md §6.11` apply.
- Eigenvector **ordering/phase** differ from NumPy's — comparisons must use
  physical observables (this port does; see `PLAN.md §6.7, §9.2`).
