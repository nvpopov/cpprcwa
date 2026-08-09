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
│   ├── ex_euv_multilayer.cpp   # EUV Mo/Si multilayer mirror (Ru cap) reflectivity
│   ├── ex_euv_absorber.cpp     # EUV TaN absorber pattern on top of the mirror
│   └── ex_quasi1d_absorber.cpp # Quasi-1D line grating (500 nm TaN bar, Lx=3.5 um)
├── python/
│   ├── cpprcwa_bind.cpp       # nanobind bindings (grcwa-compatible obj + helpers)
│   └── grcwa/__init__.py      # drop-in shim: import grcwa → cpprcwa
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

**Storage dedup (Aug 2026):** `kp_list_`/`q_list_`/`phi_list_` hold
`shared_ptr<const>` per layer. Uniform layers with identical ε share the same
`(kp, q)` object, and every uniform layer points at a single shared Identity
`phi` (the uniform eigensystem always yields `phi = I`). Patterned layers own
their matrices. Periodic stacks therefore keep ~nDistinct full matrices
instead of one copy per layer (~20× less persistent memory; private accessors
`kp(li)`/`q(li)`/`ph(li)` dereference for the hot paths).

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

**`ForwardPropagatedFieldFourier(which_layer, z_offset)`** /
**`BackwardPropagatedFieldFourier(which_layer, z_offset)`** — field
contribution from the forward (`ai`) or backward (`bi`) amplitudes alone,
evaluated in `which_layer` at depth `z_offset` (0 = front interface). For
layer 0 at z=0, `BackwardPropagatedFieldFourier` is the reflected field in
air; used to validate every reflected order against grcwa.

### 5.6 Post-processing

**`Return_eps(which_layer, Nx, Ny, component)`**
- Uniform: returns a constant `ε·ones(Nx, Ny)`.
- Patterned: takes **row 0** of the relevant `eps2` block (`xx/xy/yx/yy`) or
  `inv(epsinv)` (`zz`) and reconstructs real space via `get_ifft`.
  (This mirrors `rcwa.py` exactly — row 0 of the convolution matrix holds the
  needed Fourier coefficients.)

**`Volume_integral(which_layer, Mx, My, Mz, normalize)`**
Computes `1/A·∫V (Mx·|Ex|² + My·|Ey|² + Mz·|Ez|²)` for absorption.

**Computed block-wise (Aug 2026, PLAN.md §10.4)** — the 4nG×4nG
`outer(ab,conj(ab))`, `Mt`, `abM` and the 3nG×3nG `Mtotal`/`F` are **never
materialized** (peak ~1 GiB @ nG=1000 in the original). The block structure of
`F = [[Faxy,−Faxy],[Faz,Faz]]` and `Mt = [[Maa,Mab],[Mab,Maa]]` collapses the
trace onto 2nG×2nG blocks:
```
Mxy = block_diag(Mx, My)
C   = Faxy†·Mxy·Faxy ;  D = Faz†·Mz·Faz
A   = C + D ;  B = D − C           // T = F†·Mtotal·F = [[A,B],[B,A]]
W_A = Maa∘Aᵀ ;  W_B = Mab∘Bᵀ
val = aiᵀ(W_A)conj(ai) + biᵀ(W_A)conj(bi) + aiᵀ(W_B)conj(bi) + biᵀ(W_B)conj(ai)
if normalize: val *= normalization_
```
Identical to the original formula to ~1e-16 (golden test passes).
`matrix_zintegral_blocks(q, thickness, shift=1e-12)` returns the `Maa`/`Mab`
blocks (2nG×2nG each) directly from the `qj − conj(qi)` / `qj + conj(qi)`
formulas, with the asymmetric diagonal `shift` stability term applied only to
`qij = qj − conj(qi)` (per `PLAN.md §6.8`).

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
| EUV absorber @13.5 nm `R` (nG=97) | 0.578790 | 0.578790 | exact ✓ |
| EUV absorber @13.5 nm `R` (nG=9) | 0.566750 | 0.566750 | exact ✓ |
| EUV absorber @13.5 nm, θ=6° `R` (nG=9) | 0.587072 | 0.587072 | exact ✓ |
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

### 7.2 EUV absorber-on-mirror example

`examples/ex_euv_absorber.cpp` adds a periodic TaN absorber pattern on top of
the mirror (EUV mask geometry), validated against grcwa via
`scripts/compare_euv.py --absorber`:

- **Stack**: vacuum / **TaN absorber pattern** (300 nm cell, 60×60 nm
  rectangle, `n = 0.9562+0.0323i`, height 60 nm) / Ru cap 2.5 nm /
  40× Mo/Si bilayers / Si substrate.
- **Patterned layer**: `Add_LayerGrid(abs_t, Nx, Ny)`; the grid is 1 for
  vacuum and `n_tan²` inside the rectangle; all other (isotropic) machinery
  (FFT convolution → `Epsilon_fft` → non-Hermitian `zgeev`) applies.
- **Truncation / resolution**: `nG` is a CLI argument (default 101 → `nG_out
  = 97`), grid `Nx×Ny` defaults to 300 (1 nm pixels). Convergence: R = 0.5668
  (nG=9), 0.5788 (nG=97), 0.5791 (nG=197) — ~0.06% converged at nG≈100.
- **Result**: R = 0.578790 at 13.5 nm (grcwa-identical), vs 0.599171 for the
  bare mirror — the TaN absorber (4% areal fill) absorbs/scatters ~2% of the
  incident power.

#### Reflected-field validation (air side) & visualization

`ex_euv_absorber --field OUT` writes the reflected field in the incident
medium (backward-only, z=0): per-order Fourier coefficients (`OUT_orders.txt`),
the real-space `|E|²` grid (`OUT_grid.txt`), horizontal / vertical
cross-sections of `|E|²` through the cell center (`OUT_hscan.txt`,
`OUT_vscan.txt`), and the full complex field `Re/Im (Ex, Ey, Ez)` along both
cross-section lines (`OUT_hfield.txt`, `OUT_vfield.txt`).
`scripts/plot_reflected_field.py` reproduces the same in grcwa, compares all
orders, renders `OUT_plot.png` (2D `|E|²` map + `|E|²` cross-sections),
`OUT_absE.png` (2D `|E|` map + `|E|` cross-sections — the absolute field
value) and `OUT_crossections.png` (Re/Im of Ex, Ey, Ez along the horizontal
and vertical cross-sections, cpprcwa vs grcwa), and reports a wall-clock
performance comparison (cpprcwa timing from `OUT_perf.txt`, grcwa timed
internally).

For the nG=9 fast config both agree with grcwa to **machine precision**:

| quantity | max |cpp − grcwa| |
|---|---|---|
| per-order reflected coefficients (ex, ey, ez, all orders) | 1.6e-14 |
| real-space `|E|²` grid | 3.4e-14 |
| Ex, Ey, Ez along both cross-section lines | 2–4e-14 |

The specular order (G=0) dominates; scattered orders (G=(±1,0), (0,±1), …)
carry the diffraction from the 60 nm TaN rectangle. The real-space map shows
|E|² ≈ 0.74 in open areas, dipping to ≈ 0.39 over the absorber, with
symmetric horizontal/vertical cross-sections through the center.

An **oblique-incidence variant** (θ = 6°) is covered by a dedicated test
(`RCWA EUV absorber on mirror oblique incidence`): R = 0.587072 matches grcwa
exactly, and the reflected specular + scattered orders (matched per G-vector)
agree to machine precision.

### 7.3 Quasi-1D line-grating example

`examples/ex_quasi1d_absorber.cpp` is the same EUV mask stack, but the TaN
absorber becomes a **1D line grating** — a 500 nm bar spanning the full y
extent of a `Lx × Ly` cell with `Lx = 3.5 µm`, `Ly = nly·λ` (default 5λ =
67.5 nm, "several wavelengths"). Validated against grcwa with the identical
geometry.

- **Geometry**: `L1 = (3500, 0)`, `L2 = (0, 67.5)` nm; TaN bar 500 nm wide in
  x, uniform in y (quasi-1D). Same Ru cap / 40× Mo/Si / Si substrate as §7.2.
- **Quasi-1D harmonic set**: because the bar is y-invariant, every y≠0
  Fourier coefficient of ε vanishes and the y≠0 orders are never excited.
  The reciprocal-lattice spacing in y is `1/Ly` (large) vs `1/Lx` in x (tiny),
  so the circular truncation fills the set with the **x-only row (i, 0)**
  first — with `nG=101 → nG_out=99`, 99/99 harmonics are `(i, 0)` (pure 1D);
  larger nG then adds y≠0 harmonics (harmless dead modes that stay at zero
  amplitude). The header prints `x-only N` of `nG` to show this.
- **Order fan**: a 3.5 µm-period grating at 13.5 nm supports ±259 propagating
  x-orders; the example lists the per-order diffraction efficiencies
  (computed from the per-order reflected Poynting flux `0.5·Re(conj Ex·Hy −
  conj Ey·Hx)`), showing the expected symmetric ±m pairs. `sum(eff) = R`.
- **Result** (normal incidence, p-pol, `nG=201 → 199`): R = 0.528927, with
  199 propagating reflected orders; identical to grcwa at nG = 97 / 199 / 295
  (R = 0.528790 / 0.528927 / 0.529005).

#### Quasi-1D performance optimization (`--quasi1d`)

The example exposes a solver flag `cfg.quasi1d = true` that makes two
**exact** reductions for y-invariant structures (the y≠0 harmonics decouple
and stay at zero amplitude, so they carry no physics):

1. **1D harmonic filter** — after the circular `getG` selection, keep only the
   x-only row `(i, j=0)`. For `nG=201` this drops 199 → 113 harmonics (the 86
   y≠0 ones are dead modes). Every matrix shrinks by ~1.8× (cost scales as
   `(2nG)³` → ~5.5× less flops).
2. **Diagonal uniform-core S-matrix** — with the 1D set every uniform-layer
   `kp/q/phi` is *exactly diagonal* (all `ky=0`), so the S-matrix of an
   all-uniform range is diagonal. `GetSMatrix` now detects the trailing
   uniform suffix (the Mo/Si multilayer + substrate), computes it with a
   scalar per-harmonic recursion (O(n) per interface instead of O(n³)), and
   assembles it with the general prefix (vac | grid | Ru) using the
   **overlapping-cascade** formula `M=(I−L12·R21)⁻¹`,
   `S11=R11·M·L11`, `S12=R11·M·L12·R22+R12`,
   `S21=L21+L22·R21·M·L11`, `S22=L22·R21·M·L12·R22+L22·R22`
   (identical to the `SolveInterior` split, validated to ~2e-15 in Python and
   ~1e-13 in the per-order field comparison). The star-product step was also
   factored into a reusable lambda so the general path is bit-identical.

Wall-clock (nG=201, 40 Mo/Si bilayers, median):

| build | total | vs grcwa (11.0 s) |
|---|---|---|
| original (199 harmonics) | ~7.2 s | 1.5× |
| + uniform-layer `(kp,q,φ)` caching (81→5 setups) | ~5.7 s | 1.9× |
| + `--quasi1d` 1D filter (199→113) | ~1.6 s | 6.9× |
| + diagonal uniform-core fast path | **~0.37 s** | **~30×** |

R is unchanged (0.528927 = grcwa) at every stage; the per-order reflected
field agrees with the general path to ~6e-14 and with grcwa to ~1e-13, at
normal incidence and at θ = 6° / 30° (agreement ~1e-14).

**Note on the eps-grid convention:** the examples and `scripts/*.py` place
`n` (not `n²`) in the patterned grid (a pre-existing convention shared with
grcwa). Keep both sides consistent; using `n²` gives a different (physical)
permittivity and different R.
- `--field OUT` writes per-order coefficients (`OUT_orders.txt`), per-order
  efficiency + angle (`OUT_effic.txt`), the real-space `|E|²` grid
  (`OUT_grid.txt`) and an x-scan through the cell center (`OUT_vscan.txt`).
  `scripts/plot_quasi1d_field.py` reconstructs the reflected field from the
  Fourier coefficients and plots the **absolute field value** `|E|`
  (`OUT_absE.png`): a cross-section along x at the cell center (cpprcwa vs
  grcwa, with the absorber bar marked) and a 2D `|E|` map over the cell
  showing the y-uniform quasi-1D nature.

### 7.4 Performance vs grcwa (EUV absorber)

Optimizations applied (all verified to keep the field agreement at ~1e-14):

1. **`EIGEN_USE_BLAS`** — matrix products via OpenBLAS `zgemm` (~4.4× on matmul
   vs Eigen's single-threaded kernels).
2. **`FFTW_ESTIMATE`** plans (cached per `(Nx,Ny)`) — fast planning.
3. **OpenBLAS thread capping** at `min(cores, 6)` — with more threads the small
   `2nG×2nG` S-matrix blocks suffer thread-pool overhead (12 threads ≈ 4.9 s vs
   6 threads ≈ 1.3 s measured).
4. **Uniform-pair T-matrix caching** in the S-matrix — for periodic stacks
   (Mo/Si multilayers) the interface matrices `T11/T12` of identical
   uniform-uniform layer pairs are computed once and reused (phi = I ⇒ Q = I,
   so no per-pair inversion).
5. **LAPACK workspace reuse** (`thread_local`) — avoids a fresh `lwork` query +
   allocation per inversion (249 inversions during a solve).
6. **LU-solve + common subexpressions** in the star product — replaces forming
   `inv(P1m)` with one `zgetrf` + two `zgetrs`, and hoists `d1·S12`, `S22·T12`.

Wall-clock (whole pipeline: build + setup + solve), median of runs, nG=97:

| metric | before | after |
|---|---|---|
| setup (FFT + `zgeev` + eps) | ~340 ms | ~243 ms |
| `rt_solve` (S-matrix) | ~1700 ms | ~813 ms |
| total | ~2000 ms | ~1050 ms |

grcwa total for the same case ≈ 5.8 s → cpprcwa ≈ **5× faster** (whole
pipeline) while matching every reflected order and the real-space field to
~1e-14.

**Aug 2026 round 2 (periodic stacks / quasi-1D):**
- **Diagonal phase factors** (`d1`/`d2` as vectors, `.asDiagonal()` scaling)
  remove 3 O(n³) matmuls per Redheffer step.
- **Periodic uniform-core binary exponentiation** (full 2D): the 40×(Mo,Si)
  suffix is `period^(R-1)` + the last period's real boundary interface,
  assembled with the exact Redheffer cascade (`redheffer_cascade`). Identical
  to the sequential path to ~4e-15. (The earlier naive `S^R` doubling was
  wrong — it ends with a phantom reference layer that mismatches the substrate
  boundary.)
- **Block-diagonal M detection** in the patterned eigensolver: exactly
  zero off-diagonal blocks (quasi-1D normal incidence; D4-symmetric 2D at
  normal incidence) → two nG×nG `zgeev`s instead of one 2nG×2nG (4× flops).
- **S-matrix memoization** — no redundant S(0, N-1) rebuild for field calls.
- **TE/TM block-decoupled quasi-1D path** (ky0=0, QUASI-1D NORMAL): with
  block-diagonal kp/phi/q the grid-interface prefix and cascade run
  per-polarization on nG×nG blocks (`bstep`), 4× fewer flops; identical to
  the general path to ~4e-15. Oblique (φ≠0°) falls back to the general path.

Measured `rt_solve` (system under load, indicative) — mode-explicit:
**FULL-2D NORMAL** (θ=0°,φ=0°) nG=97 963→~300 ms, nG=201 6420→~1500 ms
(~4×); **QUASI-1D OBLIQUE** (θ=6°,φ=45°) nG=113 total 145 ms. `R` unchanged
and matching grcwa in every mode. Full tables in `benchmarks/README.md`.

**Note on S-matrix doubling:** the FIRST attempt (naive `S^R` recombination)
was reverted — it ends with a phantom reference layer that mismatches the
boundary medium (drops inter-period interfaces, double-counts boundary phases,
~4e-4 R error). The current implementation (Aug 2026) exponentiates
`period^(R-1)` and cascades with the last period's real boundary interface via
the exact Redheffer cascade — identical to the sequential path to ~4e-15.

### 7.5 Python bindings (nanobind) — grcwa drop-in

`CPPRCWA_BUILD_PYTHON=ON` (default) builds a nanobind module
(`cpprcwa.cpython-*.so`) plus a `grcwa/` shim package in the build directory.

- **`obj` class** mirrors `grcwa.obj` 1:1: constructor
  `(nG, L1, L2, freq, theta, phi, verbose=1, quasi1d=False)` (L1/L2 accept
  Python lists), `Add_LayerUniform/Grid/Fourier`, `Init_Setup`,
  `GridLayer_geteps` (isotropic; anisotropic → `NotImplementedError`),
  `MakeExcitationPlanewave`, `RT_Solve`, `GetAmplitudes(_noTranslate)`,
  `Solve_FieldFourier/OnGrid`, `Return_eps`, `Volume_integral`,
  `Solve_ZStressTensorIntegral`, and read attributes
  `nG, Layer_N, G, kx, ky, q_list, phi_list, kp_list, thickness_list,
  Uniform_ep_list, GridLayer_Nxy_list, Patterned_epinv_list,
  Patterned_ep2_list, freq, omega, L1, L2, theta, phi, a0, bN, direction,
  normalization, id_list`.
- **Module helpers**: `Lattice_Reciprocate`, `Lattice_getG`, `Lattice_SetKs`,
  `get_fft`, `get_ifft`, `Epsilon_fft`.
- **Validation**: grcwa's own `ex1.py` (nG=301, R=0.13241493181686462),
  `ex2.py` (oblique, 2 patterned layers, R=0.1325989361682126) and `ex4.py`
  (hexagonal, R=0.16445003997328628) run unchanged via the shim and match the
  original grcwa to ~1e-13. `quasi1d=True` on `obj` exposes the fast path
  (nG=201→113, R=0.528927) from Python.
- **Implementation notes**: constructor uses `nb::pointer_and_handle` +
  placement-new (nanobind `init<>` can't convert list→Eigen/numpy and has no
  convert flag for init args); member lambdas take `RCWA&` explicitly;
  Eigen/matrix returns are passed by value to avoid lifetime issues; the
  `std::complex<double>` caster requires `<nanobind/stl/complex.h>`.

### 7.6 Memory reporting & reductions (Aug 2026)

**`RCWAConfig::report_memory`** (exposed as `--mem` in the examples) makes
`Init_Setup` print an estimate of the required peak memory. All sizes depend
only on `nG` and the layer structure, so it is exact at that point:
`PrintMemoryReport()` reports persistent layer storage, the uniform-pair
T-matrix cache, the `RT_Solve` and `Volume_integral` transient peaks, and the
estimated peak RSS. Validated against `/usr/bin/time -v` RSS to within ~5–9%.

Two reductions (results bit-identical, `R` unchanged at every case):

| measure | before | after |
|---|---|---|
| quasi-1D nG=113 persistent storage | 132 MiB | **6.5 MiB** |
| quasi-1D nG=253 persistent storage | 662 MiB | **32 MiB** |
| quasi-1D nG=113 measured peak RSS | 168 MiB | **42 MiB** |
| full-2D nG=199 measured peak RSS | 503 MiB | **114 MiB** |

- **Uniform-layer dedup** (§5.1): `kp`/`q`/`phi` shared via `shared_ptr<const>`
  across identical ε (one matrix per distinct material + one shared Identity
  `phi`), instead of a per-layer copy.
- **Block-wise `Volume_integral`** (§5.6): no 4nG×4nG / 3nG×3nG matrices.
- Remaining: the `GetSMatrix` star-product scratch arena is a *speed* (not
  peak-RSS) optimization — the per-step peak is already bounded (~20 live
  2nG×2nG matrices).

---

## 8. Known Gaps / Notes- **`RT_Solve(byorder=true)`** — per-order `R_per_order`/`T_per_order` arrays
  are declared but not yet populated (scalar path is correct and validated).
- **Anisotropic `GridLayer_geteps`** — the multi-layer anisotropic path throws
  `NotImplementedError`; no example exercises it.
- **`Add_LayerFourier`** — not implemented (stub); throws if called.
- **GPU backend (Phase 9)** — **descoped** after the feasibility spike
  (`benchmarks/bench_zgeev.cpp`): cuSOLVER has no `cusolverDnZgeev` (verified
  11.5 + 12.8 headers — only Hermitian `Zheevd/j` and SVD `Zgesvd/j`), and
  cuBLAS `zgemm` matches OpenBLAS but does not beat it at the `(2nG)³`
  S-matrix size (0.7–1.1×). Only cuFFT helps (~20× on the eps FFT, a small
  setup fraction). `zgeev` and the S-matrix stay on CPU; see PLAN.md §Phase 9.
  Cheaper CPU speedup: Intel MKL threaded `zgeev` (≈1.6–2×, see README).
- **OpenMP layer parallelism** — deferred; FFTW thread-safety caveats in
  `PLAN.md §6.11` apply.
- Eigenvector **ordering/phase** differ from NumPy's — comparisons must use
  physical observables (this port does; see `PLAN.md §6.7, §9.2`).
