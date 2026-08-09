cpprcwa — Understanding & Suggested Improvements
I've read the summary carefully. This is a very mature, well-engineered RCWA port: exact numerical validation against grcwa/S4, a series of structure-exploiting optimizations (all bit-identical), memory accounting, Python drop-in bindings, and honest documentation of gaps. Below are concrete suggestions grouped by category.

1. Finishing the known gaps (highest value / lowest risk)
1.1 RT_Solve(byorder=true) per-order arrays
This is the biggest feature gap. You already compute per-order reflected Poynting flux in the quasi-1D example (0.5·Re(conj Ex·Hy − conj Ey·Hx)), so the physics is in place. Wire it into RTResult.R_per_order/T_per_order by evaluating the flux per G-index instead of summing. This unblocks diffraction-efficiency workflows (which are the main reason people use RCWA for gratings) directly from the core API instead of ad-hoc example code.

1.2 Anisotropic multi-layer path
Currently throws NotImplementedError. The single-layer machinery exists (Epsilon_fft anisotropic branch). Since the block shortcuts (§7.7.6) explicitly rely on isotropy (C·epinv=I), make sure the anisotropic path cleanly forces the full 2n×2n zgeev fallback — the §7.7.8 detector should already do this, but it deserves an explicit test.

1.3 Add_LayerFourier
A stub. If any target workload needs analytic Fourier layers (e.g. smooth gradients), implement it; otherwise document it as permanently descoped to stop it looking like unfinished work.

2. Numerical robustness
2.1 Formulation stability (Li's factorization rules)
The summary uses the classic M = ep2·kp − kkT formulation. For high-contrast / metallic gratings (TaN, Mo are lossy metals at EUV), the naive Laurent rule converges slowly in TM. Consider implementing Li's inverse rule / normal-vector field (Fast Fourier Factorization) for the in-plane ε. This can dramatically improve convergence vs nG (fewer harmonics for the same accuracy) — a physics speedup that compounds with all your flops optimizations. This is arguably higher-impact than any remaining flops tuning, since cost scales as (2nG)³.

2.2 Ill-conditioned zgeev guard
§7.7.6 already falls back when a TE/TM pair is degenerate. Extend this idea: after zgeev, optionally check eigenvector-matrix conditioning (phi inversion appears in every S-matrix step). A cheap cond(phi) estimate (1-norm via zgecon reusing the LU) could warn/flag pathological layers rather than silently propagating loss of precision.

2.3 Deterministic eigenvector normalization
The summary notes eigenvector ordering/phase differ from NumPy. For reproducibility across BLAS/LAPACK versions, consider a canonical phase fix (e.g. rotate each column so its largest-magnitude entry is real-positive). Doesn't change observables but makes golden-field regression tests robust to LAPACK swaps.

3. Performance (beyond what's done)
3.1 The zgeev bottleneck
With the S-matrix now heavily optimized, zgeev in setup is increasingly the floor (especially full-2D). Options:

MKL threaded zgeev (already noted, ~1.6–2×) — make it a documented build option.
Parallelize distinct patterned-layer zgeev calls across layers with a thread pool (independent problems). This is safer than the deferred OpenMP layer parallelism since eigensolves don't touch FFTW.
Reuse across a frequency/wavelength scan: for scans (your --scan mode), the G-set and grid are fixed; only ω changes. There's no eigenvalue-perturbation shortcut for zgeev in general, but you can at least parallelize scan points trivially (embarrassingly parallel over wavelengths).
3.2 Redheffer scratch-arena reuse
§7.6 notes the star-product arena is a speed (not RSS) optimization. If not already done, preallocate a fixed set of 2nG×2nG scratch buffers once per RT_Solve and reuse them across all interfaces — avoids repeated allocation/deallocation churn that hurts at large nG.

3.3 Single-precision / mixed-precision path
For screening/optimization sweeps where 1e-4 accuracy suffices, a complex<float> (cgeev/cgemm) path can be ~2× faster and half the memory. Gate it behind a template parameter or config flag; keep double as default.

4. Correctness & maintainability
4.1 Property-based / conservation tests
You test R+T=cos(θ) for lossless stacks. Extend with randomized property tests: random lossless multilayers must conserve energy to ~1e-12; reciprocity checks (swap source/detector orders). This catches regressions the golden tests can't (they only cover fixed geometries).

4.2 CI matrix for the block shortcuts
The §7.7.8 auto-fallback is the riskiest correctness surface (silent wrong results if the detector threshold is wrong). Add explicit tests that force each mode and compare against the full path for:

near-threshold cases (weakly y-varying pattern),
anisotropic + oblique (must fall back),
degenerate TE/TM (must fall back). Assert the chosen path (expose the mode enum to tests), not just the result.
4.3 Cache invalidation audit
Multiple caches now exist (FFTW plans, uniform (kp,q,φ) dedup, uniform-pair T-matrix, S-matrix memoization). Document and test when they invalidate — e.g. calling GridLayer_geteps twice, or mutating a layer after Init_Setup. A silent stale-cache bug here would be very hard to diagnose.

5. Documentation / API polish
§8 formatting bug: ## 8. Known Gaps / Notes- has the first bullet glued to the heading. Minor but visible.
Config surface asymmetry: Qabs folded into freq while Pscale/Gmethod are Init_Setup args, and quasi1d is on the config. This mirrors grcwa but is a bit surprising. A short "why each parameter lives where" table would help new users.
Thread-safety contract: state explicitly whether one RCWA object is reentrant / whether two objects can solve concurrently (FFTW plan cache is mutex-guarded, but BLAS thread capping is global). Important for the scan-parallelism suggestion above.
Priority ranking
Priority	Item	Rationale
High	Li/FFF factorization (2.1)	Physics-level convergence win; compounds with all flops work
High	byorder R/T arrays (1.1)	Core feature gap; physics already exists
High	Scan-point parallelism (3.1)	Trivial, huge wall-clock win for sweeps
Medium	Property/reciprocity tests (4.1)	Regression safety
Medium	Forced-mode block-shortcut tests (4.2)	Guards the riskiest correctness surface
Medium	Parallel per-layer zgeev (3.1)	Setup-time win, safe
Low	Mixed precision (3.3)	Niche but easy
Low	Doc/formatting fixes (5)	Polish
The single most impactful suggestion is implementing Li's factorization rules (2.1) — every other optimization reduces the cost per harmonic, but better factorization reduces the number of harmonics needed, which is cubically leveraged. For a lossy EUV metal absorber this could be a 2–4× reduction in required nG at fixed accuracy.

Would you like me to sketch the implementation of any of these (e.g. the byorder flux loop, or the FFF normal-vector construction)?

09.08.2026 21:01

try again and suggset something really significant


09.08.2026 21:02
The Really Significant One: Make cpprcwa Differentiable (Adjoint Gradients → Inverse Design Engine)
Why this is the step-change, not an optimization
Look at what cpprcwa is today versus what grcwa exists for. grcwa's entire reason for being (per its paper) is autograd-differentiable RCWA for inverse design — topology optimization of photonic structures. The C++ port kept the forward solver and made it 5–30× faster, but dropped the gradient capability entirely. That means:

Every optimization loop a grcwa user runs today cannot move to cpprcwa, no matter how fast the forward pass is.
Finite differences are not a substitute: your EUV absorber grid is 300×300 = 90,000 pixels → 90,000 forward solves per gradient. Dead on arrival.
The adjoint method gives all 90,000 pixel-gradients for roughly the cost of ~1 extra field reconstruction — and in your architecture, it's even cheaper than that (see below).
This transforms the project's category: from "fast validator/analyzer" to "the fastest inverse-design RCWA engine available," directly targeting EUV mask optimization (absorber topology, phase-shift masks), grating efficiency shaping, and metasurface design. Everything you've built — the 30× quasi-1D path, the memory reductions, the machine-precision field validation — compounds under an optimizer that calls the solver thousands of times.

Why your existing architecture makes this unusually cheap to build
The key structural insight: the S-matrix is excitation-independent, and you already memoize it (§7.4 round 2: "S-matrix memoization — no redundant S(0,N−1) rebuild for field calls").

Use the physical adjoint method (Lorentz reciprocity), not autodiff through zgeev:

δFoM=ω 
2
 Re∫ 
V
​
 δε(r)E 
fwd
​
 (r)⋅E 
adj
​
 (r)dr
where the adjoint field E 
adj
​
  is generated by sources determined by ∂FoM/∂(output amplitudes). For any FoM defined on reflected/transmitted orders (reflectivity, per-order efficiency, aerial-image intensity), the adjoint source is just a different incident amplitude vector a0/bN — a superposition of plane-wave orders. Consequences:

No second S-matrix build. The adjoint solve is SolveInterior/Solve_FieldFourier with a new RHS against the memoized S-matrix. The dominant O((2nG)³) work is not repeated.
No differentiation of zgeev. The notorious pain points of differentiable RCWA (degenerate eigenvalues, eigenvector phase ambiguity — which you already flag in §8) are completely bypassed. The physical adjoint only needs fields, which you validate to 1e-14.
The z-integral machinery already exists. Volume_integral computes ∫ 
z
​
  of modal-exponential products analytically via matrix_zintegral_blocks (qj − conj(qi) / qj + conj(qi) forms). The adjoint needs ∫ 
z
​
 E 
fwd
​
 ⋅E 
adj
​
  — the unconjugated variant, i.e. the same block formulas with qi instead of conj(qi). A small generalization of code you've already written and block-optimized (§7.7.2).
Per-pixel accumulation is an IFFT you already have. The overlap ε-gradient per real-space pixel is a pointwise product of the two IFFT'd fields → get_ifft per component per layer. O(N log N), negligible.
Thickness gradients come nearly free — analytic derivatives of the diagonal phase factors e 
iq 
ℓ
​
 t 
ℓ
​
 
 , which are already stored as vectors (§7.7.1).
So the total gradient cost is roughly forward solve + one field reconstruction pass — while grcwa's autograd tape replays the entire eigendecomposition chain backward.

The validation story writes itself
This is the part that fits your project's DNA perfectly: grcwa autograd is your golden reference. Extend scripts/generate_golden.py to dump grad(R)(eps_grid) from grcwa for ex1/ex2/quasi-1D/EUV-absorber cases, and compare the C++ adjoint gradient pixel-by-pixel — exactly the same methodology as your reflected-field validation, with the same expected ~1e-10-or-better agreement (adjoint vs autodiff agree analytically; only floating-point paths differ). Add the standard adjoint sanity check (directional finite difference: (FoM(ε+hδ)−FoM(ε−h δ))/2h vs ⟨grad, δ⟩) as a Catch2 property test.

Concrete deliverables
cpp
// 1. Generalized excitation (already internally supported — just expose it)
solver.SetExcitationVector(a0_full, bN_full);          // arbitrary 2nG vectors

// 2. Adjoint volume overlap (generalization of Volume_integral)
complex ov = solver.Volume_overlap(layer, Ef, Ea);     // unconjugated z-integral

// 3. The headline API
GradResult g = solver.Gradient(fom_spec);              // per-pixel dFoM/dε per grid layer
                                                       // + dFoM/dt per layer thickness
Plus:

Python binding → side-by-side demo: same optimization loop, grcwa vs cpprcwa, expected >30× per iteration on the quasi-1D case (fast path + free adjoint).
Flagship example: topology-optimize the TaN absorber (§7.2 stack) to maximize image contrast at 13.5 nm — an actual EUV mask co-optimization demo, which no open C++ RCWA offers.
The natural follow-on it unlocks (mention as Phase 2)
Once gradients exist, the second genuinely significant module becomes obvious: an EUV aerial-image (Abbe) layer — loop over source points/incidence angles (embarrassingly parallel, each ~0.4 s in quasi-1D mode), collect diffraction orders into the pupil, form the partially coherent image. Combined with the adjoint, you get source–mask optimization — the actual industrial workflow your EUV examples point at. The byorder per-order flux (currently a stub, §8) becomes a prerequisite and finally earns its place.

Effort & risk
Effort: moderate — ~90% reuses existing, validated components (SolveInterior, Solve_FieldFourier, matrix_zintegral_blocks, get_ifft, S-matrix memoization). The new math is one derivation (adjoint source ↔ FoM Jacobian, unconjugated z-integral), the kind of thing §7.7 shows this project does well.
Risks: sign/normalization conventions in the reciprocity relation (mitigated: golden-gradient tests catch any global factor immediately); complex-ω/Qabs interaction with reciprocity (restrict v1 to real ω, document); FoMs defined on interior fields need a volume adjoint source (defer to v2 — reflected-order FoMs cover the EUV use cases).
Bottom line: every remaining item in §8 makes the solver more complete; adjoint gradients make it a different class of tool — the thing the original library was built for, at 30× the speed, with a free golden-reference validation path. That's the significant move.

09.08.2026 21:04

I need more performance, especially in case of quasi 2d geometry(say x is big, y size is several wavelenghts)


09.08.2026 21:05
Quasi-2D Performance Plan (Lx ≫ λ, Ly ~ few λ, pattern varies in both x and y)
First, understand why quasi-2D is the brutal regime
In quasi-1D you got away with nG ≈ 113. In quasi-2D the harmonic set is a product: you still need Mx ≈ hundreds of x-orders (Lx = 3.5 µm supports ±259 propagating orders alone), and now My ≈ 5–15 y-orders that actually couple. So:

n 
G
​
 =M 
x
​
 ⋅M 
y
​
 ≈300×9≈2700,N=2n 
G
​
 ≈5400.
Everything cubic explodes. Do the flop accounting for your EUV stack (81 uniform interfaces + 1 grid layer) at N = 5400:

cost item	flops	note
one zgemm (N³)	~1.3e12	
one Redheffer step (~6 gemms + LU)	~1e13	
81 uniform interfaces	~8e14	← dominates: hours
one zgeev(N) (~25N³)	~3e13	minutes
The Mo/Si multilayer suffix, not the eigensolve, is the wall in quasi-2D. Your existing quasi-1D diagonal-core trick doesn't fire because it requires ky = 0 for all harmonics. That's the key thing to fix — and it turns out the restriction is unnecessary.

1. The big one: generalize the uniform-core fast path to any harmonic set (per-harmonic 2×2 blocks) — exact, kills the dominant cost
Your quasi-1D fast path exploits that uniform-layer kp/q/φ are diagonal when all ky=0. But look at the general uniform layer: harmonic i never couples to harmonic j ≠ i — only its own x/y polarization components mix:

kp 
(i)
 =ω 
2
 I 
2
​
 − 
ε
1
​
 [ 
k 
y,i
2
​
 
−k 
x,i
​
 k 
y,i
​
 
​
  
−k 
y,i
​
 k 
x,i
​
 
k 
x,i
2
​
 
​
 ],ϕ=I,q 
i
​
  doubled.
Under the permutation that interleaves components as (x 
i
​
 ,y 
i
​
 ) pairs, every matrix in a uniform–uniform Redheffer step — Q=I, P=diag(q 
ℓ
​
 )kp 
ℓ
−1
​
 kp 
ℓ+1
​
 diag(1/q 
ℓ+1
​
 ), T 
11
​
 ,T 
12
​
 , and hence the S-matrix of any all-uniform range — is block-diagonal with 2×2 blocks per harmonic, for arbitrary (k 
x,i
​
 ,k 
y,i
​
 ). No quasi-1D assumption needed.

Consequences, all exact:

Uniform interface cost: O(N³) → O(N) (a per-harmonic 2×2 Redheffer recursion — the direct generalization of your existing scalar recursion, which is just the special case where the 2×2 is diagonal).
Combine with your existing binary exponentiation (§7.7.4, boundary-corrected): the entire 40× Mo/Si mirror at N = 5400 collapses from ~8e14 flops to ~10⁷ flops — from hours to microseconds.
The assembly with the general prefix (vac | grid | Ru) reuses your validated redheffer_cascade verbatim; with the suffix S-matrix stored as 2×2 blocks, L12·R21 is an O(N²) block-scaling, so the cascade costs a handful of gemms.
This is the same kind of derivation as §7.7.5–7.7.7 (structure the physics already has), it reuses machinery you've already validated to ~4e-15, and it removes the largest term in the quasi-2D budget. What remains is the patterned zgeev + ~3–4 general star steps for the prefix — minutes, not hours. Estimated end-to-end win at N≈5400: ~50–100× on rt_solve.

2. Symmetry-adapted basis reduction (the true generalization of your TE/TM decoupling) — exact, 4–16× on everything cubic that's left
Here's a reframing worth internalizing: your quasi-1D TE/TM decoupling is y-mirror symmetry — for a y-invariant structure the symmetry group is continuous, and the two irreps are TE/TM. A quasi-2D pattern breaks the continuous symmetry but almost always keeps the discrete mirror σ 
y
​
 : ε(x,−y)=ε(x,y) (your bar-with-y-features geometries typically do).

When σ 
y
​
  holds and the incidence plane is compatible (k 
y0
​
 =0, i.e. φ=0), the convolution matrix commutes with the parity operator that maps harmonic (i,j)→(i,−j) (with the appropriate sign on E 
y
​
 ). Change basis to even/odd combinations

∣i,j⟩ 
±
​
 = 
2
​
 
1
​
 (∣i,j⟩±∣i,−j⟩),
and M, kp, all interface matrices, and the S-matrix block-diagonalize into two sectors of size ≈ N/2. Every O(N³) operation — zgeev, the grid-interface star steps, the cascade — drops 4×. If the pattern also has x-mir
09.08.2026 21:07

More ideas


09.08.2026 21:11
For quasi-2D geometries—large L 
x
​
 , moderate L 
y
​
 , and genuine variation in both directions—the main problem is no longer FFT planning or the uniform multilayer cache. It is the dense 2n 
G
​
  patterned-layer eigendecomposition and dense S-matrix algebra.

The best ideas, in priority order, are below.

1. Extend the uniform-stack fast path to arbitrary 2D harmonics
This should be implemented first because it is exact, relatively low-risk, and broadly applicable.

Your current fast uniform suffix is described as quasi-1D-specific because k 
y
​
 =0 makes the matrices diagonal. But uniform isotropic layers never couple different diffraction orders, even when every order has a different nonzero k 
y
​
 .

For harmonic g, the uniform-layer block is

K 
p
(g)
​
 =ω 
2
 I 
2
​
 − 
ε
1
​
 [ 
k 
y,g
2
​
 
−k 
x,g
​
 k 
y,g
​
 
​
  
−k 
x,g
​
 k 
y,g
​
 
k 
x,g
2
​
 
​
 ].
After permuting the vector from

[x 
0
​
 ,…,x 
n−1
​
 ,y 
0
​
 ,…,y 
n−1
​
 ]
to

[x 
0
​
 ,y 
0
​
 ,x 
1
​
 ,y 
1
​
 ,…],
the complete uniform operator consists of independent 2×2 blocks.

Even better, rotate each harmonic to its local TE/TM basis. Every isotropic uniform interface then reduces to two scalar Fresnel recursions per harmonic. Thus the whole vacuum/Ru/Mo/Si/substrate stack costs

O(n 
G
​
 N 
layers
​
 )
instead of dense O(n 
G
3
​
 ) operations.

The resulting uniform-stack reflection operator is block diagonal in the Cartesian basis:

R 
uniform
​
 =blockdiag(R 
0
​
 ,R 
1
​
 ,…,R 
n 
G
​
 −1
​
 ),R 
g
​
 ∈C 
2×2
 ,
or scalar diagonal in TE/TM coordinates.

Why this matters
This works for:

full 2D geometry;
oblique incidence;
arbitrary rectangular or circular G sets;
complex materials;
every planar multilayer suffix.
Only the coupling between the patterned region and the uniform suffix remains dense. Do not assemble the suffix as four dense 2n 
G
​
 ×2n 
G
​
  matrices; preserve its block representation and provide operations such as:

left/right multiply by the block-diagonal operator;
apply to a vector;
add to a dense matrix;
perform blockwise inversions.
Your periodic binary exponentiation becomes much less important once all 80 uniform interfaces are already O(n 
G
​
 ).

2. Add a true anisotropic harmonic truncation
The API currently chooses harmonics from one scalar nG. That is poorly matched to L 
x
​
 ≫L 
y
​
 .

Add a tensor-product mode:

G={(m 
x
​
 ,m 
y
​
 ):−M 
x
​
 ≤m 
x
​
 ≤M 
x
​
 , −M 
y
​
 ≤m 
y
​
 ≤M 
y
​
 },
with independent user controls:

cpp
struct HarmonicTruncation {
    int Mx;
    int My;
};
For example, a quasi-2D problem may need:

M 
x
​
 =250, because the long x-period supports many propagating orders;
M 
y
​
 =3 or 5, because the geometry and output converge rapidly in y.
The resulting count is

n 
G
​
 =(2M 
x
​
 +1)(2M 
y
​
 +1).
A circular physical-k cutoff may retain more y harmonics than the geometry actually needs, or require excessive x extent before completing a shell containing the desired y orders.

Add adaptive convergence
Instead of asking for a single nG, implement:

Fix M 
x
​
 , increase M 
y
​
  until the observable converges.
Fix M 
y
​
 , increase M 
x
​
 .
Repeat if necessary.
For a requested tolerance, compare:

∣R(M 
x
​
 ,M 
y
​
 +1)−R(M 
x
​
 ,M 
y
​
 )∣
and selected diffraction amplitudes.

Because the cost is cubic today, removing even 30% of the harmonics gives approximately a 3× speedup. A factor-two harmonic reduction gives approximately 8×.

This is not automatically exact, but it is a controlled truncation, just like choosing nG.

3. Exact mirror-symmetry decomposition
Many quasi-2D mask geometries are symmetric about one or both cell axes. Exploit that before eigendecomposition.

If

ε(x,−y)=ε(x,y)
and the incident Bloch vector satisfies k 
y0
​
 =0, harmonics (m 
x
​
 ,m 
y
​
 ) and (m 
x
​
 ,−m 
y
​
 ) can be transformed into even and odd combinations:

∣m 
x
​
 ,m 
y
​
 ⟩ 
±
​
 = 
2
​
 
∣m 
x
​
 ,m 
y
​
 ⟩±∣m 
x
​
 ,−m 
y
​
 ⟩
​
 .
With the correct vector-component parity, the Maxwell operator splits into two independent sectors of approximately half size.

For a cubic operation:

2( 
2
N
​
 ) 
3
 = 
4
N 
3
 
​
 ,
so one mirror symmetry gives about a 4× eigensolver reduction.

If the geometry also satisfies

ε(−x,y)=ε(x,y)
at compatible incidence, the four parity sectors can reduce cubic work by approximately

4( 
4
N
​
 ) 
3
 = 
16
N 
3
 
​
 .
Important implementation detail
The G set must be closed under the symmetry transformation. This is easiest with the rectangular M 
x
​
 ×M 
y
​
  truncation from the previous section.

Implement symmetry as an explicit mode, not a tolerance-based automatic shortcut:

cpp
enum class Symmetry {
    None,
    MirrorX,
    MirrorY,
    MirrorXY
};
Validate every sector against the full solve before enabling automatic detection.

4. Use the primitive cell when the long x-domain contains repetition
Sometimes L 
x
​
  is large because the computational cell contains repeated motifs rather than a truly unique 3.5 µm structure.

If

ε(x+a,y)=ε(x,y),L 
x
​
 =Pa,
the supercell operator is block-circulant across the P repetitions. A discrete Bloch transform splits it into P independent primitive-cell problems.

For excitation in a single Bloch sector, only that sector is needed. The reduction can be close to:

n 
G
​
 → 
P
n 
G
​
 
​
 ,cubic cost→ 
P 
3
 
1
​
 .
This is exact if the geometry really repeats.

For a structure that is periodic except for a small number of defects, a follow-up approach is:

solve the repeated background in primitive-cell Bloch sectors;
represent defects as a low-rank or localized perturbation;
use a Woodbury/domain-decomposition correction.
That is more involved, but vastly better than treating a nearly periodic structure as a generic supercell.

5. Stop constructing the full S-matrix when only one excitation is required
RT_Solve usually has one incident plane wave, but it constructs operators describing the response to every possible incident channel.

That is unnecessary for scalar R/T or one reflected field.

The current complexity is roughly:

build all patterned eigenvectors;
form dense interface matrices;
build four full S blocks;
finally multiply by a sparse excitation vector.
Introduce a SingleRHS execution mode that computes only

a 
N
​
 , b 
0
​
 
for the supplied a 
0
​
 ,b 
N
​
 .

This alone does not eliminate zgeev, but it can eliminate a significant fraction of the S-matrix work and memory. In particular:

apply the uniform suffix as per-order 2×2 blocks;
solve only the boundary system required for the current RHS;
avoid materializing unused S blocks;
use zgetrf/zgetrs rather than inverses;
reuse the factorization for p/s excitations or multiple RHSs.
Suggested API:

cpp
enum class SolveMode {
    FullScatteringMatrix,
    SingleRHS,
    BatchedRHS
};
Keep the full mode for repeated excitations, fields in many layers, and reusable device scattering matrices.

6. The largest architectural gain: a matrix-free single-RHS backend
The dense patterned-layer eigensystem is the real scalability limit:

zgeev(2n 
G
​
 )=O(n 
G
3
​
 ),memory=O(n 
G
2
​
 ).
At n 
G
​
 =2000, no amount of phase-vector optimization will make that attractive.

For large quasi-2D cases, add a second backend that solves the Fourier-domain Maxwell boundary-value problem directly for the requested excitation, without computing every patterned-layer eigenmode.

Basic design
Collapse the entire uniform substrate/multilayer into a per-harmonic TE/TM admittance or reflection boundary condition.
Keep the incident medium as another diagonal radiation boundary condition.
Discretize only the patterned region in z:
high-order Chebyshev collocation;
spectral elements;
or a small number of finite-difference planes.
Solve the resulting linear system with GMRES or another Krylov method.
Apply ε convolution matrix-free using FFTs.
A convolution operation becomes:

v 
G
​
 ⟶IFFT(v 
G
​
 )⟶ε(x,y)v(x,y)⟶FFT,
costing approximately

O(N 
x
​
 N 
y
​
 log(N 
x
​
 N 
y
​
 ))
instead of a dense O(n 
G
2
​
 ) multiplication.

The memory drops from O(n 
G
2
​
 ) to approximately O(n 
G
​
 N 
z
​
 ).

Why quasi-2D is especially suitable
Use the y-averaged or y-invariant structure as a preconditioner:

ε(x,y)= 
ε
ˉ
 (x)+Δε(x,y).
The preconditioner based on  
ε
ˉ
 (x) decomposes into the fast quasi-1D solves you already have. The Krylov iteration then handles only the genuinely 2D correction Δε.

This can be very effective when:

the main geometry is a long x-directed line;
y modulation consists of line ends, assist features, or moderate perturbations;
M 
x
​
  is large but M 
y
​
  is small.
This is the path that changes scaling from dense cubic to approximately

O(N 
iter
​
 N 
z
​
 n 
G
​
 logn 
G
​
 ).
It is the most significant long-term performance improvement.

Trade-off
This is a new solver backend rather than a small RCWA optimization. It should coexist with the existing modal backend:

dense modal RCWA for small and medium n 
G
​
 ;
matrix-free single-RHS solver for large quasi-2D cases.
7. Use the quasi-1D operator as a block preconditioner
Even if you retain the modal formulation, the quasi-2D operator has a useful decomposition by y order.

Order harmonics by m 
y
​
 :

v=[v 
−M 
y
​
 
​
 ,…,v 
0
​
 ,…,v 
M 
y
​
 
​
 ],
where each block contains all x harmonics and both field components.

Ignoring coupling between different y orders gives a block-diagonal operator:

P=blockdiag(A 
−M 
y
​
 ,−M 
y
​
 
​
 ,…,A 
M 
y
​
 ,M 
y
​
 
​
 ).
Each diagonal block is essentially a quasi-1D problem and can use your optimized backend.

Use P 
−1
  as the preconditioner for:

the matrix-free boundary solve;
iterative interface solves;
shift-invert Arnoldi, if selected eigenmodes are required.
If the y dependence is weak or has limited Fourier bandwidth, the off-diagonal block coupling is small and convergence should be fast.

This directly reuses the strongest part of the current implementation.

8. Truncate coupling bandwidth in y, not the field orders
For many quasi-2D patterns, the field may require several y orders, but the material Fourier spectrum may be concentrated near small Δm 
y
​
 .

The convolution blocks satisfy

C 
m 
y
​
 ,n 
y
​
 
​
 = 
ε
  
m 
y
​
 −n 
y
​
 
​
 (m 
x
​
 −n 
x
​
 ).
If

∣ 
ε
  
Δm 
y
​
 
​
 ∣
decays rapidly with ∣Δm 
y
​
 ∣, retain all field orders but drop material coupling blocks below a tolerance:

∥C 
Δm 
y
​
 
​
 ∥<τ∥C 
0
​
 ∥.
Then the operator becomes block-banded in m 
y
​
 .

Benefits include:

faster matrix-vector products;
better block preconditioners;
possible block-banded direct solves;
reduced memory.
This is approximate, but the error is measurable from the discarded Fourier coefficients and can be exposed through a user tolerance.

It is generally safer than deleting the associated field orders entirely.

9. Compress transmission blocks through thick patterned layers
For a patterned layer of nontrivial thickness, modes with large Imq have factors

e 
iqt
 ,∣e 
iqt
 ∣=e 
−Im(q)t
 .
Strongly evanescent modes affect reflection near the front interface but contribute negligibly to transmission across the layer. Therefore:

reflection blocks can remain dense;
transmission blocks S 
12
​
  and S 
21
​
  may be numerically low rank.
Represent them as:

S 
12
​
 ≈U 
12
​
 V 
12
†
​
 ,S 
21
​
 ≈U 
21
​
 V 
21
†
​
 .
Then use Woodbury identities in the Redheffer cascade instead of dense factorizations.

A physical rank criterion is:

e 
−Im(q 
j
​
 )t
 >τ.
Keep propagating and weakly evanescent channels, while compressing the rest.

This will not help a very thin patterned layer as much, but for a 60 nm absorber at 13.5 nm it is worth measuring the singular values of the transmission blocks. Do that benchmark before implementing the full low-rank cascade.

10. Mixed precision for dense setup, with double refinement
For the remaining dense path:

Build and factor interface systems in complex<float>.
Compute the solution in single precision.
Evaluate residuals in double precision.
Apply double-precision iterative refinement.
For well-conditioned systems this approaches double-precision accuracy while moving half the memory and using faster cgemm/cgetrf.

A conservative first target is the Redheffer/interface solves—not zgeev, where eigenvector accuracy can be more delicate.

This is especially useful once matrices no longer fit comfortably in cache.

11. Distributed dense eigensolver only as a fallback
If full eigenvectors for arbitrary 2D patterned layers are mandatory, then the workload is fundamentally dense and cubic. At that point, use software designed for it:

Intel oneMKL for better threaded LAPACK;
ELPA;
SLATE;
ScaLAPACK;
vendor multicore eigensolvers.
This will not fix scaling, but it can make n 
G
​
 ∼1000−3000 practical on a large-memory workstation or cluster.

Do this after algorithmic reductions. Parallelizing a matrix four times larger is much less valuable than avoiding it.

Recommended implementation order
Priority	Change	Exact?	Expected importance
1	Per-harmonic TE/TM uniform-stack backend for arbitrary 2D G	Yes	Very high
2	Independent M 
x
​
 ,M 
y
​
  truncation and convergence driver	Controlled	Very high
3	Mirror-parity decomposition	Yes when applicable	Up to 4–16× cubic reduction
4	Single-RHS solve without full S materialization	Yes	High
5	Primitive-cell/Bloch decomposition	Yes when applicable	Potentially enormous
6	Matrix-free patterned-region backend	Convergent numerical solve	Changes cubic scaling
7	Quasi-1D block preconditioner	Yes up to Krylov tolerance	Critical for item 6
8	y-coupling bandwidth truncation	Approximate	Geometry-dependent
9	Low-rank evanescent transmission	Approximate	Thickness-dependent
10	Mixed precision/refinement	Usually near-double	Moderate
My strongest recommendation
Implement two tracks:

Immediate track
Preserve every uniform stack as per-harmonic TE/TM reflection/admittance data.
Add rectangular (M 
x
​
 ,M 
y
​
 ) truncation.
Add parity decomposition.
Avoid constructing a full dense S-matrix for one incident wave.
These should substantially improve the existing solver without changing its mathematical model.

Strategic track
Build a matrix-free single-excitation Fourier-Maxwell backend, preconditioned by your quasi-1D solver. That is the only proposal here that removes both:

the full 2n 
G
​
 ×2n 
G
​
  zgeev;
the dense cubic scattering algebra.
For genuinely large quasi-2D geometries, this is likely the difference between incremental speedups and a solver that scales to the required harmonic counts.

09.08.2026 21:13

Need a maximum perfomance for my quasi 1d case (size x is big, y is several wavelenghts), patterned layer can be arbitrary, illumination can be arbitrary. Need maxmium perfomance here


09.08.2026 21:18
Maximum-performance strategy for your geometry
Your case is not mathematically quasi-1D if the patterned layer is arbitrary in both x and y. It is an aspect-ratio-anisotropic 2D problem:

L 
x
​
 ≫L 
y
​
 ∼a few λ,n 
x
​
 ≫n 
y
​
 ,n 
G
​
 =n 
x
​
 n 
y
​
 .
Because the pattern and illumination are arbitrary, you cannot rely on:

y-invariance;
TE/TM decoupling;
mirror symmetry;
low-rank material coupling;
normal incidence.
The current dense RCWA architecture will eventually hit a hard wall:

time=O((2n 
G
​
 ) 
3
 ),memory=O((2n 
G
​
 ) 
2
 ).
For example, n 
x
​
 =501, n 
y
​
 =7 gives n 
G
​
 =3507 and a 7014×7014 eigensystem. One complex matrix alone is about 750 MiB; the complete modal calculation requires many such matrices.

For maximum performance, the right solution is not another optimization of zgeev. It is a specialized matrix-free, single/batched-excitation backend, preconditioned by your existing quasi-1D solver.

1. Immediate exact improvement: collapse every planar stack per harmonic
Before changing the patterned-layer solver, represent every all-uniform region as independent per-harmonic TE/TM ports.

For arbitrary (k 
x,g
​
 ,k 
y,g
​
 ), uniform isotropic layers never couple different G orders. Rotate each order into its local TE/TM basis and evaluate the complete multilayer using scalar 2×2 scattering recursion.

Store only something like:

cpp
struct PlanarPort {
    ComplexVector r_te;
    ComplexVector r_tm;
    ComplexVector t_te;
    ComplexVector t_tm;
    ComplexVector y_te;
    ComplexVector y_tm;
};
This is valid for:

arbitrary incidence;
nonzero k 
y
​
 ;
propagating and evanescent orders;
complex materials;
arbitrary rectangular harmonic sets.
Cost becomes:

O(n 
G
​
 N 
layers
​
 )
and storage becomes O(n 
G
​
 ).

Do not expand the planar stack back into dense 2n 
G
​
 ×2n 
G
​
  matrices. Implement direct operations in Cartesian or TE/TM coordinates.

This eliminates the multilayer stack from the performance problem entirely.

2. Use explicit rectangular harmonic truncation
Do not use scalar nG and circular truncation for this backend. Introduce independent harmonic counts:

cpp
struct RectangularHarmonics {
    int Mx;
    int My;
};

// mx = -Mx ... Mx
// my = -My ... My
Then

n 
x
​
 =2M 
x
​
 +1,n 
y
​
 =2M 
y
​
 +1,n 
G
​
 =n 
x
​
 n 
y
​
 .
This is essential because your geometry needs many x orders and relatively few y orders.

Use storage ordered by y blocks:

[(m 
y
​
 =−M 
y
​
 , all m 
x
​
 ),…,(m 
y
​
 =M 
y
​
 , all m 
x
​
 )].
That makes the operator naturally a small n 
y
​
 ×n 
y
​
  block system, where each block contains n 
x
​
  harmonics.

Also add independent convergence control:

Increase M 
x
​
  until the required output converges.
Increase M 
y
​
 .
Repeat until both directional errors satisfy the tolerance.
This alone may avoid several times more harmonics than a generic circular selection.

3. Remove dense convolution matrices
The present implementation explicitly constructs:

C 
ij
​
 = 
ε
^
 (G 
i
​
 −G 
j
​
 )
and its inverse. For large n 
G
​
 , both construction and storage are undesirable.

Apply C matrix-free:

v 
G
​
  
IFFT
​
 v(x,y) 
×ε(x,y)
​
 ε(x,y)v(x,y) 
FFT
​
 (Cv) 
G
​
 .
This costs approximately:

O(N 
x
​
 N 
y
​
 log(N 
x
​
 N 
y
​
 ))
instead of O(n 
G
2
​
 ).

Because your harmonic set is rectangular, no irregular gather/scatter is required except padding and cropping. Use batched transforms for the field components.

Applying C 
−1
 
Do not form epsinv. Solve

Cz=b
iteratively to the required tolerance, preconditioned by multiplication with 1/ε(x,y) in real space.

However, nested iterative solves should be avoided in the final Maxwell operator. Prefer a formulation that keeps D and E as separate unknowns, or augment the system with

D=εE,
so applications of C 
−1
  are replaced by applications of C. This is important for maximum performance.

4. Replace patterned-layer zgeev with a direct boundary-value solve
The current modal route computes every eigenmode even though one illumination only requires one electromagnetic solution.

Instead, solve Maxwell’s equations directly in the patterned layer using:

Fourier representation in x,y;
high-order spectral discretization in z;
transparent boundary operators from the planar stacks;
matrix-free GMRES/FGMRES.
Use tangential fields, for example

Ψ(z)=[E 
x
​
 ,E 
y
​
 ,H 
x
​
 ,H 
y
​
 ] 
T
 .
The patterned layer is constant in z, but a direct boundary formulation avoids computing and storing all 4n 
G
​
  eigenvectors.

Recommended z discretization
For one homogeneous-in-z patterned layer, use either:

Chebyshev collocation
Approximately 8–20 z nodes will often be enough for high accuracy.

High-order spectral elements
Preferable if there are multiple patterned layers or material interfaces in z. Use one element per physical layer and polynomial degree around 4–10.

The complete linear system has size approximately:

4n 
G
​
 N 
z
​
 ,
but it is never assembled. A matrix-vector product consists mainly of:

diagonal multiplication by k 
x
​
 ,k 
y
​
 ;
batched FFT/IFFT;
pointwise multiplication by ε(x,y);
small z-derivative operations;
per-order planar-port boundary operations.
Memory becomes roughly:

O(n 
G
​
 N 
z
​
 )
instead of O(n 
G
2
​
 ).

The solve returns every reflected and transmitted diffraction order for the supplied arbitrary illumination.

5. The critical preconditioner: independent quasi-1D solves per y order
This is where your existing optimized quasi-1D implementation becomes extremely valuable.

Order the unknowns by m 
y
​
 . Ignoring material coupling between different y harmonics gives:

P=blockdiag(A 
−M 
y
​
 
​
 ,A 
−M 
y
​
 +1
​
 ,…,A 
M 
y
​
 
​
 ).
Each A 
m 
y
​
 
​
  is a quasi-1D x problem with constant

k 
y
​
 =k 
y0
​
 + 
L 
y
​
 
2πm 
y
​
 
​
 .
Your existing oblique quasi-1D machinery already solves exactly this type of problem.

Therefore, construct a preconditioner containing one optimized x-only solve for each y order:

P 
−1
 r= 
​
  
A 
−M 
y
​
 
−1
​
 r 
−M 
y
​
 
​
 
⋮
A 
M 
y
​
 
−1
​
 r 
M 
y
​
 
​
 
​
  
​
 .
The n 
y
​
  blocks are independent and can be executed in parallel.

This preconditioner captures:

the large x dimension;
x diffraction;
arbitrary k 
y
​
  for each channel;
the complete planar multilayer;
propagating and evanescent behavior in z.
GMRES only has to correct coupling between different y orders caused by the arbitrary 2D pattern.

That is exactly the right decomposition for n 
x
​
 ≫n 
y
​
 .

6. Strengthen the preconditioner when y coupling is large
For a strongly arbitrary y pattern, the block-diagonal preconditioner may require too many iterations. Add progressively stronger variants.

Level 0: y-averaged preconditioner
Use only  
ε
^
  
Δm 
y
​
 =0
​
 . Cheapest option.

Level 1: independent y-channel blocks
Include each channel’s actual k 
y
​
 , but no inter-channel material coupling.

Level 2: nearest-neighbor y coupling
Retain:

Δm 
y
​
 ∈{−1,0,+1}.
This produces a block-tridiagonal preconditioner in y. Because n 
y
​
  is small, solve it using block cyclic reduction or block Thomas elimination, with each x block handled by the quasi-1D backend.

Level b: y-band preconditioner
Retain:

∣Δm 
y
​
 ∣≤b.
Choose b from the material Fourier spectrum:

∑ 
Δm 
y
​
 
​
 ∥ 
ε
^
  
Δm 
y
​
 
​
 ∥ 
2
 
∑ 
∣Δm 
y
​
 ∣>b
​
 ∥ 
ε
^
  
Δm 
y
​
 
​
 ∥ 
2
 
​
 <τ 
P
​
 .
The full operator remains exact; only the preconditioner is truncated. Therefore, the final answer is controlled by the GMRES residual, not by the preconditioner approximation.

Use FGMRES so that the preconditioner can vary or use inner iterative solves.

7. Solve only for requested illuminations
“Arbitrary illumination” does not require constructing the complete scattering matrix.

For an arbitrary illumination vector f, solve:

Au=f.
The solution contains the response in all output diffraction orders. This is dramatically cheaper than constructing the response to every possible incident order.

Provide three execution modes:

cpp
enum class SolveMode {
    SingleRHS,
    BatchedRHS,
    FullScatteringMatrix
};
Single RHS
Best for one plane wave or one coherent superposition.

Batched RHS
Use block GMRES for several illuminations, such as:

p and s polarization;
multiple source angles;
multiple incident diffraction orders.
Block FFTs and block Krylov operations improve hardware utilization.

Many sequential illuminations
Use Krylov recycling such as GCRO-DR. The operator and preconditioner stay unchanged while only the RHS changes.

If a truly complete scattering matrix is needed, apply the solver to blocks of incident channels. But do not make that the default path.

8. This architecture makes GPU acceleration worthwhile
The existing GPU feasibility result was negative because the workload depended on:

non-Hermitian zgeev;
medium-sized dense zgemm;
CPU/GPU transfer overhead.
The matrix-free backend has a completely different workload:

batched 2D FFTs;
pointwise complex multiplication;
diagonal k 
x
​
 /k 
y
​
  operations;
vector reductions;
block Krylov operations.
These are excellent GPU workloads.

Keep all Krylov vectors resident on the GPU. Transfer only:

material grids during setup;
illumination vectors;
final reflected/transmitted coefficients.
Use:

cuFFT or rocFFT;
cuBLAS for block vector operations;
custom fused kernels for Maxwell operator application;
CUDA graphs if repeated operator applications have a fixed sequence.
For rectangular harmonics, pad dimensions to FFT-friendly sizes. For example, a requested 501×7 harmonic set may use a 1024×16 transform grid if aliasing requirements permit.

Unlike the old dense backend, this GPU route can plausibly produce an order-of-magnitude gain.

9. Fuse the matrix-free Maxwell operator
Do not implement the operator as many temporary Eigen vectors.

A single operator application should:

Apply z differentiation.
Multiply Fourier coefficients by k 
x
​
 ,k 
y
​
 .
Pack all needed field components into one batched inverse FFT.
Multiply by material data in real space.
Perform one batched forward FFT.
Crop the retained harmonics.
Add diagonal/curl/boundary terms.
Use one contiguous allocation:

cpp
// Conceptual layout:
[field][z_node][my][mx]
Make mx the contiguous dimension for FFT and SIMD efficiency.

Avoid:

per-component allocations;
Eigen expression temporaries;
separate FFT calls for each component;
repeated harmonic gather/scatter;
explicit diagonal matrices.
The operator should have zero dynamic allocation after setup.

10. Deflate difficult propagating and near-cutoff channels
Krylov convergence may slow down near:

Wood anomalies;
grazing diffraction orders;
guided resonances;
high-Q layer modes.
Create a small coarse space containing:

all propagating port channels;
near-cutoff orders satisfying ∣q 
g
​
 ∣<q 
threshold
​
 ;
a few Ritz vectors retained from previous solves.
Use deflated GMRES or GCRO-DR. The number of such channels is normally much smaller than the total harmonic count.

For angle or wavelength scans, recycle the coarse space between nearby points.

11. Frequency and illumination batching
Same geometry and frequency, many illuminations
Reuse:

FFT plans;
material spectra;
planar-port operators;
preconditioner factorizations;
Krylov deflation vectors.
Use block GMRES.

Wavelength or angle scan
Operators change, but nearby points are strongly related. Reuse:

previous solution as the initial guess;
previous Ritz/deflation vectors;
preconditioner structure;
FFT plans and geometry data.
Parallelize different scan points only after making the single-point solve efficient. Excessive concurrent BLAS/FFT threading can otherwise reduce performance.

12. Keep the dense backend as a reference
The matrix-free backend will converge to a numerical tolerance rather than reproduce the dense path bit-for-bit.

Retain the current solver for:

small harmonic sets;
regression testing;
complete modal information;
Volume_integral formulas based on modal q;
validation of the new backend.
For every new benchmark, compare:

scalar R/T;
every reflected/transmitted order;
Fourier fields at selected z values;
energy conservation for lossless cases;
residual norm.
A useful backend selector is:

cpp
enum class Backend {
    DenseModal,
    MatrixFreeAspectRatio,
    Auto
};
Auto can estimate dense memory and flop cost and switch when n 
G
​
  exceeds a threshold.

Expected scaling
Suppose:

n 
x
​
 =501,n 
y
​
 =7,n 
G
​
 =3507.
Existing dense modal backend
Eigensystem dimension: 7014.
One dense complex matrix: approximately 750 MiB.
Multiple-gigabyte working set.
Cubic eigensolver and interface algebra.
Likely tens of seconds to minutes, depending on hardware.
Matrix-free backend
With N 
z
​
 =12:

Unknown count: approximately 4×3507×12≈168,000.
No dense 3507 
2
  or 7014 
2
  matrices.
Each iteration: a small number of batched FFTs on a padded rectangular grid.
Preconditioner: seven independent quasi-1D solves.
Memory: likely hundreds of MiB or less, depending on Krylov restart length.
GPU-friendly.
Actual speed depends primarily on the preconditioned iteration count:

10–30 iterations: excellent;
30–80: still likely much faster than dense modal RCWA;
hundreds: strengthen the y-band preconditioner and deflation.
Recommended implementation sequence
Phase 1: fastest improvement inside current RCWA
Add rectangular (M 
x
​
 ,M 
y
​
 ) harmonics.
Collapse all uniform stacks into per-order TE/TM port operators.
Add a single-RHS path.
Benchmark the remaining patterned zgeev separately.
This is exact and should be implemented regardless of the next phase.

Phase 2: matrix-free prototype
Implement FFT-based Cv.
Implement the tangential Maxwell operator.
Add spectral discretization in z.
Add top/bottom planar-port boundary conditions.
Solve with CPU FGMRES.
Use the y-diagonal quasi-1D preconditioner.
Validate against dense RCWA at moderate sizes.
Phase 3: maximum-performance implementation
Add y-band preconditioning.
Add deflation and Krylov recycling.
Fuse operator kernels.
Port FFT/operator/Krylov vectors to GPU.
Add block-RHS solves.
Add automatic backend and preconditioner selection.
Bottom line
For an arbitrary 2D patterned layer and arbitrary illumination, there is no exact TE/TM, symmetry, or y-invariance shortcut. The maximum-performance architecture is:

rectangular Fourier basis+matrix-free Maxwell solve+planar TE/TM port boundaries+quasi-1D per-y-channel preconditioner
​
 
This directly exploits your only guaranteed structural advantage:

n 
x
​
 ≫n 
y
​
 .
It removes the two fundamental bottlenecks of the current implementation:

full 2n 
G
​
 ×2n 
G
​
  non-Hermitian eigendecomposition;
dense O(n 
G
2
​
 ) storage and O(n 
G
3
​
 ) scattering algebra.
If maximum performance is the actual objective, this specialized backend is substantially more important than further tuning of zgeev, OpenBLAS, or the existing Redheffer implementation.
