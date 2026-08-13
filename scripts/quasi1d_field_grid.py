#!/usr/bin/env python3
"""Compare Solve_FieldOnGrid: grcwa (original) vs cpprcwa (nanobind).

Like quasi1d_field_fourier.py, but compares the REAL-SPACE fields on the
(Nx, Ny) grid returned by Solve_FieldOnGrid instead of the Fourier-space
coefficients. Same quasi-1D EUV line-grating geometry (TaN bar on the full
40x Mo/Si multilayer) at oblique incidence.

Because the real-space field is a physical quantity (unique regardless of the
internal eigenvector ordering/phase and of the harmonic ordering), no order
matching is needed — only the harmonic SET must match (checked by comparing
the sorted G pairs). An explicit Nxy is passed for every layer, so uniform
layers (which store no grid) are also compared.

Usage:
    python3 scripts/quasi1d_field_grid.py [nG_req] [Nx] [Ny]
        [theta_deg] [phi_deg] [abs_t] [nb] [tol]

Nx must exceed the largest |G_x| (grid FFT covers the harmonic extent); the
defaults (nG=61, Nx=80) satisfy this, but e.g. nG=101 needs Nx >= ~150.

Needs:
    - the original grcwa checkout (grcwa_orig/ next to the repo, or
      $CPPRCWA_GRCWA_ORIG)
    - a build of the cpprcwa nanobind module (build/, build-mkl/, or
      $CPPRCWA_BUILD)

Exit code 0 = all fields agree within tol, 1 = mismatch.
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)

# ── locate the original grcwa library ───────────────────────────────────────
for cand in (os.environ.get("CPPRCWA_GRCWA_ORIG"),
             os.path.join(_repo, "..", "grcwa_orig"),
             os.path.join(os.path.dirname(_repo), "grcwa_orig")):
    if cand and os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "grcwa")):
        sys.path.insert(0, cand)
        break
import grcwa  # noqa: E402

# ── locate the cpprcwa nanobind module ──────────────────────────────────────
for cand in (os.environ.get("CPPRCWA_BUILD"),
             os.path.join(_repo, "build"),
             os.path.join(_repo, "build-mkl")):
    if cand and os.path.isdir(cand):
        sys.path.insert(0, cand)
        break
import cpprcwa  # noqa: E402


def eps_from_n(n):
    return n * n


def make_stack(obj, abs_t, Nx, Ny, nb):
    """Add the quasi-1D EUV line-grating layer stack (shared by both libs):
    vacuum / TaN grid / Ru cap / nb x (Mo, Si) / Si substrate."""
    n_ru = 0.9114 + 0.0171j
    n_mo = 0.9226 + 0.0064j
    n_si = 0.9997 + 0.0018j
    obj.Add_LayerUniform(1.0, eps_from_n(1.0 + 0j))       # incident vacuum
    obj.Add_LayerGrid(abs_t, Nx, Ny)                       # TaN line grating
    obj.Add_LayerUniform(2.5, eps_from_n(n_ru))            # Ru cap
    for _ in range(nb):
        obj.Add_LayerUniform(2.8, eps_from_n(n_mo))        # Mo
        obj.Add_LayerUniform(4.2, eps_from_n(n_si))        # Si
    obj.Add_LayerUniform(1.0, eps_from_n(n_si))            # substrate


def make_epgrid(Nx, Ny, wx_nm, Lx_nm):
    epgrid = np.ones((Nx, Ny), dtype=complex)
    u = np.linspace(0, 1, Nx)
    half = wx_nm / (2 * Lx_nm)
    for i in range(Nx):
        if abs(u[i] - 0.5) < half:
            epgrid[i, :] = 0.9562 + 0.0323j   # TaN (n, not n^2 — shared convention)
    return epgrid


def main():
    nG_req = int(sys.argv[1]) if len(sys.argv) > 1 else 61
    Nx = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    Ny = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    theta_deg = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
    phi_deg = float(sys.argv[5]) if len(sys.argv) > 5 else 45.0
    abs_t = float(sys.argv[6]) if len(sys.argv) > 6 else 60.0
    nb = int(sys.argv[7]) if len(sys.argv) > 7 else 40
    tol = float(sys.argv[8]) if len(sys.argv) > 8 else 1e-8

    Lx = 3500.0
    Ly = 67.5
    wx = 500.0
    lam = 13.5
    Nxy = [Nx, Ny]

    # ── grcwa ───────────────────────────────────────────────────────────────
    g = grcwa.obj(nG_req, [Lx, 0], [0, Ly], 1.0 / lam,
                  np.radians(theta_deg), np.radians(phi_deg), verbose=0)
    make_stack(g, abs_t, Nx, Ny, nb)
    g.Init_Setup()
    g.GridLayer_geteps(make_epgrid(Nx, Ny, wx, Lx).flatten())
    g.MakeExcitationPlanewave(1, 0, 0, 0, order=0)

    # ── cpprcwa ─────────────────────────────────────────────────────────────
    c = cpprcwa.obj(nG=nG_req, L1=[Lx, 0.0], L2=[0.0, Ly],
                    freq=1.0 / lam, theta=np.radians(theta_deg),
                    phi=np.radians(phi_deg), quasi1d=False)
    make_stack(c, abs_t, Nx, Ny, nb)
    c.Init_Setup()
    c.GridLayer_geteps(make_epgrid(Nx, Ny, wx, Lx).flatten())
    c.MakeExcitationPlanewave(p_amp=1.0, p_phase=0.0, s_amp=0.0, s_phase=0.0)

    print(f"grcwa:   nG_req={nG_req} -> nG={g.nG}")
    print(f"cpprcwa: nG_req={nG_req} -> nG={c.nG}")

    # Harmonic SETS must match (ordering is irrelevant for real-space fields).
    Gg = np.asarray(g.G)
    Gc = np.asarray(c.G)
    if Gg.shape != Gc.shape or not np.array_equal(np.sort(Gg, axis=0),
                                                  np.sort(Gc, axis=0)):
        print("FATAL: harmonic sets differ between grcwa and cpprcwa")
        return 1

    deep_mo = 2 * (nb // 2) + 1         # Mo layer at index 2k+1, k = nb//2
    layers = [0, 1, 2, deep_mo, g.Layer_N - 1]   # vac, grating, Ru, deep Mo, substrate
    zs = [0.0, 0.5, 1.0]                # fractions of each layer thickness
    comps = ["ex", "ey", "ez", "hx", "hy", "hz"]

    worst = 0.0
    for l in layers:
        for z in zs:
            # grcwa returns [[ex,ey,ez],[hx,hy,hz]] directly for scalar z;
            # cpprcwa always returns a list with one entry per z.
            fg = np.array(g.Solve_FieldOnGrid(l, z, Nxy))          # (2,3,Nx,Ny)
            fc = np.array(c.Solve_FieldOnGrid(l, z, Nxy)[0])       # (2,3,Nx,Ny)
            fg = fg.reshape(6, Nx, Ny)
            fc = fc.reshape(6, Nx, Ny)
            scale = max(np.abs(fg).max(), 1e-30)
            d = np.abs(fg - fc).max() / scale
            worst = max(worst, d)
            for ci, name in enumerate(comps):
                if np.abs(fg[ci]).max() > 1e-12:
                    rel = np.abs(fg[ci] - fc[ci]).max() / np.abs(fg[ci]).max()
                else:
                    rel = np.abs(fg[ci] - fc[ci]).max()
                print(f"layer {l:2d} z={z:4.1f}  {name:2s}  rel {rel:12.3e}")
            print(f"layer {l:2d} z={z:4.1f}  max-norm  {d:12.3e}")

    print(f"\nworst relative field diff = {worst:.3e}  (tol = {tol})")
    ok = worst <= tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
