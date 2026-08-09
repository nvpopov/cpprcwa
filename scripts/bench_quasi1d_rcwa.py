#!/usr/bin/env python3
"""grcwa reference for the quasi-1D EUV absorber geometry at nG=400.

Mirrors examples/ex_quasi1d_absorber.cpp except grcwa has no quasi-1D mode,
so it builds the full 2D circular harmonic set (the exact, non-reduced solve).
"""
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
for cand in (os.path.join(_repo, "..", "grcwa_orig"),
             os.path.join(os.path.dirname(_repo), "grcwa_orig")):
    if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "grcwa")):
        sys.path.insert(0, cand)
        break
import grcwa  # noqa: E402


def eps_from_n(n):
    return n * n


def main():
    nG = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    abs_t = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    nly = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    lam = float(sys.argv[5]) if len(sys.argv) > 5 else 13.5
    Nx = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
    Ny = 4
    theta_deg = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    phi_deg = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0

    n_mo = 0.9226 + 0.0064j
    n_si = 0.9997 + 0.0018j
    n_ru = 0.9114 + 0.0171j
    n_tan = 0.9562 + 0.0323j

    Lx = 3500.0
    Ly = nly * lam

    t0 = time.perf_counter()
    th = theta_deg * np.pi / 180.0
    ph = phi_deg * np.pi / 180.0
    obj = grcwa.obj(nG, [Lx, 0], [0, Ly], 1.0 / lam, th, ph, verbose=0)
    epgrid = np.ones((Nx, Ny), dtype=complex)
    half = 500.0 / (2 * Lx)
    for i in range(Nx):
        u = i / (Nx - 1)
        if abs(u - 0.5) >= half:
            continue
        epgrid[i, :] = n_tan
    obj.Add_LayerUniform(1.0, eps_from_n(1.0 + 0j))
    obj.Add_LayerGrid(abs_t, Nx, Ny)
    obj.Add_LayerUniform(2.5, eps_from_n(n_ru))
    for _ in range(nb):
        obj.Add_LayerUniform(2.8, eps_from_n(n_mo))
        obj.Add_LayerUniform(4.2, eps_from_n(n_si))
    obj.Add_LayerUniform(1.0, eps_from_n(n_si))
    obj.Init_Setup()
    obj.GridLayer_geteps(epgrid.flatten())
    t1 = time.perf_counter()
    obj.MakeExcitationPlanewave(1, 0, 0, 0, order=0)
    R, T = obj.RT_Solve(normalize=1)
    t2 = time.perf_counter()

    print(f"grcwa: nG_req={nG}->nG={obj.nG} (full 2D) Nx={Nx} Ny={Ny} nb={nb}")
    print(f"perf_ms: setup={1000*(t1-t0):.1f} rt_solve={1000*(t2-t1):.1f} total={1000*(t2-t0):.1f}")
    print(f"R = {R:.6f}  T = {T:.6g}  R+T = {R+T:.6f}")


if __name__ == "__main__":
    main()
