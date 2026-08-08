#!/usr/bin/env python3
"""Plot the absolute reflected-field magnitude |E| for the quasi-1D EUV
line-grating example (ex_quasi1d_absorber --field OUT).

Reconstructs the reflected electric field in the incident medium (air, z=0)
from the per-order Fourier coefficients (OUT_orders.txt) and plots the
absolute field value |E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2):

  - panel 1: |E| along x through the cell center (y = Ly/2), cpprcwa vs grcwa,
             with the TaN absorber bar marked and |E|^2 on a twin axis;
  - panel 2: 2D |E| map over the cell (x × y), showing the y-uniform
             (quasi-1D) nature of the line-grating field.

The reconstruction convention is E(x,y) = sum_m E_m exp(i kx_m x + i ky_m y),
kx_m = 2π·gx/Lx, ky_m = 2π·gy/Ly (matches the C++ get_ifft). As a sanity check
the script also compares against the cpprcwa real-space |E|^2 grid
(OUT_grid.txt).

Usage:
    python3 scripts/plot_quasi1d_field.py OUT [nG] [abs_t] [nb] [nly] [lambda] [Nx]

Output:
    - OUT_absE.png   (|E| cross-section + 2D |E| map)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grcwa_orig"))
_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
for cand in (os.path.join(_repo, "..", "grcwa_orig"),
             os.path.join(os.path.dirname(_repo), "grcwa_orig")):
    if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "grcwa")):
        sys.path.insert(0, cand)
        break


def eps_from_n(n):
    return n * n


def build_grcwa(nG_req, abs_t, nb, lam, Lx, Ly, wx, Nx, Ny):
    n_mo, n_si, n_ru, n_tan = (0.9226 + 0.0064j, 0.9997 + 0.0018j,
                               0.9114 + 0.0171j, 0.9562 + 0.0323j)
    epgrid = np.ones((Nx, Ny), dtype=complex)
    u = np.linspace(0, 1, Nx)
    half = wx / (2 * Lx)
    for i in range(Nx):
        if abs(u[i] - 0.5) < half:
            epgrid[i, :] = n_tan

    import grcwa  # noqa: E402
    obj = grcwa.obj(nG_req, [Lx, 0], [0, Ly], 1.0 / lam, 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(1.0, eps_from_n(1.0 + 0j))
    obj.Add_LayerGrid(abs_t, Nx, Ny)
    obj.Add_LayerUniform(2.5, eps_from_n(n_ru))
    for _ in range(nb):
        obj.Add_LayerUniform(2.8, eps_from_n(n_mo))
        obj.Add_LayerUniform(4.2, eps_from_n(n_si))
    obj.Add_LayerUniform(1.0, eps_from_n(n_si))
    obj.Init_Setup()
    obj.GridLayer_geteps(epgrid.flatten())
    obj.MakeExcitationPlanewave(1, 0, 0, 0, order=0)
    return obj


def grcwa_reflected_orders(obj):
    """Reflected-field Fourier coefficients (backward-only, layer 0, z=0)."""
    from grcwa.rcwa import TranslateAmplitudes
    ai, bi = obj.GetAmplitudes_noTranslate(0)
    aim, bim = TranslateAmplitudes(obj.q_list[0], obj.thickness_list[0], 0.0,
                                   np.zeros_like(ai), bi)
    fhxy = obj.phi_list[0] @ (aim + bim)
    fhx, fhy = fhxy[:obj.nG], fhxy[obj.nG:]
    tmp1 = (aim - bim) / obj.omega / obj.q_list[0]
    tmp2 = obj.phi_list[0] @ tmp1
    fexy = obj.kp_list[0] @ tmp2
    fey, fex = -fexy[:obj.nG], fexy[obj.nG:]
    fez = (obj.ky * fhx - obj.kx * fhy) / obj.omega
    fez = fez / obj.Uniform_ep_list[0]
    return obj.G, fex, fey, fez


def load_cpp_orders(path):
    data = np.loadtxt(path)
    gx = data[:, 1].astype(int)
    gy = data[:, 2].astype(int)
    ex = data[:, 3] + 1j * data[:, 4]
    ey = data[:, 5] + 1j * data[:, 6]
    ez = data[:, 7] + 1j * data[:, 8]
    return gx, gy, ex, ey, ez


def reconstruct(gx, gy, cex, cey, cez, x, y, Lx, Ly):
    """E(x,y) = sum_m c_m exp(i kx_m x + i ky_m y). Returns (Ex, Ey, Ez),
    each shaped (len(x), len(y))."""
    kx = 2 * np.pi * gx / Lx
    ky = 2 * np.pi * gy / Ly
    phase = np.exp(1j * (np.outer(x, kx)[:, None, :]
                         + np.outer(y, ky)[None, :, :]))    # (nx, ny, nG)
    Ex = np.tensordot(phase, cex, axes=([2], [0]))
    Ey = np.tensordot(phase, cey, axes=([2], [0]))
    Ez = np.tensordot(phase, cez, axes=([2], [0]))
    return Ex, Ey, Ez


def load_cpp_grid(path):
    lines = open(path).read().splitlines()
    return np.loadtxt(lines[2:])   # skip header lines


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out = sys.argv[1]
    nG_req = int(sys.argv[2]) if len(sys.argv) > 2 else 201
    abs_t = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    nly = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
    lam = float(sys.argv[6]) if len(sys.argv) > 6 else 13.5
    Nx = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
    Ny = 4

    Lx = 3500.0
    wx = 500.0
    Ly = nly * lam

    gx, gy, ex, ey, ez = load_cpp_orders(out + "_orders.txt")
    nG = len(ex)

    # Reference real-space |E|^2 grid from the C++ output (sanity check).
    cpp_grid = load_cpp_grid(out + "_grid.txt")
    cpp_absE2 = cpp_grid[:, 4].reshape(Nx, Ny)

    # Reconstruct cpprcwa reflected field on the x cross-section line.
    xs = np.linspace(0, Lx, 2001)
    Ex_c, Ey_c, Ez_c = reconstruct(gx, gy, ex, ey, ez, xs, [Ly / 2], Lx, Ly)
    absE_c = np.sqrt(np.abs(Ex_c[:, 0]) ** 2 + np.abs(Ey_c[:, 0]) ** 2
                     + np.abs(Ez_c[:, 0]) ** 2)

    # Sanity check: our reconstruction vs the C++ real-space |E|^2 grid.
    # get_ifft evaluates the field on the FFT lattice i*Lx/Nx, j*Ly/Ny (the
    # x/y labels in OUT_grid.txt use i/(Nx-1) and differ by ~1 px).
    xx = np.arange(Nx) * Lx / Nx
    yy = np.arange(Ny) * Ly / Ny
    Ex_g2, Ey_g2, Ez_g2 = reconstruct(gx, gy, ex, ey, ez, xx, yy, Lx, Ly)
    absE2_rec = np.abs(Ex_g2) ** 2 + np.abs(Ey_g2) ** 2 + np.abs(Ez_g2) ** 2
    grid_check = np.max(np.abs(absE2_rec - cpp_absE2))
    print(f"reconstruction check vs cpprcwa OUT_grid.txt |E|^2: max diff = {grid_check:.3e}")

    # ── grcwa reference (optional) ──
    G_g, ex_g, ey_g, ez_g = None, None, None, None
    try:
        obj = build_grcwa(nG_req, abs_t, nb, lam, Lx, Ly, wx, Nx, Ny)
        obj.RT_Solve(normalize=1)
        G_g, ex_g, ey_g, ez_g = grcwa_reflected_orders(obj)
        print(f"grcwa nG_req={nG_req} -> nG={obj.nG}")
    except Exception as e:
        print(f"grcwa unavailable ({e}); plotting cpprcwa only")

    # 2D |E| map over the cell on a fine grid (Ny fine to show quasi-1D).
    Ng = 513
    x2d = np.linspace(0, Lx, Ng)
    y2d = np.linspace(0, Ly, 97)
    Ex_m, Ey_m, Ez_m = reconstruct(gx, gy, ex, ey, ez, x2d, y2d, Lx, Ly)
    absE_map = np.sqrt(np.abs(Ex_m) ** 2 + np.abs(Ey_m) ** 2 + np.abs(Ez_m) ** 2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel 1: |E| cross-section along x at y = Ly/2.
    ax = axes[0]
    ax.axvspan((Lx - wx) / 2, (Lx + wx) / 2, color="0.85", zorder=0,
               label="TaN bar")
    ax.plot(xs, absE_c, "r-", lw=1.4, label="cpprcwa |E|")
    if G_g is not None:
        Ex_l, Ey_l, Ez_l = reconstruct(G_g[:, 0], G_g[:, 1], ex_g, ey_g, ez_g,
                                       xs, [Ly / 2], Lx, Ly)
        absE_g_line = np.sqrt(np.abs(Ex_l[:, 0]) ** 2 + np.abs(Ey_l[:, 0]) ** 2
                              + np.abs(Ez_l[:, 0]) ** 2)
        ax.plot(xs, absE_g_line, "b--", lw=1.0, label="grcwa |E|")
        print(f"|E| cross-section agreement: max |cpp - grcwa| = "
              f"{np.max(np.abs(absE_c - absE_g_line)):.3e}")
    ax.set_title("Reflected |E| along x (y = cell center)")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("|E|")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xs, absE_c ** 2, color="k", lw=0.8, ls=":", alpha=0.6,
             label="cpprcwa |E|²")
    ax2.set_ylabel("|E|²", color="0.3")
    ax2.tick_params(axis="y", labelcolor="0.3")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right",
              fontsize=8)

    # Panel 2: 2D |E| map over the cell. The line grating is y-invariant, so
    # the map is a band: bright open areas, dark band over the TaN bar.
    ax3 = axes[1]
    im = ax3.pcolormesh(x2d, y2d, absE_map.T, shading="gouraud", cmap="inferno")
    ax3.axvline((Lx - wx) / 2, color="w", lw=1.0, ls="--")
    ax3.axvline((Lx + wx) / 2, color="w", lw=1.0, ls="--")
    ax3.set_title("Reflected |E| over the cell\n(y-invariant line grating)")
    ax3.set_xlabel("x [nm]")
    ax3.set_ylabel("y [nm]")
    ax3.set_aspect("auto")
    fig.colorbar(im, ax=ax3, fraction=0.046, label="|E|")

    fig.suptitle(f"Quasi-1D EUV line grating — reflected |E|, nG={nG}, "
                 f"Lx={Lx:.0f} nm, Ly={Ly:.1f} nm, λ={lam:.1f} nm", fontsize=11)
    fig.tight_layout()
    png = out + "_absE.png"
    fig.savefig(png, dpi=150)
    print(f"saved {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
