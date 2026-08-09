#!/usr/bin/env python3
"""Plot reflected |E|, |Ex|, |Ey| for the quasi-1D EUV line grating at
oblique incidence, with a grcwa reference.

Reads the per-order reflected-field coefficients written by
ex_quasi1d_absorber --field OUT --theta DEG --phi DEG --pol p|s and
reconstructs the real-space field in the incident medium (air, z=0):

    E(x,y) = sum_m E_m exp(i kx_m x + i ky_m y)

with kx_m = kx0 + 2π·gx/Lx, ky_m = ky0 + 2π·gy/Ly, where the incident
wavevector k0 = (2π/λ)(sinθ cosφ, sinθ sinφ) is the order-0 offset (matching
the C++ Lattice_SetKs convention). Three side-by-side panels along x through
the cell center (y = Ly/2): |E|, |Ex|, |Ey|.

Usage:
    python3 scripts/plot_quasi1d_oblique.py OUT theta_deg phi_deg \
        [nG] [abs_t] [nb] [nly] [lambda] [Nx] [Ny]

Output:
    OUT_absE.png
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


def build_grcwa(nG_req, abs_t, nb, lam, Lx, Ly, wx, Nx, Ny, theta_deg, phi_deg, pol):
    n_mo, n_si, n_ru, n_tan = (0.9226 + 0.0064j, 0.9997 + 0.0018j,
                               0.9114 + 0.0171j, 0.9562 + 0.0323j)
    epgrid = np.ones((Nx, Ny), dtype=complex)
    u = np.linspace(0, 1, Nx)
    half = wx / (2 * Lx)
    for i in range(Nx):
        if abs(u[i] - 0.5) < half:
            epgrid[i, :] = n_tan

    import grcwa  # noqa: E402
    obj = grcwa.obj(nG_req, [Lx, 0], [0, Ly], 1.0 / lam,
                    np.radians(theta_deg), np.radians(phi_deg), verbose=0)
    obj.Add_LayerUniform(1.0, eps_from_n(1.0 + 0j))
    obj.Add_LayerGrid(abs_t, Nx, Ny)
    obj.Add_LayerUniform(2.5, eps_from_n(n_ru))
    for _ in range(nb):
        obj.Add_LayerUniform(2.8, eps_from_n(n_mo))
        obj.Add_LayerUniform(4.2, eps_from_n(n_si))
    obj.Add_LayerUniform(1.0, eps_from_n(n_si))
    obj.Init_Setup()
    obj.GridLayer_geteps(epgrid.flatten())
    if pol == "s":
        obj.MakeExcitationPlanewave(0, 0, 1, 0, order=0)
    else:
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


def reconstruct(gx, gy, cex, cey, cez, x, y, Lx, Ly, kx0, ky0):
    """E(x,y) = sum_m c_m exp(i (kx0 + 2πgx/Lx) x + i (ky0 + 2πgy/Ly) y)."""
    kx = kx0 + 2 * np.pi * gx / Lx
    ky = ky0 + 2 * np.pi * gy / Ly
    phase = np.exp(1j * (np.outer(x, kx)[:, None, :]
                         + np.outer(y, ky)[None, :, :]))    # (nx, ny, nG)
    Ex = np.tensordot(phase, cex, axes=([2], [0]))
    Ey = np.tensordot(phase, cey, axes=([2], [0]))
    Ez = np.tensordot(phase, cez, axes=([2], [0]))
    return Ex, Ey, Ez


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    out = sys.argv[1]
    theta_deg = float(sys.argv[2])
    phi_deg = float(sys.argv[3])
    nG_req = int(sys.argv[4]) if len(sys.argv) > 4 else 201
    abs_t = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
    nb = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    nly = float(sys.argv[7]) if len(sys.argv) > 7 else 5.0
    lam = float(sys.argv[8]) if len(sys.argv) > 8 else 13.5
    Nx = int(sys.argv[9]) if len(sys.argv) > 9 else 1000
    Ny = int(sys.argv[10]) if len(sys.argv) > 10 else 4

    pol = "s" if "_s_" in os.path.basename(out) else "p"

    Lx = 3500.0
    wx = 500.0
    Ly = nly * lam

    gx, gy, ex, ey, ez = load_cpp_orders(out + "_orders.txt")
    nG = len(ex)

    # Incident wavevector offsets (order 0), matching Lattice_SetKs.
    k0 = 2 * np.pi / lam
    kx0 = k0 * np.sin(np.radians(theta_deg)) * np.cos(np.radians(phi_deg))
    ky0 = k0 * np.sin(np.radians(theta_deg)) * np.sin(np.radians(phi_deg))

    # Reconstruct reflected field on an x cross-section line at y = Ly/2.
    xs = np.linspace(0, Lx, 2001)
    Ex_c, Ey_c, Ez_c = reconstruct(gx, gy, ex, ey, ez, xs, [Ly / 2], Lx, Ly, kx0, ky0)
    Ex_c, Ey_c, Ez_c = Ex_c[:, 0], Ey_c[:, 0], Ez_c[:, 0]
    absE_c = np.sqrt(np.abs(Ex_c) ** 2 + np.abs(Ey_c) ** 2 + np.abs(Ez_c) ** 2)

    # ── grcwa reference (optional) ──
    G_g, ex_g, ey_g, ez_g = None, None, None, None
    try:
        obj = build_grcwa(nG_req, abs_t, nb, lam, Lx, Ly, wx, Nx, Ny,
                          theta_deg, phi_deg, pol)
        obj.RT_Solve(normalize=1)
        G_g, ex_g, ey_g, ez_g = grcwa_reflected_orders(obj)
        print(f"grcwa nG_req={nG_req} -> nG={obj.nG}")
    except Exception as e:
        print(f"grcwa unavailable ({e}); plotting cpprcwa only")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return 0

    # Reference grcwa lines, one per panel (|Ex|, |Ey|, |E|).
    ref = None
    if G_g is not None:
        Ex_r, Ey_r, Ez_r = reconstruct(G_g[:, 0], G_g[:, 1], ex_g, ey_g, ez_g,
                                       xs, [Ly / 2], Lx, Ly, kx0, ky0)
        Ex_r, Ey_r, Ez_r = Ex_r[:, 0], Ey_r[:, 0], Ez_r[:, 0]
        absE_r = np.sqrt(np.abs(Ex_r) ** 2 + np.abs(Ey_r) ** 2 + np.abs(Ez_r) ** 2)
        ref = [np.abs(Ex_r), np.abs(Ey_r), absE_r]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    for ax, (data, lab), ref_line in zip(
            axes,
            [(np.abs(Ex_c), "|Ex|"),
             (np.abs(Ey_c), "|Ey|"),
             (absE_c, "|E|")],
            ref if ref is not None else [None, None, None]):
        ax.axvspan((Lx - wx) / 2, (Lx + wx) / 2, color="0.85", zorder=0,
                   label="TaN bar")
        ax.plot(xs, data, "r-", lw=1.4, label="cpprcwa")
        if ref_line is not None:
            ax.plot(xs, ref_line, "b--", lw=1.0, label="grcwa")
        ax.set_title(lab)
        ax.set_xlabel("x [nm]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    if ref is not None:
        print(f"|Ex| agreement: max |cpp - grcwa| = {np.max(np.abs(np.abs(Ex_c) - ref[0])):.3e}")
        print(f"|Ey| agreement: max |cpp - grcwa| = {np.max(np.abs(np.abs(Ey_c) - ref[1])):.3e}")
        print(f"|E|  agreement: max |cpp - grcwa| = {np.max(np.abs(absE_c - ref[2])):.3e}")

    fig.suptitle(f"Quasi-1D EUV line grating — reflected field, "
                 f"{pol}-pol, θ={theta_deg}°, φ={phi_deg}°, nG={nG}, λ={lam:.1f} nm",
                 fontsize=11)
    fig.tight_layout()
    png = out + "_absE.png"
    fig.savefig(png, dpi=150)
    print(f"saved {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
