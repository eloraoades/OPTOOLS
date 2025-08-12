from matplotlib.colors import TwoSlopeNorm
from PYTHON.simulator_functions.generator import surface_from_Z1kxky_uniform_phase
from PYTHON.simulator_functions.spectrum_wrappers import def_spectrum_for_surface
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../PYTHON'))


# ---------- Utilities ----------
def hs_from_spectrum_2d(Z2D: np.ndarray, dkx: float, dky: float) -> float:
    """Hs from 2D variance spectrum (single-sided): Hs = 4 * sqrt(∬ Z dkx dky)."""
    variance = float(np.sum(Z2D) * dkx * dky)
    return 4.0 * np.sqrt(variance)


def hs_from_field(eta: np.ndarray) -> float:
    """Hs from a spatial realization: Hs = 4 * sqrt(var(eta))."""
    return 4.0 * np.sqrt(float(np.var(eta)))


def run_case(
    typeSpec: str = "Gaussian",
    x_grid_points: int = 1024,
    y_grid_points: int = 1024,
    x_spacing: float = 4.0,
    y_spacing: float = 4.0,
    mean_direction_degrees: float = 60.0,
    local_depth: float = 100.0,
    wave_period: float = 10.0,
    significant_wave_height: float = 4.0,
    sk_theta = None,
    sk_k = None,
    seed: int = 10,
    do_plot: bool = True
):
    """
    Build spectrum, synthesize a surface, and compare Hs from spectrum vs field.
    Returns a dict of diagnostics.
    """
    # If Gaussian and spreads not provided, set relative spreads based on deep-water k
    if (sk_theta is None) or (sk_k is None):
        wavelength = 9.81 / np.pi * (wave_period ** 2)  # deep-water approx
        k0 = 2.0 * np.pi / wavelength                   # [rad/m]
        sx = 0.1 * k0
        sk_theta = sx if sk_theta is None else sk_theta
        sk_k = sx if sk_k is None else sk_k

    # Build spectrum on FFT grid
    Z2D, kX, kY, dkx, dky = def_spectrum_for_surface(
        nx=x_grid_points, ny=y_grid_points,
        dx=x_spacing, dy=y_spacing,
        theta_m=mean_direction_degrees, h=local_depth,
        T0=wave_period, Hs=significant_wave_height,
        sk_theta=sk_theta, sk_k=sk_k,
        typeSpec=typeSpec, verbose=False
    )

    # Synthesize a realization (uniform phase path)
    S2D_r, S2D_i, X, Y, *_ = surface_from_Z1kxky_uniform_phase(
        Z2D, kX, kY, seed
    )

    # Diagnostics: Hs from spectrum vs field
    hs_spec = hs_from_spectrum_2d(Z2D, dkx, dky)
    hs_field = hs_from_field(S2D_r)
    sigma_field = hs_field / 4.0
    print(f"σ = {sigma_field:.3f} m,  max|ζ| = {np.max(np.abs(S2D_r)):.3f} m,  max/σ = {np.max(np.abs(S2D_r))/sigma_field:.2f}")


    if do_plot:
        sigma = hs_spec / 4.0         # or hs_field/4.0
        kappa = 3.0                   # show ±3σ for both plots (adjust to taste)
        vmax = kappa * sigma

        # -------- Surface --------
        plt.figure(figsize=(6, 5))
        #vmax = np.max(np.abs(S2D_r))
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        cmap = plt.cm.seismic  # red-blue colormap with white at 0
        ax = plt.gca()
        im0 = ax.pcolormesh(
            X/1000.0, Y/1000.0, S2D_r,
            cmap=cmap, norm=norm, shading='auto'
        )
        plt.colorbar(im0, ax=ax, label="ζ [m]")
        ax.set_title(f"Surface ({typeSpec})")
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(X.min()/1000.0, X.max()/1000.0)
        ax.set_ylim(Y.min()/1000.0, Y.max()/1000.0)
        plt.show()

        # -------- Envelope --------
        A = np.sqrt(S2D_r**2 + S2D_i**2)  # |analytic| envelope (non-negative)
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im1 = ax.pcolormesh(X/1000.0, Y/1000.0, A, 
                            cmap=plt.cm.cividis, norm=plt.Normalize(0, A.max()), shading='auto')
        plt.colorbar(im1, ax=ax, label="|ζ| [m]")
        ax.set_title("Surface envelope")
        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(X.min()/1000.0, X.max()/1000.0)
        ax.set_ylim(Y.min()/1000.0, Y.max()/1000.0)
        plt.tight_layout(); plt.show()

    return {
        "Spectrum": typeSpec,
        "Grid (nx,ny)": f"{x_grid_points}×{y_grid_points}",
        "Spacing (dx,dy) [m]": f"{x_spacing},{y_spacing}",
        "Hs(spectrum) [m]": hs_spec,
        "Hs(field) [m]": hs_field,
        "Ratio field/spec": (hs_field / hs_spec) if hs_spec > 0 else np.nan,
        # ---- arrays to reuse for spectral plots
        "Z2D": Z2D, "kX": kX, "kY": kY, "dkx": dkx, "dky": dky,
        "X": X, "Y": Y, "S2D_r": S2D_r, "S2D_i": S2D_i,
    }