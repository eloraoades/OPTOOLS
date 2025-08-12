import numpy as np
import scipy.interpolate as spi  # function griddata

from wave_physics_functions.spectra import (
    Gaussian_1Dspectrum_kx, 
    PM_spectrum_k, 
    define_Gaussian_spectrum_kxky, 
    define_spectrum_PM_cos2n
)
from wave_physics_functions.transforms import spectrum_to_kxky
from wave_physics_functions.dispersion import sig_from_k

def def_spectrumG_for_surface_1D(
        nx: int = 2048, 
        dx: float = 10,
        T0: float = 10, 
        Hs: float = 4, 
        sk_k0: float = 0.1, 
        h: float | None = None,  
        verbose: bool = False
):
    """
    Build a **1D Gaussian** variance spectrum Z1(kx) on an FFT-consistent kx grid.

    Parameters
    ----------
    nx : int
        Number of spatial samples in x.
    dx : float
        Spatial step [m].
    T0 : float
        Peak period [s] used by the Gaussian model (via wrapped helper).
        (#TODO: clarify exact role in Gaussian_1Dspectrum_kx; is it defining k0 via dispersion?)
    Hs : float
        Target significant wave height [m].
    sk_k0 : float
        Gaussian spread parameter along kx (σ in k-space) [rad/m] units implied by helper.
        (#TODO: confirm the units in Gaussian_1Dspectrum_kx.)
    h : float | None
        Water depth [m]; passed through to helper (finite-depth dispersion if used).
    verbose : bool
        Prints Hs check if True.

    Returns
    -------
    Z1 : np.ndarray
        1D variance spectrum Z1(kx) [m^2 per (rad/m)], length nx.
    kX : np.ndarray
        Wavenumber grid kx [rad/m], length nx (FFT-centered).
    sk : float
        Spread parameter returned by Gaussian_1Dspectrum_kx (units depend on helper).
        #TODO: document units of `sk` from the helper.

    Notes
    -----
    - Normalization: The raw Gaussian spectrum gets rescaled such that
          Hs_out = 4 * sqrt(∫ Z1 dkx)  matches the requested Hs.
    """
    dkx = 2*np.pi/(dx*nx)  # [rad/m]
    kX = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2*np.pi  # [rad/m]

    # --- only Gaussian -------------
    Z1_Gaussian, kX, sk = Gaussian_1Dspectrum_kx(kX, T0, sk_k0, h=h)
    # Scale to target Hs: since Hs = 4 sqrt(∫ Z1 dkx), set Z1 ← (Hs/4)^2 * Z1_raw / (∫ Z1_raw dkx)
    # NOTE: The code below multiplies by (Hs/4)**2 but does not divide by the integral of Z1_Gaussian.
    #       It implicitly assumes the helper returns a spectrum whose integral is 1.  #TODO: verify.
    Z1 = (Hs/4)**2 * Z1_Gaussian

    # Hs check (diagnostic only)
    sumZ1 = 4*np.sqrt(sum(Z1.flatten() * dkx))
    if verbose:
        print('Hs for Gaussian : ', sumZ1)

    return Z1, kX, sk


def def_spectrumPM_for_surface_1D(
    nx: int = 2048,
    dx: float = 10,
    T0: float = 10,
    h: float | None = None,
    verbose: bool = False
):
    """
    Build a **1D Pierson-Moskowitz (PM)** variance spectrum Z1(kx) on an FFT kx grid.

    Parameters
    ----------
    nx : int
        Number of samples.
    dx : float
        Spatial step [m].
    T0 : float
        Peak period [s], used to set the PM peak frequency fp = 1/T0.
    h : float | None
        Depth [m], forwarded to PM_spectrum_k (finite depth if supported).
    verbose : bool
        Print Hs check if True.

    Returns
    -------
    Z1_PM : np.ndarray
        PM variance spectrum Z1(kx) [m^2 per (rad/m)], length nx.
    kX : np.ndarray
        Wavenumber grid kx [rad/m], length nx.

    Notes
    -----
    - Hs diagnostic is printed (if verbose) as 4*sqrt(∫ Z1_PM dkx).
    - Any NaNs in PM_spectrum_k output are set to 0 before returning.
    """
    
    dkx = 2*np.pi/(dx*nx)  # [rad/m]
    kX = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2*np.pi  # [rad/m]

    # --- only PM -------------
    Z1_PM = PM_spectrum_k(kX, 1/T0, h=h)  # helper expects frequency in [Hz] (fp = 1/T0)


    # Hs diagnostic (computed before NaN cleanup but only uses finite values)
    sumZ1 = 4*np.sqrt(sum(Z1_PM[np.isfinite(Z1_PM)].flatten() * dkx))
    if verbose:
        print('Hs for Pierson-Moskowitz : ', sumZ1)  

    # Fill NaNs with zeros
    Z1_PM[np.isnan(Z1_PM)] = 0

    return Z1_PM, kX    


def def_spectrumJONSWAP_for_surface_1D(
    nx: int = 2048,
    dx: float = 10,
    T0: float = 10,
    h: float | None = None,
    gammafac: float = 3.3,
    sigA: float = 0.07,
    sigB: float = 0.09,
    verbose: bool = False
):
    """
    Build a **1D JONSWAP-shaped** variance spectrum by PM(k) × γ^{exp(…)} in frequency space.

    Parameters
    ----------
    nx, dx : int, float
        Grid size and spatial step [m].
    T0 : float
        Peak period [s]; defines fp = 1/T0.
    h : float | None
        Depth [m], for dispersion via sig_from_k and PM_spectrum_k.
    gammafac : float
        JONSWAP peak enhancement factor γ (Default ~3.3).
    sigA, sigB : float
        Low-/high-frequency width parameters σ_A, σ_B (dimensionless in frequency domain).
    verbose : bool
        Print Hs check if True.

    Returns
    -------
    Z1_JS : np.ndarray
        JONSWAP-augmented variance spectrum Z1(kx) [m^2 per (rad/m)], length nx.
    kX : np.ndarray
        Wavenumber grid kx [rad/m], length nx.

    Notes
    -----
    - Constructs JONSWAP factor in *frequency* domain using f(k) from dispersion:
        fX = σ(k)/2π with σ = angular frequency from dispersion (via sig_from_k).
      Then applies γ^{exp[-(f-fp)^2/(2 σ_AB^2 fp^2)]}, where σ_AB = σ_A below fp, σ_B above fp.
    - Any NaNs in spectrum are zeroed.
    """
    dkx = 2*np.pi/(dx*nx)  # [rad/m]

    kX = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2*np.pi  # [rad/m]
    fX = sig_from_k(kX, h=h) / (2*np.pi)  # f(k) [Hz] from dispersion
    Z1_PM = PM_spectrum_k(kX, 1/T0, h=h)
    Z1_PM[np.isnan(Z1_PM)] = 0

    fp = 1/T0  # [Hz]
    sigAB = np.where(fX < fp, sigA, sigB)  # piecewise width
    JSfactor = gammafac ** np.exp((-(fX - fp)**2) / (2 * sigAB**2 * fp**2))
    Z1_JS = Z1_PM * JSfactor
    Z1_JS[np.isnan(Z1_JS)] = 0

    # Hs diagnostic
    sumZ1 = 4*np.sqrt(sum(Z1_JS[np.isfinite(Z1_JS)].flatten() * dkx))
    if verbose:
        sumZ1 = 4*np.sqrt(sum(Z1_JS[np.isfinite(Z1_JS)].flatten() * dkx))
        print('Hs for Jonswap : ', sumZ1)

    return Z1_JS, kX


def def_spectrum_for_surface(
    nx: int = 2048, ny: int = 2048,
    dx: float = 10, dy: float = 10,
    theta_m: float = 30,
    h: float = 1000,
    T0: float = 10,
    Hs: float = 4,
    sk_theta: float = 0.001, sk_k: float = 0.001,
    nk: int = 1001, nth: int = 36,
    klims: tuple[float, float] = (0.0002, 0.2),
    n: int = 4,
    typeSpec: str = 'Gaussian',
    verbose: bool = False
):
    """
    Build a **2D** variance spectrum Z1(kx,ky) for surface synthesis on an FFT grid.

    Parameters
    ----------
    nx, ny : int
        Spatial grid sizes.
    dx, dy : float
        Spatial steps [m].
    theta_m : float
        Mean/peak direction [deg] (used differently per model).
    h : float
        Depth [m].
    T0 : float
        Peak period [s].
    Hs : float
        Target significant wave height [m] (used by Gaussian branch).
    sk_theta, sk_k : float
        Gaussian spreads (directional and along-k spreads); units depend on model:
        - `sk_theta` is in radians if passed to define_Gaussian_spectrum_kxky with θ in radians.
          (Below, we pass theta_m*np.pi/180 to that helper, so **θ is in radians** there.)
        - `sk_k` is a spread in k [rad/m].
        #TODO: confirm exact units expected by define_Gaussian_spectrum_kxky.
    nk, nth : int
        Discretization sizes for PM * cos^{2n} directional model (polar grid).
    klims : (float, float)
        k-range [rad/m] for PM polar grid.
    n : int
        Exponent for directional distribution cos(θ)^{2n}.
    typeSpec : {'Gaussian','PM'}
        Spectrum family to generate.
    verbose : bool
        Print Hs diagnostics.

    Returns
    -------
    Z1 : np.ndarray
        2D variance spectrum on (kx,ky) grid [m^2 per (rad/m)^2], shape (ny, nx).
    kX, kY : np.ndarray
        Meshgrids of kx, ky [rad/m], shape (ny, nx).
    dkx, dky : float
        Spectral steps [rad/m].

    Notes
    -----
    - Gaussian branch: builds an anisotropic Gaussian in (kx,ky) and **normalizes to Hs** explicitly.
    - PM branch: builds a PM × cos^{2n} spectrum in polar (k,θ), transforms to (kx,ky),
      then interpolates onto FFT grid via `griddata` (fill_value=0). No subsequent Hs renormalization.
      #TODO: Add a consistent post-interpolation renormalization to match the user-requested Hs.
    - θ is in **degrees** for the user-facing API (theta_m), but converted to **radians** when
      passed to the lower-level helper(s) that expect radians.
    """
    dkx = 2*np.pi/(dx*nx)  # [rad/m]
    dky = 2*np.pi/(dy*ny)  # [rad/m]

    kX0 = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2*np.pi  # [rad/m]
    kY0 = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2*np.pi  # [rad/m]
    kX, kY = np.meshgrid(kX0, kY0)  # shapes (ny, nx), [rad/m]

    if typeSpec == 'Gaussian':
        if verbose:
            print(
                'Gaussian spectrum selected. Available options:\n'
                ' - Hs, sk_theta, sk_k.\n'
                ' With (sk_k, sk_theta) the sigma values for the k-axis along the main direction and\n'
                ' perpendicular to it respectively.\n'
                ' Other options (common to all spectrum types): nx, ny, dx, dy, T0, theta_m, h'
            )

        # define_Gaussian_spectrum_kxky expects θ in radians → pass theta_m*np.pi/180
        Z1_Gaussian0, kX, kY = define_Gaussian_spectrum_kxky(
            kX, kY, T0, theta_m*np.pi/180, sk_theta, sk_k, h=h
        )

        # Normalize to target Hs: Z1 ← (Hs/4)^2 * Z1_raw / (∑ Z1_raw dkx dky)
        Z1 = (Hs/4)**2 * Z1_Gaussian0 / np.sum(Z1_Gaussian0.flatten() * dkx * dky)

        # Hs diagnostic
        sumZ1 = 4*np.sqrt(sum(Z1.flatten() * dkx * dky))
        if verbose:
            print('Hs for Gaussian : ', sumZ1)

    elif typeSpec == 'PM':
        if verbose:
            print(
                'Pierson-Moskowitz* cos(theta)^(2*n) spectrum selected. Available options:\n'
                ' - nk, nth, klims, n.\n'
                ' With n the exponent of the directional distribution: cos(theta)^(2*n)\n'
                ' Other options (common to all spectrum types): nx, ny, dx, dy, T0, theta_m, h'
            )

        # Build polar grid in k and θ (θ in radians)
        k = np.linspace(klims[0], klims[1], nk)  # [rad/m]
        thetas = np.linspace(0, 360*(nth-1)/nth, nth) * np.pi/180.  # [rad]; wraps 0..(360-Δθ)

        # Define PM × cos^{2n} on polar grid at peak direction theta_m (in radians)
        Ekth, k, th = define_spectrum_PM_cos2n(k, thetas, T0, theta_m*np.pi/180., h=h, n=n)

        # Transform polar (k,θ) spectrum to Cartesian E(kx,ky) (helper handles Jacobian)
        # NOTE: The first argument '1' is passed unchanged (per original).  #TODO: document what it means.
        Ekxky, kx, ky = spectrum_to_kxky(1, Ekth, k, thetas, h=h)

        # Interpolate onto FFT grid (kx,ky) with fill_value=0 outside convex hull
        Z1 = spi.griddata((kx.flatten(), ky.flatten()), Ekxky.flatten(), (kX, kY), fill_value=0)

        # Hs diagnostic (no explicit renormalization here)
        sumZ1 = 4*np.sqrt(sum(Z1.flatten() * dkx * dky))
        if verbose:
            print('Hs for Pierson Moskowitz : ', sumZ1)

    # NOTE: No else branch—typeSpec must be 'Gaussian' or 'PM'.  #TODO: consider raising ValueError for others.

    return Z1, kX, kY, dkx, dky