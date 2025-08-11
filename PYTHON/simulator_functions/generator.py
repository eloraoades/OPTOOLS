import numpy as np
from scipy.interpolate import griddata
from wave_physics_functions.transforms import spectrum_from_kth_to_kxky, wavespec_Efth_to_Ekxky
from typing import Optional, tuple

import numpy as np


def surface_1D_from_Z1kx(
    Z1: np.ndarray,
    kX: np.ndarray,
    i: int,
    nx: int = None,
    dx: float = None,
    dkx: float = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate a 1D spatial surface realization from a given spectral density (Z1) in kx-space.

    Parameters
    ----------
    Z1 : np.ndarray
        1D array of spectral density values (variance per unit wavenumber), shape (nx,).
    kX : np.ndarray
        1D array of wavenumbers (rad/m), same length as Z1.
    i : int
        Seed for random phase generation, for repeatable results.
    nx : int, optional
        Number of spatial points. If None, inferred from Z1.
    dx : float, optional
        Spatial step (m). If None, calculated from dkx and nx.
    dkx : float, optional
        Spectral step (rad/m). If None, calculated from dx and nx.

    Returns
    -------
    S1_r : np.ndarray
        Real part of the synthesized spatial field (length nx).
    S1_i : np.ndarray
        Imaginary part (useful for analytic envelope).
    X : np.ndarray
        Array of spatial positions (m), shape (nx,).
    dkx : float
        Spectral step used (rad/m).
    """
    # If nx (number of points) is not given, infer from spectrum size
    if nx is None:
        nx = len(Z1)

    # Set dx (spatial step) if not provided.
    # If both dx and dkx are None, infer dx from kX step and nx.
    if (dx is None):
        if dkx is None:
            # This ensures that the total spatial domain is 2pi/(delta_kx), matching FFT conventions.
            dx = 2 * np.pi / ((kX[1] - kX[0]) * nx)
        else:
            # If dkx is given, use that directly to compute dx
            dx = 2 * np.pi / (dkx * nx)

    # Similarly, if dkx (spectral step) is not given, calculate from dx and nx
    if (dkx is None):
        dkx = 2 * np.pi / (dx * nx)

    # Initialize a reproducible random number generator with seed i
    rng = np.random.default_rng(i)

    # Generate random phases uniformly in [0,1), then convert to phase in [0, 2pi)
    rg = rng.uniform(low=0.0, high=1.0, size=(nx))

    # Construct the complex spectrum with random phase and correct amplitude
    # sqrt(2 * Z1 * dkx): amplitude for correct variance; exp(1j * 2*pi*rg): random phase
    zhats = np.fft.ifftshift(np.sqrt(2 * Z1 * dkx) *
                             np.exp(1j * 2 * np.pi * rg))

    # TODO: (Unused) Centered version of wavenumber array — not used but may have been for debugging or further processing ?
    kx0 = np.fft.ifftshift(kX)

    # Inverse FFT to synthesize the real space field
    S1_r = np.real(np.fft.ifft(zhats, norm="forward"))
    # Useful for envelope/analytic signal analysis
    S1_i = np.imag(np.fft.ifft(zhats, norm="forward"))

    # Create a spatial grid from 0 to (nx-1)*dx in steps of dx
    X = np.arange(0, nx * dx, dx)

    return S1_r, S1_i, X, dkx


def surface_2D_from_Z1kxky(
    Z1: np.ndarray,
    kX: np.ndarray,
    kY: np.ndarray,
    i: int,
    nx: int = None,
    ny: int = None,
    dx: float = None,
    dy: float = None,
    dkx: float = None,
    dky: float = None,
    phase_type: str = 'uniform',
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """
    Generate a 2D spatial surface realization from a given 2D spectral density (Z1) in (kx, ky) space.

    Parameters
    ----------
    Z1 : np.ndarray
        2D spectral density array (variance per unit wavenumber^2), shape (ny, nx).
    kX : np.ndarray
        2D or 1D array of kx values (rad/m), shape (ny, nx) or (nx,).
    kY : np.ndarray
        2D or 1D array of ky values (rad/m), shape (ny, nx) or (ny,).
    i : int
        Seed for random phase generation.
    nx, ny : int, optional
        Number of grid points in x and y. If None, inferred from kX, kY.
    dx, dy : float, optional
        Grid step in x and y (m).
    dkx, dky : float, optional
        Spectral step in kx and ky (rad/m).
    phase_type : str, optional
        'uniform' (default) for uniformly random phases, or 'normal' for Gaussian random amplitudes/phases.
    verbose : bool, optional
        If 'True', print detailed info.

    Returns
    -------
    S2_r : np.ndarray
        Real part of synthesized 2D surface.
    S2_i : np.ndarray
        Imaginary part (for analytic envelope).
    X : np.ndarray
        x-coordinate grid (meters).
    Y : np.ndarray
        y-coordinate grid (meters).
    kX0, kY0 : np.ndarray
        Unique kx and ky wavenumber vectors (rad/m).
    i : int
        RNG seed actually used.
    dkx, dky : float
        Final spectral steps (rad/m).

    Notes
    -----
    - Reconstructs a spatial field with prescribed 2D spectrum using the random phase/amplitude method.
    - Amplitude scaling ensures variance matches input spectrum.
    - See e.g., [T. Elfouhaily et al., JGR, 1997] for similar random sea surface methods.
    """

    # Ensure kX0 and kY0 are unique, sorted arrays of kx/ky
    kX0 = np.unique(kX)
    kY0 = np.unique(kY)

    # If not provided, infer nx, ny from size of kX0, kY0
    if nx is None:
        nx = len(kX0)
    if ny is None:
        ny = len(kY0)

    # Compute half-grid shifts (shx, shy), used in old FFT code to center zero-frequency.
    shx = np.floor(nx/2-1)
    shy = np.floor(ny/2-1)

    # Optionally print grid and spectral spacings
    if verbose:
        print('from vec kX0, dkx = ', (kX0[1] - kX0[0]))
        print('from vec kY0, dky = ', (kY0[1] - kY0[0]))

    # Set dx and dy (spatial steps). Fallback to computing from dkx/dky if not given.
    if dx is None:
        if dkx is None:
            dx = 2 * np.pi / ((kX0[1] - kX0[0]) * nx)
        else:
            dx = 2 * np.pi / (dkx * nx)
    if dy is None:
        if dky is None:
            dy = 2 * np.pi / ((kY0[1] - kY0[0]) * ny)
        else:
            dy = 2 * np.pi / (dky * ny)

    # If dkx/dky still not set, compute from dx/dy and grid size
    if dkx is None:
        dkx = 2 * np.pi / (dx * nx)
    if dky is None:
        dky = 2 * np.pi / (dy * ny)

    if verbose:
        print("variables : ")
        print(' dx = ', dx, ' ; dy = ', dy, ' ; nx = ', nx, ' ; ny = ', ny)
        print('dkx = ', dkx, ' ; dky = ', dky)

    # --- Phase and amplitude assignment ---
    # Seeded random generator for reproducibility
    rng = np.random.default_rng(i)
    if phase_type == 'uniform':
        # Uniform random phases in [0,1)
        rg = rng.uniform(low=0.0, high=1.0, size=(ny, nx))
    else:
        # If phase_type is not 'uniform', use normal distribution for amplitudes/phases (less common)
        rg = rng.normal(0, 1, (ny, nx))

    # Construct complex spectral coefficients: amplitude * exp(i*phase)
    # sqrt(2*Z1*dkx*dky): amplitude to match variance per Parseval's theorem
    # 2*pi*rg: random phase in [0, 2pi)
    zhats = np.fft.ifftshift(np.sqrt(2 * Z1 * dkx * dky)
                             * np.exp(1j * 2 * np.pi * rg))

    # TODO: (Unused) These are ifftshifted copies of the wavenumber arrays, possibly for later use or verification.
    ky2D = np.fft.ifftshift(kY)
    kx2D = np.fft.ifftshift(kX)

    # Compute inverse 2D FFT to synthesize the surface
    # Real part: physical sea surface; Imag part: for analytic envelope methods (e.g., Hilbert transform)
    S2_r = np.real(np.fft.ifft2(zhats, norm="forward"))
    S2_i = np.imag(np.fft.ifft2(zhats, norm="forward"))

    # Build spatial coordinate arrays (from 0 to (nx-1)*dx and (ny-1)*dy)
    X = np.arange(0, np.floor(nx * dx), dx)
    Y = np.arange(0, np.floor(ny * dy), dy)

    return S2_r, S2_i, X, Y, kX0, kY0, i, dkx, dky


def surface_from_Efth_noxr(
    eftn: np.ndarray,
    fren: np.ndarray,
    df_vec: np.ndarray,
    th_vec: np.ndarray,
    dth: float,
    seed: int = 0,
    nx: int = 2048,
    ny: int = 2048,
    dx: float = 10.0,
    dy: float = 10.0,
    h: float = 10000.0,
    iswvnb: int = 0,
    verbose: int = 1
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, int,
    np.ndarray, np.ndarray, np.ndarray, float, float, float, float
]:
    """
    Generate a 2D free surface realization from a directional wave energy spectrum E(f, θ).

    Parameters
    ----------
    eftn : np.ndarray
        2D directional spectrum array, E(f, θ) [m^2/Hz/deg] or [m^2/Hz/rad] (check θ units).
        Shape: (nf, nθ).
    fren : np.ndarray
        1D array of frequency values [Hz], length nf.
    df_vec : np.ndarray
        1D array of frequency bin widths [Hz], length nf.
    th_vec : np.ndarray
        1D array of direction values [deg or rad], length nθ.  #TODO: Confirm units
    dth : float
        Directional bin width [deg or rad].  #TODO: Confirm units!
    seed : int, optional
        Random seed for surface phase (default: 0).
    nx, ny : int, optional
        Number of spatial points in x and y (default: 2048).
    dx, dy : float, optional
        Spatial step in x and y [m] (default: 10.0).
    h : float, optional
        Water depth [m] (default: 10000.0).  #TODO: Currently not used ? 
    iswvnb : int, optional
        If 1, indicates input spectrum is in wavenumber space, else frequency space (default: 0).
    verbose : int, optional
        If 1, prints info (default: 1).

    Returns
    -------
    S2_r : np.ndarray
        Real part of synthesized surface [m], shape (ny, nx).
    S2_i : np.ndarray
        Imaginary part [m], useful for analytic signal methods.
    X : np.ndarray
        x-grid [m], shape (nx,).
    Y : np.ndarray
        y-grid [m], shape (ny,).
    rg : int
        RNG seed used for phase.
    kX0 : np.ndarray
        Unique kx wavenumbers [rad/m], shape (nx,).
    kY0 : np.ndarray
        Unique ky wavenumbers [rad/m], shape (ny,).
    Ekxky : np.ndarray
        2D spectral density on (kx, ky) grid [m^2/rad^2/m^2], shape (ny, nx).  #TODO: Confirm units
    dkx : float
        Spectral step in kx [rad/m].
    dky : float
        Spectral step in ky [rad/m].
    Hskk : float
        Significant wave height [m] estimated from the (kx,ky) spectrum.
    Qkk : float
        Spectral narrowness parameter (see De Carlo et al. 2023).

    Notes
    -----
    - Converts input E(f, θ) spectrum into cartesian (kx, ky) spectrum using dispersion and group velocity.
    - Energy is conserved by normalizing total variance (significant wave height).
    - See De Carlo et al. (2023) and standard ocean wave modeling texts for details on spectral transforms and random phase synthesis.

    References
    ----------
    - See De Carlo etf al. (2023), eq. 16.
    - Komen et al. (1996), "Dynamics and Modelling of Ocean Waves".
    """

    [nf, nt] = np.shape(eftn)  # nf: frequency bins, nt: directional bins

    tpi = 2 * np.pi
    grav = 9.81

    # Significant wave height from input spectrum, integrating over all frequencies and directions
    # NOTE: efth units must be [m^2/Hz/deg] or [m^2/Hz/rad], matching dth units
    Hs1 = 4 * np.sqrt(np.sum(np.sum(eftn, axis=1) * df_vec) * dth)  # [m]

    # -- Handle spectrum periodicity in θ (wraparound at 0/360 or -π/π) --
    dlast = th_vec[0] + 360.  # #TODO: Confirm that th_vec is in degrees!
    # Extended θ vector for wraparound [deg]
    dirm = np.concatenate([th_vec, [dlast]])

    elast = eftn[:, 0]  # First direction spectrum (for wraparound)
    # Extended E(f, θ) for wraparound
    eftm = np.concatenate([eftn.T, [elast]]).T

    # -- Convert from (f, θ) to (kx, ky) via linear dispersion relation --
    # Compute wavenumber for each frequency: k = (2πf)^2 / g (deep water approximation)
    kn = (2 * np.pi * fren)**2 / grav  # [rad/m]

    # Shape (nf, nt+1): wavenumber for each (f, θ)
    kn2 = np.tile(kn.reshape(nf, 1), (1, nt+1))

    # Group velocity for each k: Cg = 0.5 * sqrt(g / k) (deep water)
    # Jacobian d(f,θ)→d(kx,ky): Cg/(k*2π)
    Cg2 = np.sqrt(grav / (kn2)) * 0.5  # [m/s]
    Jac = Cg2 / (kn2 * tpi)  # [s / (rad/m)]

    # θ grid (extended): convert from degrees to radians for trig
    dirm2 = np.tile(dirm.T, (nf, 1)) * np.pi / 180.  # [rad]

    # Decompose k into (kx, ky) for each (f, θ)
    kxn = kn2 * np.cos(dirm2)  # [rad/m]
    kyn = kn2 * np.sin(dirm2)  # [rad/m]

    # -- Build output (kx, ky) grid matching spatial grid size --
    dkx = 2 * np.pi / (dx * nx)  # [rad/m]
    dky = 2 * np.pi / (dy * ny)  # [rad/m]
    kX0 = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi  # [rad/m]
    kY0 = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2 * np.pi  # [rad/m]

    kX, kY = np.meshgrid(kX0, kY0)  # [rad/m], shape (ny, nx)

    # -- Interpolate the spectrum from (f, θ) to (kx, ky) grid --
    # The Jacobian ensures energy conservation in transformation
    Ekxky = griddata(
        (kxn.flatten(), kyn.flatten()),
        (eftm * Jac).flatten(),
        (kX, kY),
        method='nearest'
    )  # [m^2/rad^2/m^2] #TODO: Confirm/clarify units!

    # -- Normalize energy to conserve significant wave height --
    Hs2 = 4 * np.sqrt(np.sum(np.sum(Ekxky)) * dkx * dky)  # [m]
    Ekxky = Ekxky * (Hs1 / Hs2)**2  # Scale so output spectrum matches input Hs
    E_total = np.sum(Ekxky.flatten()) * dkx * dky  # Total variance [m^2]

    # -- Calculate spectral narrowness (Qkk) --
    # Note: For double-sided spectrum, Qkk uses 0.5 factor (see De Carlo et al.)
    Qkk = np.sqrt(np.sum(Ekxky.flatten()**2) * dkx * dky * 0.5) / E_total

    # Significant wave height from final (kx,ky) spectrum [m]
    Hskk = 4 * np.sqrt(E_total)

    # -- Diagnostic printout (optional) --
    if verbose == 1:
        print('Hs1,Hs2:', Hs1, Hs2, Hskk)

    # -- Synthesize the spatial surface from (kx,ky) spectrum --
    S2_r, S2_i, X, Y, kX0, kY0, rg, dkx, dky = surface_2D_from_Z1kxky(
        Ekxky, kX, kY, seed)

    # h is currently unused! #TODO: Consider using water depth for finite-depth dispersion in future.
    return S2_r, S2_i, X, Y, rg, kX0, kY0, Ekxky, dkx, dky, Hskk, Qkk


def surface_from_Efth(
    Efth: np.ndarray,
    f_vec: np.ndarray,
    df_vec: np.ndarray,
    th_vec: np.ndarray,
    dth: float,
    seed: int = 0,
    nx: int = 2048,
    ny: int = 2048,
    dx: float = 10.0,
    dy: float = 10.0,
    h: float = 10000.0,
    iswvnb: int = 0,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, int,
    np.ndarray, np.ndarray, np.ndarray, float, float, float, float
]:
    """
    Build a 2D free-surface realization η(x,y) from a directional variance spectrum E(f, θ).

    Conventions
    ------------------------------------------
    - E(f, θ) here is the **variance density** of free-surface elevation (not energy density).
      Units typically [m^2 / (Hz · deg)] or [m^2 / (Hz · rad)] depending on θ units.

      #TODO: Confirm θ units for this code path; below I *assume degrees* where noted.
    - No ρg factor is included (we’re not converting to energy per unit area).
    - Wavenumbers kx, ky are in [rad/m]. Spatial steps dx, dy are in [m]. Frequencies f in [Hz].

    Parameters
    ----------
    Efth : np.ndarray
        Directional variance spectrum E(f, θ), shape (n_phi, nk) per the code below (see dims).
        #TODO: The naming is slightly confusing: dims = ['n_phi','nk'] but coords map φ->th_vec, k->f_vec.
        Units: [m^2 / (Hz · deg)] or [m^2 / (Hz · rad)] depending on θ convention.
    f_vec : np.ndarray
        Frequency vector [Hz], length nk.
    df_vec : np.ndarray
        Frequency bin widths [Hz], length nk.
    th_vec : np.ndarray
        Direction vector [deg] or [rad], length n_phi.  #TODO: Confirm; wrap logic below assumes degrees.
    dth : float
        Direction bin width [deg] or [rad].  #TODO: Confirm consistency with th_vec.
    seed : int
        RNG seed for phase realization.
    nx, ny : int
        Spatial grid sizes in x and y.
    dx, dy : float
        Spatial steps [m].
    h : float
        Depth [m]. (Currently unused in this function.)  #TODO: Consider finite-depth dispersion in future.
    iswvnb : int
        Flag: if nonzero, treat input as wavenumber-based spectrum path (see below).

    Returns
    -------
    S2_r : np.ndarray
        Real part of synthesized surface η(x,y) [m], shape (ny, nx).
    S2_i : np.ndarray
        Imaginary part (analytic-signal companion) [m], shape (ny, nx).
    X : np.ndarray
        x-grid [m], length nx.
    Y : np.ndarray
        y-grid [m], length ny.
    rg : int
        Seed echoed back (for reproducibility bookkeeping).
    kX0 : np.ndarray
        Unique kx samples [rad/m], length nx.
    kY0 : np.ndarray
        Unique ky samples [rad/m], length ny.
    Ekxky_for_surf : np.ndarray
        Variance spectrum on (kx,ky) grid used for synthesis. Units: [m^2 / (rad/m)^2].
        Note: internally we pass dkx, dky in [rad/m]; see scaling notes where (2π) factors appear.
        #TODO: Unit audit: confirm factorization so that ∑∑ Ekxky dkx dky = variance [m^2].
    dkx : float
        Spectral step in kx [rad/m].
    dky : float
        Spectral step in ky [rad/m].
    Hskk : float
        Significant wave height inferred from Ekxky [m]. (Hs = 4√variance)
    Qkk : float
        Spectral narrowness parameter (per De Carlo et al., single-sided correction applied).

    Physics notes
    -------------
    - Remap E(f,θ) → E(kx,ky) and then synthesize η via inverse 2D FFT with random phases.
    - The helper `wavespec_Efth_to_Ekxky` handles the polar→Cartesian transform with specified dkx,dky.
      Pass dkx,dky in *cycles/m* (i.e., divided by 2π), then convert back to [rad/m] downstream.
      #TODO: This dual unit usage is subtle—document carefully to avoid mistakes.
    - After remapping, we compute total variance ∑ Ekxky dkx dky to get Hs and Qkk, and to check consistency.
    """
    import xarray as xr

    g = 9.81  # [m/s^2]

    # Wrap spectrum in xarray for clean edge-handling in θ (close the -180/180 or 0/360 gap).
    # NOTE: dims order is ['n_phi','nk'] with coords 'phi_vector' (θ) and 'k_vector' (f).
    # #TODO: Consider renaming 'k_vector'→'f_vector' to reduce confusion.
    spec = xr.DataArray(
        Efth,
        dims=['n_phi', 'nk'],
        coords={
            "phi_vector": (["n_phi"], th_vec),  # θ grid [deg or rad]
            "k_vector": (["nk"], f_vec),        # frequency grid [Hz]
        },
    )

    # Duplicate first and last direction to enforce 2π periodicity before interpolation.
    spec_bis = xr.concat(
        [spec.isel(n_phi=-1), spec, spec.isel(n_phi=0)], dim="n_phi")

    # --- change the first and last new values to have a 2pi revolution ---------------
    A = np.concatenate([[-360], np.zeros((spec.sizes['n_phi'])), [360]])
    factor = xr.DataArray(A, dims="n_phi")
    # Shift the θ coordinate to close the loop.
    spec_bis['phi_vector'].values = spec_bis['phi_vector'] + factor
    spec_bis = spec_bis.interpolate_na(dim='n_phi')  # Fill any gaps.
    # #TODO: This code assumes degrees for θ because of ±360 shifts. Confirm th_vec units.

    # --- Build the target Cartesian spectral grid consistent with spatial resolution ---
    dkx = 2 * np.pi / (dx * nx)  # [rad/m]
    dky = 2 * np.pi / (dy * ny)  # [rad/m]
    kX0 = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * \
        2 * np.pi  # [rad/m], length nx
    kY0 = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * \
        2 * np.pi  # [rad/m], length ny
    kX, kY = np.meshgrid(kX0, kY0)  # [rad/m], shapes (ny, nx)

    # Polar coordinates from Cartesian (for interpolation in θ,f or k,θ depending on branch)
    kK = (np.sqrt(kX**2 + kY**2))         # |k| [rad/m]
    kPhi = np.arctan2(kY, kX) * 180 / np.pi  # direction [deg]
    # Alias used below; currently set equal to |k|. #TODO: Naming is confusing.
    kF = kK

    # Map θ into [0, 360) if degrees are assumed
    # [deg]  #TODO: If θ is in radians, this needs adjustment.
    kPhi[kPhi < 0] = kPhi[kPhi < 0] + 360

    # xarray “stacked” helper for interpolation indexing
    # currently = |k| [rad/m]
    kF2 = xr.DataArray(kF, coords=[("ky", kY0), ("kx", kX0)])
    kPhi2 = xr.DataArray(kPhi, coords=[("ky", kY0), ("kx", kX0)])   # [deg]
    FPhi2s = xr.Dataset({'kF': kF2, 'kPhi': kPhi2}
                        ).stack(flattened=["ky", "kx"])

    if iswvnb:
        # --- Branch: input handling in wavenumber/θ space via spectrum_from_kth_to_kxky ---
        # Keep this reference as requested; do not remove.
        Ekxky0, kx, ky = spectrum_from_kth_to_kxky(
            np.squeeze(spec_bis.compute().data),
            # NOTE: labeled as 'k_vector' but is f_vec earlier. #TODO
            spec_bis['k_vector'].compute().data,
            spec_bis["phi_vector"].compute().data
        )
        # Wrap into xarray for interpolation onto (kx,ky) grid
        Ekxky = xr.DataArray(
            Ekxky0, dims=("nf", "n_phi"),
            coords={"nf": f_vec, "n_phi": spec_bis['phi_vector']}
        )
        # Interpolate onto requested FPhi2s (kF,kPhi). fill_value=0 outside support.
        B = Ekxky.interp(nf=FPhi2s.kF, n_phi=FPhi2s.kPhi,
                         kwargs={"fill_value": 0})
        B.name = 'Ekxky_new'
        B0 = B.reset_coords(("nf", "n_phi"))
        Ekxky_for_surf = B0.Ekxky_new.unstack(dim='flattened')
        # Integrate to get total variance [m^2]
        E_total = np.sum(Ekxky_for_surf.values.flatten()) * dkx * dky
        # Qkk using single-sided correction factor 0.5 (per De Carlo et al., 2023)
        Qkk = np.sqrt(np.sum(Ekxky_for_surf.values.flatten()**2)
                      * dkx * dky * 0.5) / E_total
    else:
        # --- Branch: transform Efth(f,θ) to Ekxky(kx,ky) using provided helper ---
        # NOTE: wavespec_Efth_to_Ekxky expects dkx,dky in cycles/m, not rad/m → divide by (2π).
        Ekxky, kx, ky, kx2, ky2 = wavespec_Efth_to_Ekxky(
            Efth,
            f_vec,
            df_vec,
            th_vec,
            dth,
            dkx=dkx / (2 * np.pi),  # [cycles/m]
            dky=dky / (2 * np.pi),  # [cycles/m]
            nkx=ny // 2,
            nky=nx // 2,
            doublesided=0
        )
        # Convert returned Ekxky to [per rad/m]^2 by dividing by (2π)^2 (since input dk were in cycles/m)
        Ekxky_for_surf = Ekxky / (2 * np.pi)**2
        # Total variance [m^2]
        E_total = np.sum(Ekxky_for_surf.flatten()) * dkx * dky
        # Narrowness parameter with single-sided factor
        Qkk = np.sqrt(np.sum(Ekxky_for_surf.flatten()**2)
                      * dkx * dky * 0.5) / E_total

    # Significant wave height from total variance (Hs = 4 * sqrt(variance))
    Hskk = 4 * np.sqrt(E_total)  # [m]

    # Synthesize the spatial surface using the already-checked 2D routine.
    S2_r, S2_i, X, Y, kX0, kY0, rg, dkx, dky = surface_2D_from_Z1kxky(
        Ekxky_for_surf, kX, kY, seed
    )

    return S2_r, S2_i, X, Y, rg, kX0, kY0, Ekxky_for_surf, dkx, dky, Hskk, Qkk


def surface_from_Z1kxky(
    Z1: np.ndarray,
    kX: np.ndarray,
    kY: np.ndarray,
    nx: int | None = None,
    ny: int | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dkx: float | None = None,
    dky: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize a 2D surface η(x,y) from a provided Cartesian variance spectrum Z1(kx, ky).

    Conventions & Units
    -------------------
    - Z1: variance spectral density on a Cartesian grid, units so that
           ∑∑ Z1 * dkx * dky = Var[η]  [m^2].
           With kx, ky in [rad/m], dkx, dky are also [rad/m].
    - kX, kY: wavenumber samples [rad/m]. Often 2D meshgrids of shape (ny, nx).
      (#TODO: Confirm if kX, kY are vectors or 2D arrays in the current code path.)
    - dx, dy: spatial steps [m]; nx, ny: number of spatial samples.

    Notes
    -----
    - This version uses a *Gaussian* random array `rg` inside exp(i 2π rg), i.e., non-uniform phases.
      That is unusual compared to uniform[0,1) phases. I chose to keep it as-is to preserve original behavior.
      (#TODO: consider switching to uniform phases for standard random-phase synthesis.)
    - Uses np.roll-based centering rather than fftshift/ifftshift used elsewhere.
      (#TODO: unify centering strategy across functions for clarity.)
    - Scales spatial field by (nx*ny) because ifft2 here uses default normalization.
      (#TODO: document/standardize FFT normalization across other files in module.)
    """

    kX0 = np.unique(kX)  # [rad/m]
    kY0 = np.unique(kY)  # [rad/m]

    if nx == None:
        nx = Z1.shape[1]
    if ny == None:
        ny = Z1.shape[0]

    # Half-grid offsets (legacy centering helpers)
    shx = np.floor(nx/2-1)
    shy = np.floor(ny/2-1)

    # Infer dx from dkx (or from kX0 spacing) without changing logic
    if (dx == None):
        if dkx == None:
            dx = 2*np.pi/((kX0[1] - kX0[0])*nx)  # [m]
        else:
            dx = 2*np.pi/(dkx*nx)                # [m]

    # Infer dy from dky (or from kY0 spacing). Note the original had a floor().
    if (dy == None):
        if dky == None:
            dy = np.floor(2*np.pi/((kY0[1] - kY0[0])*ny))  # [m]
            # TODO: The floor() here is unusual and can distort dy; consider removing in refactor.
        else:
            dy = 2*np.pi/(dky*ny)                          # [m]

    # Back-compute dkx, dky if still None.
    if (dkx == None):
        dkx = 2*np.pi/(dx*nx)  # [rad/m]
    if (dky == None):
        dky = 2*np.pi/(dy*ny)  # [rad/m]

    # Random array: normal(0,1) then used as phase -> exp(i 2π rg). See note above.
    rg = np.random.normal(0, 1, (ny, nx))  # kept as-is
    zhats = np.roll(
        np.sqrt(2*Z1*dkx*dky) * np.exp(1j*2*np.pi*rg),
        (-int(shy), -int(shx)), axis=(0, 1)
    )

    # “Centered” versions of ky, kx via roll (not used, but kept for compatibility)
    ky2D = np.roll(kY, (-int(shy), -int(shx)), axis=(0, 1))
    # TODO: verify future need
    kx2D = np.roll(kX, (-int(shy), -int(shx)), axis=(0, 1))

    # Inverse 2D FFT (default normalization), then scale by (nx*ny) to match intended energy.
    # TODO: prefer a consistent 'norm="forward"' approach like other functions to avoid manual scaling.
    S1 = np.real(np.fft.ifft2(zhats)) * (nx*ny)  # [m]

    # Spatial grids
    X = np.arange(0, nx*dx, dx)  # [m]
    Y = np.arange(0, ny*dy, dy)  # [m]

    return S1, X, Y


def surface_from_Z1kxky_uniform_phase(
    Z1: np.ndarray,
    kX: np.ndarray,
    kY: np.ndarray,
    i: int,
    nx: int | None = None,
    ny: int | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dkx: float | None = None,
    dky: float | None = None
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float
]:
    """
    Synthesize a 2D surface η(x,y) from Z1(kx,ky) using **uniform** random phases (seeded).

    Differences vs. `surface_from_Z1kxky`
    -------------------------------------
    - Uses uniform phases u ~ U[0,1) → exp(i 2π u), which is the standard random-phase model.
    - Uses fft ifftshift-based centering (consistent with other functions in this file).
    - Uses np.fft.ifft2(..., norm="forward") so scaling matches spectral density directly.
      (No extra (nx*ny) factor.)

    Units & Conventions
    -------------------
    - Z1: variance spectral density on (kx,ky) so that ∑∑ Z1 * dkx * dky = Var[η] [m^2].
    - kX, kY: [rad/m]. Often 2D arrays (meshgrids).
    - dkx, dky: [rad/m]. dx, dy: [m].

    Returns
    -------
    S1 : np.ndarray
        Real part η(x,y) [m], shape (ny, nx).
    S2 : np.ndarray
        Imaginary companion (analytic-signal) [m], shape (ny, nx).
    X, Y : np.ndarray
        Spatial grids [m].
    kX0 : np.ndarray
        Unique kx samples [rad/m].
    rg : int
        RNG seed echoed back.
    dkx, dky : float
        Spectral steps [rad/m].
    """

    kX0 = np.unique(kX)  # [rad/m]
    kY0 = np.unique(kY)  # [rad/m]

    if nx == None:
        nx = Z1.shape[1]
    if ny == None:
        ny = Z1.shape[0]

    # Legacy “half-grid” offsets; kept for compatibility (not used further here).
    shx = np.floor(nx/2-1)
    shy = np.floor(ny/2-1)

    # Infer dx,dy from dkx,dky (or from kX0,kY0 spacing)
    if (dx == None):
        if dkx == None:
            dx = 2*np.pi/((kX0[1] - kX0[0])*nx)  # [m]
        else:
            dx = 2*np.pi/(dkx*nx)                # [m]

    if (dy == None):
        if dky == None:
            dy = np.floor(2*np.pi/((kY0[1] - kY0[0])*ny))  # [m]
            # TODO: floor() again; consider removing in refactor.
        else:
            dy = 2*np.pi/(dky*ny)                          # [m]

    if (dkx == None):
        dkx = 2*np.pi/(dx*nx)  # [rad/m]
    if (dky == None):
        dky = 2*np.pi/(dy*ny)  # [rad/m]

    # Seeded uniform random phases u ∈ [0,1)
    rng = np.random.default_rng(i)
    # keep variable name to match original
    rg = rng.uniform(low=0.0, high=1.0, size=(ny, nx))

    # Build complex spectrum with correct amplitude and uniform phase, centered via ifftshift
    zhats = np.fft.ifftshift(np.sqrt(2*Z1*dkx*dky) * np.exp(1j*2*np.pi*rg))

    # “Centered” kY, kX via ifftshift. # NOTE: Not used downstream, but kept intentionally.
    ky2D = np.fft.ifftshift(kY)
    kx2D = np.fft.ifftshift(kX)  # TODO: revisit future need for these.

    # Inverse 2D FFT with 'forward' normalization: Parseval-friendly with dkx, dky scaling
    S1 = np.real(np.fft.ifft2(zhats, norm="forward"))  # [m]
    S2 = np.imag(np.fft.ifft2(zhats, norm="forward"))  # [m]

    # Spatial grids
    X = np.arange(0, nx*dx, dx)  # [m]
    Y = np.arange(0, ny*dy, dy)  # [m]

    return S1, S2, X, Y, kX0, rg, dkx, dky
