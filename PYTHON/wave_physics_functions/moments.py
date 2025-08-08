# The following code was originally a part of the OPTOOLS project, developed by Fabrice Ardhuin & Marina de Carlo. 
# The original repo can be found at https://github.com/ardhuin/OPTOOLS.git 

# This file has been adapted from the original file "wave_physiscs_functions.py" (OPTOOLS/PYTHON) & contains 
# functions for computing moment statistics from wave spectra, including significant wave height, mean direction, and directional spread.

import numpy as np

def wavespec_Efth_to_first3(efth, fren, dfreq, dirn, dth):
    """
    Computes first 3 directional moments from E(f,theta) spectrum.

    Parameters:
        efth   : 2D array, spectral energy density E(f, θ)
        fren   : 1D array, frequency bins [Hz]
        dfreq  : 1D array, frequency step per bin
        dirn   : 1D array, direction bins [deg]
        dth    : float, directional step [deg]

    Returns:
        Ef     : 1D array, spectral energy density per frequency
        th1m   : 1D array, first moment mean direction [deg]
        sth1m  : 1D array, first moment directional spread [deg]
        Hs     : float, significant wave height [m]
        Tm0m1  : float, mean wave period Tm0,1 [s]
        Qf     : float, frequency peakedness factor
        Qkk    : float, directional peakedness factor
    """
    d2r = np.pi / 180
    grav = 9.81
    nf, nt = efth.shape
    dir_rad = dirn * d2r

    Ef = np.sum(efth, axis=1) * dth
    Etot = np.sum(Ef * dfreq)
    eftn = 0.5 * (efth + np.roll(efth, nt // 2, axis=1))

    a1 = np.sum(efth * np.cos(dir_rad), axis=1) * dth / Ef
    b1 = np.sum(efth * np.sin(dir_rad), axis=1) * dth / Ef
    m1 = np.sqrt(a1**2 + b1**2)

    Q1 = np.sum(eftn**2, axis=1) * dth
    Q2 = np.sum(Q1 * dfreq * grav**2 / (2 * (2 * np.pi)**4 * fren**3))

    Qkk = np.sqrt(Q2) / Etot
    Qf = np.sqrt(np.sum(Ef**2 * dfreq)) / Etot
    Tm0m1 = np.sum(Ef * dfreq / fren) / Etot
    Hs = 4 * np.sqrt(Etot)
    
    th1m = np.arctan2(b1, a1) / d2r
    sth1m = np.sqrt(np.abs(2 * (1 - m1))) / d2r

    return Ef, th1m, sth1m, Hs, Tm0m1, Qf, Qkk

def wavespec_Efth_to_first5(efth, fren, dfreq, dirn, dth):
    """
    Computes first 5 directional moments from E(f,theta) spectrum.

    Returns:
        Ef     : 1D array, spectral energy per frequency
        th1m   : 1D array, 1st moment mean direction [deg]
        sth1m  : 1D array, 1st moment spread [deg]
        th2m   : 1D array, 2nd moment mean direction [deg]
        sth2m  : 1D array, 2nd moment spread [deg]
        Hs     : float, significant wave height [m]
        Tm0m1  : float, mean period Tm0,1 [s]
        Qf     : float, frequency peakedness
        Qkk    : float, directional peakedness
    """
    d2r = np.pi / 180
    grav = 9.81
    nf, nt = efth.shape
    dir_rad = dirn * d2r

    Ef = np.sum(efth, axis=1) * dth
    Etot = np.sum(Ef * dfreq)
    eftn = 0.5 * (efth + np.roll(efth, nt // 2, axis=1))

    a1 = np.sum(efth * np.cos(dir_rad), axis=1) * dth / Ef
    b1 = np.sum(efth * np.sin(dir_rad), axis=1) * dth / Ef
    m1 = np.sqrt(a1**2 + b1**2)

    a2 = np.sum(efth * np.cos(2 * dir_rad), axis=1) * dth / Ef
    b2 = np.sum(efth * np.sin(2 * dir_rad), axis=1) * dth / Ef
    m2 = np.sqrt(a2**2 + b2**2)

    Q1 = np.sum(eftn**2, axis=1) * dth
    Q2 = np.sum(Q1 * dfreq * grav**2 / (2 * (2 * np.pi)**4 * fren**3))

    Qkk = np.sqrt(Q2) / Etot
    Qf = np.sqrt(np.sum(Ef**2 * dfreq)) / Etot
    Tm0m1 = np.sum(Ef * dfreq / fren) / Etot
    Hs = 4 * np.sqrt(Etot)

    th1m = np.arctan2(b1, a1) / d2r
    th2m = np.arctan2(b2, a2) / d2r
    
    sth1m = np.sqrt(np.abs(2 * (1 - m1))) / d2r
    sth2m = np.sqrt(np.abs(0.5 * (1 - m2))) / d2r

    return Ef, th1m, sth1m, th2m, sth2m, Hs, Tm0m1, Qf, Qkk

def wavespec_MEM(a0, a1, a2, b1, b2, ndirs):
    """
    Computes directional distribution using the Maximum Entropy Method (MEM).

    Parameters:
        a0   : energy spectrum
        a1   : 1st cosine moment
        a2   : 2nd cosine moment
        b1   : 1st sine moment
        b2   : 2nd sine moment
        ndirs: number of directional bins

    Returns:
        sp2d : directional spectrum E(f,θ)
        D    : normalized directional spreading function
        dirs : directional bins [deg]
    """
    nfreq = np.size(a0)
    d2r = np.pi / 180
    dtheta = 360 / ndirs
    dirs = np.arange(ndirs) * dtheta
    
    c1 = a1 + 1j * b1
    c2 = a2 + 1j * b2
    
    p1 = (c1 - c2 * np.conj(c1)) / (1 - np.abs(c1)**2)
    p2 = c2 - c1 * p1

    x = 1 - p1 * np.conj(c1) - p2 * np.conj(c2)
    x = np.tile(np.real(x), (ndirs, 1)).T

    theta = dirs * d2r
    e1 = np.tile(np.exp(-1j * theta), (nfreq, 1))
    e2 = np.tile(np.exp(-2j * theta), (nfreq, 1))

    y = np.abs(1 - np.outer(p1, e1[0]) - np.outer(p2, e2[0]))**2
    
    D = x / y
    D /= np.sum(D, axis=1, keepdims=True)

    sp2d = np.outer(a0, np.ones(ndirs)) * D / (d2r * dtheta)
    
    return sp2d, D, dirs
