# The following code was originally a part of the OPTOOLS project, developed by Fabrice Ardhuin & Marina de Carlo.
# The original repo can be found at https://github.com/ardhuin/OPTOOLS.git

# This file has been adapted from the original file "wave_physiscs_functions.py" (OPTOOLS/PYTHON) & contains
# functions to define 2D (directional) wave spectra, including Pierson-Moskowitz and Gaussian spectra.

import numpy as np
from dispersion import k_from_f, f_from_k


def infos_from_wvl(L, h=None):
    """
    Print information about a wave given its wavelength.
    
    Parameters:
        L: Wavelength in meters.
        h: float, optional
    """
    k = 2*np.pi/L
    f = f_from_k(k, h=h)
    T = 1/f

    print('From a wavelength of ', L, ' m : -----------------------')
    print('     - wavenumber k   =   '+f'{k:.4f}'.rjust(6)+' rad/m')
    if h is None:
        print('   With the infinite depth approximation :')
    else:
        print('   With a depth of ', h, ' m')
    print('     - frequency f    =   '+f'{f:.3f}'.rjust(6)+' Hz')
    print('     - period T       =   '+f'{T:.2f}'.rjust(6)+' s')
    print('--------------------------------------------------------')


def infos_from_wvnb(k, h=None):
    """
    Print information about a wave given its wavenumber.
    
    Parameters:
        k: Wavenumber in radians per meter.
        h: Depth [float, optional]
    """
    L = 2*np.pi/k
    f = f_from_k(k, h=h)
    T = 1/f

    print('From a wavenumber of ', k, ' rad/m : -----------------------')
    print('     - wavelength L   =   '+f'{L:.1f}'.rjust(6)+' m')
    if h is None:
        print('   With the infinite depth approximation :')
    else:
        print('   With a depth of ', h, ' m')
    print('     - frequency f    =   '+f'{f:.3f}'.rjust(6)+' Hz')
    print('     - period T       =   '+f'{T:.2f}'.rjust(6)+' s')
    print('--------------------------------------------------------')


def infos_from_T0(T, h=None):
    """
    Print information about a wave given its period.

    Parameters: 
        T: Period in seconds.
        h: Depth [float, optional]    
    """
    f = 1/T
    k = k_from_f(f, h=h)
    L = 2*np.pi/k

    print('From a period of ', T, ' s : -----------------------')
    print('     - frequency f    =   '+f'{f:.3f}'.rjust(6)+' Hz')
    if h is None:
        print('   With the infinite depth approximation :')
    else:
        print('   With a depth of ', h, ' m')
    print('     - wavelength L   =   '+f'{L:.1f}'.rjust(6)+' m')
    print('     - wavenumber k   =   '+f'{k:.4f}'.rjust(6)+' rad/m')
    print('--------------------------------------------------------')


def infos_from_f(f, h=None):
    """
    Print information about a wave given its frequency.
    Parameters:
        f: Frequency in Hertz.
        h: Depth [float, optional]
    """
    T = 1/f
    k = k_from_f(f, h=h)
    L = 2*np.pi/k

    print('From a frequency of ', f, ' Hz : -----------------------')
    print('     - period T       =   '+f'{T:.2f}'.rjust(6)+' s')
    if h is None:
        print('   With the infinite depth approximation :')
    else:
        print('   With a depth of ', h, ' m')
    print('     - wavelength L   =   '+f'{L:.1f}'.rjust(6)+' m')
    print('     - wavenumber k   =   '+f'{k:.4f}'.rjust(6)+' rad/m')
    print('--------------------------------------------------------')
