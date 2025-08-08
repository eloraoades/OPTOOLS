from dispersion import sig_from_k, k_from_f
from spectral_transformations import dfdk_from_k
import numpy as np


def PM_spectrum_f(f, fm, g=9.81):
    """ 
    Pierson and Moskowitz (1964) spectrum in frequency space.
    
    Parameters: 
    	f : frequency [Hz]
        fm : max frequency [Hz]
		g : gravity [m/s^2] (default is 9.81 m/s^2)
    
    Returns
    -------
		E : PM spectrum in frequency space [m^2/Hz]
    """

    # There are 2 ways of writing the PM spectrum:
    #  - eq 12 of Pierson and Moskowitz (1964) where exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
    #  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
    # See Hasselmann et al. 1973 for the explanation
    alpha = 8.1*10**-3
    E = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp((-5/4)*((fm/f)**4))
    return E


def PM_spectrum_k(k, fm, D=None, g=9.81):
    # There are 2 ways of writing the PM spectrum:
    #  - eq 12 of Pierson and Moskowitz (1964) with exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
    #  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
    # See Hasselmann et al. 1973 for the explanation
    f = sig_from_k(k, D=D)/(2*np.pi)
    Ef = PM_spectrum_f(f, fm, g=g)
    dfdk = dfdk_from_k(k, D=D)

    return Ef*dfdk


def Gaussian_1Dspectrum_kx(kX, T0, sk_k0, D=None):
    kp = k_from_f(1/T0, D=D)
    sk_k = kp*sk_k0
    Z1D_Gaussian = 1/(np.sqrt(2*np.pi)*sk_k) * \
        np.exp(- 0.5*((kX-kp)**2)/(sk_k**2))

    return Z1D_Gaussian, kX, sk_k