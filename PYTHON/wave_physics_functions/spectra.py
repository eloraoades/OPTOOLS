# The following code was originally a part of the OPTOOLS project, developed by Fabrice Ardhuin & Marina de Carlo. 
# The original repo can be found at https://github.com/ardhuin/OPTOOLS.git 

# This file has been adapted from the original file "wave_physiscs_functions.py" (OPTOOLS/PYTHON) & contains 
# functions to define 1D (non-directional) and 2D (directional) wave spectra, including Pierson-Moskowitz and Gaussian spectra.

import numpy as np
from wave_physics_functions.dispersion import sig_from_k, k_from_f
from wave_physics_functions.transforms import dfdk_from_k
import numpy as np


def PM_spectrum_f(f, fm, g=9.81):
    """ 
    Pierson and Moskowitz (1964) spectrum in frequency space.

    Parameters: 
        f : frequency [Hz]
        fm : max frequency [Hz]
                g : gravity [m/s^2] (default is 9.81 m/s^2)

    Returns
                E : PM spectrum in frequency space [m^2/Hz]
    """
    # There are 2 ways of writing the PM spectrum:
    #  - eq 12 of Pierson and Moskowitz (1964) where exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
    #  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
    # See Hasselmann et al. 1973 for the explanation
    alpha = 8.1*10**-3
    E = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp((-5/4)*((fm/f)**4))
    return E


def PM_spectrum_k(k, fm, h=None, g=9.81):
    """    
    Pierson and Moskowitz (1964) spectrum in wavenumber space.

    Parameters:
        k: Wavenumber [1/m]
        fm: Max frequency [Hz]
        h: Water depth [m] (optional, if None, deep water approximation is used)
        g: Gravity [m/s^2] (default is 9.81 m/s^2)

    Returns:
        Ek: PM spectrum in wavenumber space [m^2/Hz]
    """
    # There are 2 ways of writing the PM spectrum:
    #  - eq 12 of Pierson and Moskowitz (1964) with exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
    #  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
    # See Hasselmann et al. 1973 for the explanation
    f = sig_from_k(k, h=h)/(2*np.pi)
    Ef = PM_spectrum_f(f, fm, g=g)
    dfdk = dfdk_from_k(k, h=h)

    return Ef*dfdk


def Gaussian_1Dspectrum_kx(kX, T0, sk_k0, h=None):
    """
    This function defines a 1D Gaussian spectrum in wavenumber space.
    
    Parameters: 
        kX: Wavenumber in x direction [1/m]
        T0: Peak period [s]
        sk_k0: Standard deviation in wavenumber [1/m]
        h: Depth in meters (optional, if None, deep water approximation is used)
    
    Returns:
        Z1D_Gaussian: 1D Gaussian spectrum in wavenumber space [m^2/Hz]
        kX: Wavenumber in x direction [1/m] #TODO: is this needed? it's an input parameter
        sk_k: Standard deviation in wavenumber [1/m] #TODO: is this needed? 
        """

    kp = k_from_f(1/T0, h=h)
    sk_k = kp*sk_k0
    Z1D_Gaussian = 1/(np.sqrt(2*np.pi)*sk_k) * \
        np.exp(- 0.5*((kX-kp)**2)/(sk_k**2))

    return Z1D_Gaussian, kX, sk_k


def define_spectrum_PM_cos2n(k, th, T0, theta_m, n=4, h=None):
    """ 
    This function defines a 2D Pierson-Moskowitz spectrum with a cosine squared directional distribution.

    Parameters: 
        k: Wavenumber [1/m]
        th: Directional angles [radians]
        T0: Peak wave period [s]
        theta_m: Mean direction [radians] #TODO: degrees or radians ? 
        n: Exponent for the cosine squared distribution (default = 4)
        h: Depth in meters (optional, if None, deep water approximation is used)

    Returns: 
        Ekth: 2D Pierson-Moskowitz spectrum in wavenumber-theta [m^2/Hz]
        k: Wavenumber [1/m] #TODO: is this needed ?
        th: Directional angles [radians] #TODO: is this needed ?
    """
    Ek = PM_spectrum_k(k, 1/T0, h=h)
    dth = th[1]-th[0]
    Eth = np.cos(th-theta_m)**(2*n)
    II = np.where(np.cos(th-theta_m) < 0)[0]
    Eth[II] = 0
    sth = sum(Eth*dth)
    Ekth = np.broadcast_to(Ek, (len(th), len(k))) * \
        np.broadcast_to(Eth, (len(k), len(th))).T / sth
    return Ekth, k, th 


def define_Gaussian_spectrum_kxky(kX, kY, T0, theta_m, sk_theta, sk_k, h=None):
    """
    This function defines a 2D Gaussian spectrum in k-space.

    Parameters: 
        kX: Wavenumber in x direction [1/m]
        kY: Wavenumber in y direction [1/m]
        T0: Peak period [s]
        theta_m: Mean direction [radians]
        sk_theta: Standard deviation in direction [radians]
        sk_k: Standard deviation in wavenumber [1/m]
        h: Depth in meters (optional, if None, deep water approximation is used)

    Returns:
        Z1_Gaussian: 2D Gaussian spectrum in k-space [m^2/Hz]
        kX: Wavenumber in x direction [1/m] #TODO: is this needed ?
        kY: Wavenumber in y direction [1/m] #TODO: is this needed ?
    """
    if (len(kX.shape) == 1) & (len(kY.shape) == 1):
        kX, kY = np.meshgrid(kX, kY)
    elif (len(kX.shape) == 1) | (len(kY.shape) == 1):
        print('Error : kX and kY should either be: \n      - both vectors of shapes (nx,) and (ny,) \n  OR  - both matrices of shape (ny,nx)')
        print(
            r"/!\ Proceed with caution /!\ kX and kY have been flattened to continue running")
        kX = kX.flatten()
        kY = kY.flatten()
    kp = k_from_f(1/T0, h=h)
    
    # rotation of the grid => places kX1 along theta = theta_m
    kX1 = kX*np.cos(theta_m)+kY*np.sin(theta_m)
    kY1 = -kX*np.sin(theta_m)+kY*np.cos(theta_m)

    Z1_Gaussian = 1/(2*np.pi*sk_theta*sk_k) * np.exp(- 0.5 *
                                                     ((((kX1-kp)**2)/((sk_k)**2))+kY1**2/sk_theta**2))

    return Z1_Gaussian, kX, kY
