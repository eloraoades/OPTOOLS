import numpy as np
from dispersion import k_from_f
from spectra_1d import PM_spectrum_k


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
            '/!\ Proceed with caution /!\ kX and kY have been flattened to continue running')
        kX = kX.flatten()
        kY = kY.flatten()
    kp = k_from_f(1/T0, h=h)
    
    # rotation of the grid => places kX1 along theta = theta_m
    kX1 = kX*np.cos(theta_m)+kY*np.sin(theta_m)
    kY1 = -kX*np.sin(theta_m)+kY*np.cos(theta_m)

    Z1_Gaussian = 1/(2*np.pi*sk_theta*sk_k) * np.exp(- 0.5 *
                                                     ((((kX1-kp)**2)/((sk_k)**2))+kY1**2/sk_theta**2))

    return Z1_Gaussian, kX, kY
