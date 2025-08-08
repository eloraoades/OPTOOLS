import numpy as np


def phase_speed_from_k(k, h=None, g=9.81):
    """
    This function computes the phase speed using wavenumber (k) and depth (h).
    
    Parameters:
        k: Wavenumber [1/m]
        h: Depth in meters (optional, if None, deep water approximation is used)
        g: Gravitational acceleration in m/s^2 (default is 9.81)    
    
    Returns: 
        C: Phase speed [m/s]
    """
    if h is None:
        # print("Deep water approximation")
        C = np.sqrt(g/k)
    else:
        # print("General case")
        C = np.sqrt(g*np.tanh(k*h)/k)
    return C


def phase_speed_from_sig_k(sig, k):
    """
    This function computes the phase speed from angular frequency (sig) and wavenumber (k).

    Parameters: 
        sig: Angular frequency [rad/s]
        k: Wavenumber [1/m]

    Returns:
        C: Phase speed [m/s]
    """
    return sig/k


def group_speed_from_k(k, h=None, g=9.81):
    """
    This function computes the group speed (Cg) using wavenumber (k) and depth (h).

    Parameters: 
        k: Wavenumber [1/m]
        h: Depth [m] (Optional: if None, deep water approximation is used)
        g: Gravitational acceleration in m/s^2 (default is 9.81)

    Returns:
        Cg: Group speed [m/s]
    """
    C = phase_speed_from_k(k, h=h, g=g)
    if h is None:
        # print("Deep water approximation")
        Cg = C/2
    else:
        # print("General case")
        Cg = C*(0.5 + ((k*h)/(np.sinh(2*k*h))))
    return Cg


def sig_from_k(k, h=None, g=9.81):
    """
    This function computes angular frequency, sigma, from the wavenumber (k) and depth (h).
    
    Parameters:
        k: Wavenumber [1/m]
        h: Depth in meters (optional, if None, deep water approximation is used)
        g: Gravitational acceleration in m/s^2 (default is 9.81
        
    Returns:
        sig: Angular frequency [rad/s]
    """
    if h is None:
        # print("Deep water approximation")
        sig = np.sqrt(g*k)
    else:
        # print("General case")
        sig = np.sqrt(g*k*np.tanh(k*h))
    return sig


def f_from_sig(sig):
    """
    This funcion computes wave frequency (f) from angular frequency (sig).

    Parameters:
        sig: Angular frequency [rad/s]

    Returns:
        f: Frequency [Hz]
    """
    return sig/(2*np.pi)


def sig_from_f(f):
    """
    This function computes the angular frequency (sig) from the frequency (f).
    
    Parameters:
        f: Frequency [Hz]

    Returns:
        sig: Angular frequency [rad/s]
    """
    return 2*np.pi*f


def f_from_k(k, h=None, g=9.81):
    """
    This function computes the frequency (f) from the wavenumber (k) and depth (h). 
        
    Parameters:
        k: Wavenumber [1/m]
        h: Depth in meters (optional, if None, deep water approximation is used)
        g: Gravitational acceleration in m/s^2 (default is 9.81)
    
    Returns:
        f: Frequency [Hz]"""
    sig = sig_from_k(k, h=h, g=g)
    return sig/(2*np.pi)


def period_from_sig(sig):
    """
    This function computes the period (T) from the angular frequency (sig).
    
    Parameters:
        sig: Angular frequency [rad/s]
    
    Returns:
        T: Period [s]"""
    return (2*np.pi)/sig


def period_from_wvl(wvl, h=None):
    """
    This function computes the period (T) from the wavelength (wvl).
    It uses the dispersion relation to find the wavenumber (k) and then computes the period (T) from the wavenumber (k). 
    
    Parameters:
        wvl: Wavelength [m]
        h: Depth in meters (optional, if None, deep water approximation is used)

    Returns:
        T: Period [s]    
    """
    k = (2*np.pi)/wvl
    sig = sig_from_k(k, h=h) 
    T = period_from_sig(sig)
    return T


def k_from_f(f, h=None, g=9.81):
    """
    This function computes the wavenumber k from the frequency f and depth h.
    It uses the linear dispersion relation for deep water waves or shallow water waves.
    If h is None, it assumes deep water approximation.

    Parameters: 
        f: Frequency [Hz]
        h: Depth in meters (optional, if None, deep water approximation is used)
        g: Gravitational acceleration in m/s^2 (default is 9.81)
    
    Returns:
        k: Wavenumber [1/m]
    """
    eps = 0.000001
    sig = np.array(2*np.pi*f) # angular frequency 
    if h is None:
        # print("Deep water approximation")
        k = sig**2/g
    else:
        Y = h*sig**2/g
        X = np.sqrt(Y)
        I = 1
        F = 1.
        while (abs(np.max(F)) > eps):
            H = np.tanh(X)
            F = Y-(X*H)
            FD = -H-(X/(np.cosh(X)**2))
            X = X-(F/FD)

        k = X/h

    return k  # wavenumber
