"""
The following code has been adapted from the original file "wave_physiscs_functions.py" within the OPTOOLS repository.
It contains functions for spectral transformations, including Jacobians and variable changes.
"""

import numpy as np
import sys
from dispersion import group_speed_from_k, k_from_f, f_from_k


def dfdk_from_k(k, h=None):
    """
    This function computes the Jacobian df/dk from the wavenumber (k).

    Parameters: 
        k: Wavenumber [1/m]
        h: Depth in meters (optional, if None, deep water approximation is used)

    Returns:
        dfdk: Jacobian df/dk [Hz/(1/m)]
    """
    Cg = group_speed_from_k(k, depth=h, g=9.81)
    return Cg/(2*np.pi)


def spectrum_from_fth_to_kth(Efth, f, theta, h=None):
    """
    This function transforms a spectrum defined in frequency (f) and direction (theta) to a spectrum defined in wavenumber (k) and direction (theta), 
    where Efth is the f-theta spectrum and Ekth is the k-theta spectrum.

    Parameters: 
        Efth: Wave energy spectrum in frequency and direction space [m^2/Hz]
        f: Frequency [Hz]
        theta: Direction [radians]
        h: Depth [m] (optional, if None, deep water approximation is used)

    Returns:
        Ekth: Spectrum in wavenumber and direction [m^2/(Hz*1/m)]
        k: Wavenumber [1/m]
        theta: Direction [radians]
    """
    shape_Efth = np.shape(Efth)

    if len(shape_Efth) < 2:
        print('Error: spectra should be 2D')
    else:
        if shape_Efth[0] == shape_Efth[1]:
            print('Warning: same dimension for freq and theta.\n  Proceed with caution: The computation is done considering Efth = f(f,theta)')
        elif (shape_Efth[1] == len(f)) & (shape_Efth[0] == len(theta)):
            Efth = np.swapaxes(Efth, 0, 1)
        elif (shape_Efth[1] == len(theta)) & (shape_Efth[0] == len(f)):
            print('All good: Efth have the shape : (f,theta)')
        else:
            print('shape_Efth[1] : ', shape_Efth[1], ' vs ', len(f),
                  '// shape_Efth[0] :', shape_Efth[0], ' vs ', len(theta))
            print('Error: Efth should have the shape : (f,theta)')
    shape_Efth = np.shape(np.moveaxis(Efth, 0, -1))
    k = k_from_f(f, h=h)
    dfdk = dfdk_from_k(k, h=h)
    Ekth = Efth*np.moveaxis(np.broadcast_to(dfdk, shape_Efth), -1, 0)
    return Ekth, k, theta


def spectrum_from_kth_to_kxky(Ekth, k, theta):
    """
    This function transforms a spectrum defined in wavenumber (k) and direction (theta)
    to a spectrum defined in Cartesian coordinates (kx, ky), where Ekth is the k-theta spectrum.

    Parameters:
        Ekth: Wave energy spectrum in wavenumber and direction space [m^2/(Hz*1/m)]
        k: Wavenumber [1/m]
        theta: Direction [radians]

    Returns:
        Ekxky: Spectrum in Cartesian coordinates [m^2/(Hz*1/m)]
        kx: Wavenumber in x-direction [1/m]
        ky: Wavenumber in y-direction [1/m]
    """
    try:
        shape_Ekth = np.shape(Ekth)
        if len(shape_Ekth) < 2:
            print('Error: spectra should be 2D')
        else:
            if shape_Ekth[0] == shape_Ekth[1]:
                print('Warning: same dimension for k and theta.\n  Proceed with caution: The computation is done considering Ekth = f(k,theta)')
            elif ((shape_Ekth[1] == len(k)) & (shape_Ekth[0] == len(theta))) | ((shape_Ekth[1] == len(theta)) & (shape_Ekth[0] == len(k))):
                if (shape_Ekth[1] == len(k)) & (shape_Ekth[0] == len(theta)):
                    Ekth = np.swapaxes(Ekth, 0, 1)
            else:
                print('shape_Ekth[1] : ', shape_Ekth[1], ' vs ', len(k),
                      '// shape_Ekth[0] :', shape_Ekth[0], ' vs ', len(theta))
                print('Error: Ekth should have the shape : (k,theta)')

        # send k-axis to last -> in order to broadcast k along every dim
        shape_Ekth2 = np.shape(np.moveaxis(Ekth, 0, -1))

        # get only shape k,theta for the broadcast of the dimensions kx,ky
        shape_Ekth2Dkth = Ekth.shape[0:2]

        if np.max(theta) > 100:
            theta = theta*np.pi/180
        kx = np.moveaxis(np.broadcast_to(
            k, shape_Ekth2Dkth[::-1]), -1, 0) * np.cos(np.broadcast_to(theta, shape_Ekth2Dkth))
        ky = np.moveaxis(np.broadcast_to(
            k, shape_Ekth2Dkth[::-1]), -1, 0) * np.sin(np.broadcast_to(theta, shape_Ekth2Dkth))
        Ekxky = Ekth/np.moveaxis(np.broadcast_to(k, shape_Ekth2), -1, 0)
        return Ekxky, kx, ky

    except Exception as inst:
        print('in spec to kxky : ', inst, ', line number : ',
              sys.exc_info()[2].tb_lineno)


def spectrum_from_fth_to_kxky(Efth, f, theta, h=None):
    """
    This function transforms a spectrum defined in frequency (f) and direction (theta) to a spectrum defined in Cartesian coordinates (kx, ky),
    where Efth is the f-theta spectrum and Ekxky is the kx-ky spectrum.

    Parameters:
        Efth: Wave energy spectrum in frequency and direction space [m^2/Hz]
        f: Frequency [Hz]
        theta: Direction [degrees]#TODO: check if degrees or radians    
        h: Depth [m] (optional, if None, deep water approximation is used)

    Returns:
        Ekxky: Spectrum in Cartesian coordinates [m^2/(Hz*1/m)]
        kx: Wavenumber in x-direction [1/m]
        ky: Wavenumber in y-direction [1/m]
    """
    shape_Efth = np.shape(Efth)
    # print(shape_Efth)
    if len(shape_Efth) < 2:
        print('Error: spectra should be 2D')
    else:
        if shape_Efth[0] == shape_Efth[1]:
            print('Warning: same dimension for freq and theta.\n  Proceed with caution: The computation is done considering Efth = f(f,theta)')
        elif ((shape_Efth[1] == len(f)) & (shape_Efth[0] == len(theta))) | ((shape_Efth[1] == len(theta)) & (shape_Efth[0] == len(f))):
            if (shape_Efth[1] == len(f)) & (shape_Efth[0] == len(theta)):
                Efth = np.swapaxes(Efth, 0, 1)
        else:
            print('Error: Efth should have the shape : (f,theta)')

    # send f-axis to last -> in order to broadcast f along every dim
    shape_Efth2 = np.shape(np.moveaxis(Efth, 0, -1))
    # get only shape f,theta for the broadcast of the dimensions kx,ky
    shape_Efth2Dfth = Efth.shape[0:2]
    k = k_from_f(f, h=h)
    dfdk = dfdk_from_k(k, h=h)

    if np.max(theta) > 100:
        theta = theta*np.pi/180

    kx = np.moveaxis(np.broadcast_to(
        k, shape_Efth2Dfth[::-1]), -1, 0) * np.cos(np.broadcast_to(theta, shape_Efth2Dfth))
    ky = np.moveaxis(np.broadcast_to(
        k, shape_Efth2Dfth[::-1]), -1, 0) * np.sin(np.broadcast_to(theta, shape_Efth2Dfth))
    Ekxky = Efth * np.moveaxis(np.broadcast_to(dfdk / k, shape_Efth2), -1, 0)
    return Ekxky, kx, ky


def spectrum_to_kxky(typeSpec, Spec, ax1, ax2, h=None):
    """
    This function transforms a spectrum defined in either frequency and direction (typeSpec=0) or
    wavenumber and direction (typeSpec=1) to a spectrum defined in Cartesian coordinates (kx, ky).

    Parameters:
        typeSpec: Type of the spectrum (0 = spectra in frequency-theta, 1 = spec in wavenumber-theta)
        Spec: Spectrum to be transformed
        ax1: First axis (frequency or wavenumber)
        ax2: Second axis (direction)
        h: Depth [m] (Optional: if None, deep water approximation is used)

    Returns:
        Ekxky: Spectrum in Cartesian coordinates [m^2/(Hz*1/m)]
        kx: Wavenumber in x-direction [1/m]
        ky: Wavenumber in y-direction [1/m]
    """
    if typeSpec == 0:  # from f, theta
        Ekxky, kx, ky = spectrum_from_fth_to_kxky(Spec, ax1, ax2, h=h)
    elif typeSpec == 1:  # from k, theta
        Ekxky, kx, ky = spectrum_from_kth_to_kxky(Spec, ax1, ax2)
    else:
        raise ValueError('typeSpec should be 0 = (f,theta) or 1 = (k,theta)')
    return Ekxky, kx, ky


def spectrum_f_to_k(Ef, f, h=None):
    """
    This function transforms a spectrum defined in frequency (f) to a spectrum defined in wavenumber (k),
    where Ef is the frequency spectrum and Ek is the wavenumber spectrum.

    Parameters:
        Ef: Wave energy spectrum in frequency space [m^2/Hz]
        f: Frequency [Hz]
        h: Depth [m] (optional, if None, deep water approximation is used)

    Returns:
        Ek: Spectrum in wavenumber space [m^2/(Hz*1/m)]
        k: Wavenumber [1/m]       
    """
    shape_Ef = np.array(np.shape(Ef))
    ind = np.where(shape_Ef == len(f))[0]
    if len(ind) == 0:
        ValueError('Error: spectra should have an axis with same dimension as f')
    elif len(ind) > 1:
        print('Warning: same dimension for different axes.\n  Proceed with caution: The computation is done considering Ef = f(...,f)')
        if ind[-1] < (len(shape_Ef)-1):
            Ef = np.swapaxes(Ef, ind[-1], -1)
    elif len(ind) == 1:
        # pass the f axis as last dim : to broadcast
        Ef = np.swapaxes(Ef, ind, -1)

    k = k_from_f(f, h=h)
    dfdk = dfdk_from_k(k, h=h)
    shape_Ef2 = np.shape(Ef)
    Ek = np.swapaxes(Ef*np.broadcast_to(dfdk, shape_Ef2), -1, ind)
    return Ek, k


def spectrum_k_to_f(Ek, k, h=None):
    """
    This function transforms a spectrum defined in wavenumber (k) to a spectrum defined in
    frequency (f), where Ek is the wavenumber spectrum and Ef is the frequency spectrum.

    Parameters:
        Ek: Wave energy spectrum in wavenumber space [m^2/(Hz*1/m)]
        k: Wavenumber [1/m]
        h: Depth [m] (optional, if None, deep water approximation is used)
    Returns:
        Ef: Spectrum in frequency space [m^2/Hz]
        f: Frequency [Hz]
        """
    shape_Ek = np.array(np.shape(Ek))
    ind = np.where(shape_Ek == len(k))[0]
    if len(ind) == 0:
        print('Error: spectra should have an axis with same dimension as k')
    elif len(ind) > 1:
        print('Warning: same dimension for different axes.\n  Proceed with caution: The computation is done considering Ek = f(...,k)')
        if ind[-1] < (len(shape_Ek)-1):
            ind0 = int(ind[-1])
            Ek = np.swapaxes(Ek, ind0, -1)
    elif len(ind) == 1:
        ind0 = int(ind)
        # pass the f axis as last dim : to broadcast
        Ek = np.swapaxes(Ek, ind0, -1)

    f = f_from_k(k, h=h)
    dfdk = dfdk_from_k(k, h=h)
    shape_Ek2 = np.shape(Ek)
    Ef = np.swapaxes(Ek/np.broadcast_to(dfdk, shape_Ek2), -1, ind0)
    return Ef, f
