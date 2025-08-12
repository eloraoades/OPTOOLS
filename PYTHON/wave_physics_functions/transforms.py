# The following code was originally a part of the OPTOOLS project, developed by Fabrice Ardhuin & Marina de Carlo. 
# The original repo can be found at https://github.com/ardhuin/OPTOOLS.git 

# This file has been adapted from the original file "wave_physiscs_functions.py" (OPTOOLS/PYTHON) & contains 
# functions to define spectral transformations, including the conversion of wave spectra from frequency-direction 
# space to wavenumber-direction space, and vice versa.


import numpy as np
import sys
from scipy.interpolate import griddata
from wave_physics_functions.dispersion import group_speed_from_k, k_from_f, f_from_k


def wavespec_Efth_to_Ekxky(eft1s,
                           fren,
                           dfreq,
                           dirn,
                           dth,
                           dkx=0.0001, dky=0.0001,
                           nkx=250, nky=250,
                           doublesided=1,
                           verbose=0,
                           trackangle=0
                           ):
    """
    Converts E(f,theta) spectrum from buoy or model to E(kx,ky) spectrum similar to image spectrum
    using griddata interpolation. 
    2023/11/14: preliminary version, assumes dfreq is symmetric (not eaxctly true with WW3 output and waverider data) 
    
    Parameters: 
        etfs1 : spectrum in frequency and direction space
        fren : frequency axis
        dfreq : frequency step
        dirn : directional axis
        dth : direction step
        dkx : wavenumber step in x direction (default = 0.0001)
        dky : wavenumber step in y direction (default = 0.0001)
        nkx : number of wavenumber points in x direction (default =
    Returns: 
        Ekxky: spectrum
        kx: wavenumber in [1/m]  
    """
    
    [nf, nt] = np.shape(eft1s)
    tpi = 2*np.pi
    grav = 9.81

    # makes a double sided spectrum
    if doublesided == 1:
        eftn = 0.5*(eft1s+np.roll(eft1s, nt//2, axis=1))
    else:
        eftn = eft1s
    Hs1 = 4*np.sqrt(np.sum(np.sum(eftn, axis=1) * dfreq)*dth)

    # wraps around directions
    dlast = dirn[0]+360.
    dirm = np.concatenate([dirn, [dlast]])
    elast = eftn[:, 0]
    eftm1 = np.concatenate([eftn.T, [elast]]).T

    # adds zero energy in a low frequency to avoid interpolation across k=0
    ffirst = fren[0]-0.9*(fren[1]-fren[0])
    frem = np.concatenate([[ffirst], fren])
    efirst = eftm1[0, :]*0
    eftm = np.concatenate([[efirst], eftm1])

    # plt.pcolormesh(fren, dirm, np.log10(eftm).T)
    km = (2*np.pi*frem)**2/(grav*2*np.pi)   # cycles / meter
    km2 = np.tile(km.reshape(nf+1, 1), (1, nt+1))

    # eftn*df*dth = Ek*k*dk*dth -> Ek = efth *df /(k * dk)  =  efth *Cg /k
    Cg2 = np.sqrt(grav/(km2*tpi))*0.5
    Jac = Cg2/km2
    dirm2 = np.tile(dirm.T, (nf+1, 1))*np.pi/180.
    kxn = km2*np.cos(dirm2+trackangle)
    kyn = km2*np.sin(dirm2+trackangle)
    # plt.scatter(kxn,kyn,  marker='.', s = 20)
    kx = np.linspace(-nkx*dkx, (nkx-1)*dkx, nkx*2)
    ky = np.linspace(-nky*dky, (nky-1)*dky, nky*2)
    # should we transpose kx2 and ky2 ???
    kx2, ky2 = np.meshgrid(kx, ky, indexing='ij')
    Ekxky = griddata((kxn.flatten(), kyn.flatten()),
                     (eftm*Jac).flatten(), (kx2, ky2), method='nearest')
    Hs2 = 4*np.sqrt(np.sum(np.sum(Ekxky))*dkx*dky)

    # make sure energy is exactly conserved (assuming kmax is consistent with fmax
    if verbose == 1:
        print('Hs1,Hs2:', Hs1, Hs2)
    Ekxky = Ekxky * (Hs1/Hs2)**2
    return Ekxky, kx, ky, kx2, ky2


def dfdk_from_k(k, h=None):
    """
    This function computes the Jacobian df/dk from the wavenumber (k).

    Parameters: 
        k: Wavenumber [1/m]
        h: Depth in meters (optional, if None, deep water approximation is used)

    Returns:
        dfdk: Jacobian df/dk [Hz/(1/m)]
    """
    Cg = group_speed_from_k(k, h=h, g=9.81)
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
