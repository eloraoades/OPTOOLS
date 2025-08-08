import numpy as np
import scipy.interpolate as spi  # function griddata

from wave_physics_functions.spectra import Gaussian_1Dspectrum_kx, PM_spectrum_k, define_Gaussian_spectrum_kxky, define_spectrum_PM_cos2n
from wave_physics_functions.transforms import spectrum_to_kxky
from wave_physics_functions.dispersion import sig_from_k

#TODO: Add documentation for functions in this file 

def def_spectrumG_for_surface_1D(nx=2048, dx=10, T0=10, Hs=4, sk_k0=0.1, h=None, verbose=False):
    dkx = 2*np.pi/(dx*nx)

    kX = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))*2*np.pi

    # --- only Gaussian -------------
    Z1_Gaussian, kX, sk = Gaussian_1Dspectrum_kx(kX, T0, sk_k0, h=h)
    Z1 = (Hs/4)**2*Z1_Gaussian
    sumZ1 = 4*np.sqrt(sum(Z1.flatten()*dkx))
    if verbose:
        print('Hs for Gaussian : ', sumZ1)

    return Z1, kX, sk


# Hs=4,sk_k0=0.1
def def_spectrumPM_for_surface_1D(nx=2048, dx=10, T0=10, h=None, verbose=False):
    dkx = 2*np.pi/(dx*nx)

    kX = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))*2*np.pi

    # --- only PM -------------
    Z1_PM = PM_spectrum_k(kX, 1/T0, h=h)
    # Z1_Gaussian,kX,sk = Gaussian_1Dspectrum_kx(kX,T0,sk_k0,h=h)
    # Z1 =(Hs/4)**2*Z1_Gaussian
    sumZ1 = 4*np.sqrt(sum(Z1_PM[np.isfinite(Z1_PM)].flatten()*dkx))
    if verbose:
        print('Hs for Gaussian : ', sumZ1)
    Z1_PM[np.isnan(Z1_PM)] = 0
    return Z1_PM, kX


def def_spectrumJONSWAP_for_surface_1D(nx=2048, dx=10, T0=10, h=None, gammafac=3.3, sigA=0.07, sigB=0.09, verbose=False):
    dkx = 2*np.pi/(dx*nx)

    kX = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))*2*np.pi
    fX = sig_from_k(kX, h=h)/(2*np.pi)
    Z1_PM = PM_spectrum_k(kX, 1/T0, h=h)
    Z1_PM[np.isnan(Z1_PM)] = 0
    fp = 1/T0
    sigAB = np.where(fX < fp, sigA, sigB)
    JSfactor = gammafac**np.exp((-(fX-fp)**2)/(2*sigAB**2*fp**2))
    Z1_JS = Z1_PM*JSfactor
    Z1_JS[np.isnan(Z1_JS)] = 0
    sumZ1 = 4*np.sqrt(sum(Z1_JS[np.isfinite(Z1_JS)].flatten()*dkx))
    if verbose:
        sumZ1 = 4*np.sqrt(sum(Z1_JS[np.isfinite(Z1_JS)].flatten()*dkx))
        print('Hs for Jonswap : ', sumZ1)
    return Z1_JS, kX


def def_spectrum_for_surface(
        nx=2048, ny=2048, 
        dx=10, dy=10, 
        theta_m=30, 
        h=1000, 
        T0=10, 
        Hs=4, 
        sk_theta=0.001, sk_k=0.001,
        nk=1001, nth=36, 
        klims=(0.0002, 0.2), 
        n=4, 
        typeSpec='Gaussian', 
        verbose=False
    ):
    
    dkx = 2*np.pi/(dx*nx)
    dky = 2*np.pi/(dy*ny)

    kX0 = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))*2*np.pi
    kY0 = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))*2*np.pi
    kX, kY = np.meshgrid(kX0, kY0)

    if typeSpec == 'Gaussian':
        if verbose:
            print('Gaussian spectrum selected. Available options:\n - Hs, sk_theta, sk_k. \nWith (sk_k, sk_theta) the sigma values for the k-axis along the main direction and perpendicular to it respectively \n Other options (common to all spectrum types) are : nx, ny, dx, dy, T0, theta_m, h')

        Z1_Gaussian0, kX, kY = define_Gaussian_spectrum_kxky(
            kX, kY, T0, theta_m*np.pi/180, sk_theta, sk_k, h=h)

        Z1 = (Hs/4)**2*Z1_Gaussian0/np.sum(Z1_Gaussian0.flatten()*dkx*dky)
        sumZ1 = 4*np.sqrt(sum(Z1.flatten()*dkx*dky))
        if verbose:
            print('Hs for Gaussian : ', sumZ1)

    elif typeSpec == 'PM':
        if verbose:
            print('Pierson-Moskowitz* cos(theta)^(2*n) spectrum selected. Available options:\n - nk, nth, klims, n. \nWith n the exponent of the directional distribution: cos(theta)^(2*n)\n Other options (common to all spectrum types) are : nx, ny, dx, dy, T0, theta_m, h')
        k = np.linspace(klims[0], klims[1], nk)
        thetas = np.linspace(0, 360*(nth-1)/nth, nth)*np.pi/180.

        Ekth, k, th = define_spectrum_PM_cos2n(
            k, thetas, T0, theta_m*np.pi/180., h=h, n=n)
        Ekxky, kx, ky = spectrum_to_kxky(1, Ekth, k, thetas, h=h)

        Z1 = spi.griddata((kx.flatten(), ky.flatten()),
                          Ekxky.flatten(), (kX, kY), fill_value=0)
        sumZ1 = 4*np.sqrt(sum(Z1.flatten()*dkx*dky))
        if verbose:
            print('Hs for Pierson Moskowitz : ', sumZ1)

    return Z1, kX, kY, dkx, dky
