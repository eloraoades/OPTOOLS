import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
import xarray as xr  # only used for filters: this dependency should be removed


def define_filter_annexA(Xa_c, Ya_c, DiamChelton, nkx_c, nky_c, dx_c, dy_c):
    # Uses approximation r0**2/rc**2 = R0/Hs
    twopi = 2*np.pi
    rc = DiamChelton/2
    [Xa_c2, Ya_c2] = np.meshgrid(Xa_c, Ya_c, indexing='ij')

    r0 = np.sqrt((Xa_c2)**2+(Ya_c2)**2)

    # Defines a Gaussian filter scaled with rc
    G_Lc20 = np.exp(-0.5 * r0**2 / (rc)**2)
    G_Lc2 = G_Lc20/(rc**2*twopi)

    Id = np.zeros(np.shape(G_Lc2))
    Id[nkx_c//2, nky_c//2] = 1/(dx_c*dy_c)

    #  This is the same as Jr0= A / (pi*h*Hs) * J
    Jr0 = (4*dx_c*dy_c/(np.pi*rc**2)) * (r0/rc)**2 * \
        (6 - ((2*r0/rc)**4)) * np.exp(- 4 * r0**4 / rc**4)
    #    Jr1 = fftconvolve((Id-G_Lc2),Jr0,mode='same')
    #    Filter_new = (G_Lc2+Jr1)
    Jr1 = fftconvolve(Id, Jr0, mode='same')
    # print('TEST:',np.shape(Jr0),np.sum(Jr0.flatten()),np.sum(Jr1.flatten()))

    Filter_new = (Jr1)

    phi_x0 = xr.DataArray(Filter_new,
                          dims=['x', 'y'],
                          coords={
                              "x": Xa_c,
                              "y": Ya_c,
                          },
                          )
    return phi_x0


def define_filter_J2_annexA(Xa_c, Ya_c, DiamChelton, nkx_c, nky_c, dx_c, dy_c, isplot=0):
    twopi = 2*np.pi
    rc = DiamChelton/2
    [Xa_c2, Ya_c2] = np.meshgrid(Xa_c, Ya_c, indexing='ij')
    r0 = np.sqrt((Xa_c2)**2+(Ya_c2)**2)
    
    # Defines a Gaussian filter scaled with rc
    G_Lc20 = np.exp(-0.5 * r0**2 / (rc)**2)
    G_Lc2 = G_Lc20/(rc**2*twopi)
    Id = np.zeros(np.shape(G_Lc2))
    Id[nkx_c//2, nky_c//2] = 1/(dx_c*dy_c)
    
    #     plt.plot(Xa_c,Id[:,nky_c//2]-G_Lc2[:,nky_c//2])
    # Uses approximation r0**2/rc**2 = R0/Hs
    #  This is the same as J200= -A / (4*2*pi*h*Hs) * J2
    J200 = (dx_c*dy_c/(4*np.pi*rc**2)) * \
        (2 - 16*((r0/rc)**4)) * np.exp(- 4 * r0**4 / rc**4)
    # Jr2 = fftconvolve((Id-G_Lc2),J200,mode='same')
    Jr2 = fftconvolve((Id), J200, mode='same')
    print('TES2:', np.shape(Jr2), np.sum(
        J200.flatten()), np.sum(Jr2.flatten()))
    Filter_new = (Jr2)
    # print('Sum:',np.sum(np.abs(Filter_new)*dx_c*dy_c),dx_c*dy_c)
    # Filter_new = -Filter_new/np.sum(np.abs(Filter_new)*dx_c*dy_c)
    if isplot:
        plt.figure()
        plt.plot(Xa_c, G_Lc2[:, nky_c//2], label='G_{Lc}')
        #plt.plot(Xa_c, J20[:, nky_c//2], label='J20')
        plt.plot(Xa_c, Jr2[:, nky_c//2], label='Jr2')
        plt.grid(True)
        plt.legend()
    phi_x0 = xr.DataArray(Filter_new,
                          dims=['x', 'y'],
                          coords={
                              "x": Xa_c,
                              "y": Ya_c,
                          },
                          )
    return phi_x0


def define_filter_G(Xa_c, Ya_c, DiamChelton, nkx_c, nky_c, dx_c, dy_c, isplot=0):
    twopi = 2*np.pi
    rc = DiamChelton/2
    [Xa_c2, Ya_c2] = np.meshgrid(Xa_c, Ya_c, indexing='ij')
    r0 = np.sqrt((Xa_c2)**2+(Ya_c2)**2)
    # Defines a Gaussian filter scaled with rc
    G_Lc20 = np.exp(-0.5 * r0**2 / (rc)**2)
    G_Lc2 = G_Lc20/(rc**2*twopi)
    Filter_new = (G_Lc2)
    if isplot:
        plt.figure()
        plt.plot(Xa_c, G_Lc2[:, nky_c//2], label='G_{Lc}')
        #plt.plot(Xa_c, J20[:, nky_c//2], label='J20')
        #plt.plot(Xa_c, Jr2[:, nky_c//2], label='Jr2')
        plt.grid(True)
        plt.legend()
    phi_x0 = xr.DataArray(Filter_new,
                          dims=['x', 'y'],
                          coords={
                              "x": Xa_c,
                              "y": Ya_c,
                          },
                          )
    return phi_x0
