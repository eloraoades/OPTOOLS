import numpy as np
from scipy.interpolate import griddata
from wave_physics_functions.transforms import spectrum_from_kth_to_kxky, wavespec_Efth_to_Ekxky


def surface_1D_from_Z1kx(Z1, kX, i, nx=None, dx=None, dkx=None):
    if nx is None:
        nx = len(Z1)

    if (dx is None):
        if dkx is None:
            dx = 2*np.pi/((kX[1] - kX[0])*nx)
        else:
            dx = 2*np.pi/(dkx*nx)

    if (dkx is None):
        dkx = 2*np.pi/(dx*nx)

    rng = np.random.default_rng(i)
    rg = rng.uniform(low=0.0, high=1.0, size=(nx))
    zhats = np.fft.ifftshift(np.sqrt(2*Z1*dkx)*np.exp(1j*2*np.pi*rg))
    kx0 = np.fft.ifftshift(kX)

    S1_r = np.real(np.fft.ifft(zhats, norm="forward"))
    S1_i = np.imag(np.fft.ifft(zhats, norm="forward"))

    X = np.arange(0, nx*dx, dx)  # from 0 to (nx-1)*dx with a dx step

    return S1_r, S1_i, X, dkx


def surface_2D_from_Z1kxky(Z1, kX, kY, i, nx=None, ny=None, dx=None, dy=None, dkx=None, dky=None, phase_type='uniform', verbose=False):
    # /!\ Watch out : shape(S) = (ny,nx)
    # usually when doing X,Y=np.meshgrid(x,y) with size(x)=nx and size(y)=ny => size(X)=size(Y)= (ny,nx)
    kX0 = np.unique(kX)
    kY0 = np.unique(kY)
    if nx is None:
        nx = len(kX0)
    if ny is None:
        ny = len(kY0)

    shx = np.floor(nx/2-1)
    shy = np.floor(ny/2-1)
    if verbose:
        print('from vec kX0, dkx = ', (kX0[1] - kX0[0]))
        print('from vec kY0, dky = ', (kY0[1] - kY0[0]))
    if (dx is None):
        if dkx is None:
            dx = 2*np.pi/((kX0[1] - kX0[0])*nx)
        else:
            dx = 2*np.pi/(dkx*nx)

    if (dy is None):
        if dky is None:
            dy = (2*np.pi/((kY0[1] - kY0[0])*ny))
        else:
            dy = 2*np.pi/(dky*ny)
    if (dkx is None):
        dkx = 2*np.pi/(dx*nx)
    if (dky is None):
        dky = 2*np.pi/(dy*ny)

    if verbose:
        print("variables : ")
        print(' dx = ', dx, ' ; dy = ', dy, ' ; nx = ', nx, ' ; ny = ', ny)
        print('dkx = ', dkx, ' ; dky = ', dky)

    #  Defines random phases with seed i
    rng = np.random.default_rng(i)
    if phase_type == 'uniform':
        rg = rng.uniform(low=0.0, high=1.0, size=(ny, nx))
    else:
        rg = rng.normal(0, 1, (ny, nx))
    zhats = np.fft.ifftshift(np.sqrt(2*Z1*dkx*dky)*np.exp(1j*2*np.pi*rg))
    ky2D = np.fft.ifftshift(kY)
    kx2D = np.fft.ifftshift(kX)

    #     real part
    S2_r = np.real(np.fft.ifft2(zhats, norm="forward"))
    #     also computes imaginary part (useful for envelope calculations)
    S2_i = np.imag(np.fft.ifft2(zhats, norm="forward"))

    X = np.arange(0, np.floor(nx*dx), dx)  # from 0 to (nx-1)*dx with a dx step
    Y = np.arange(0, np.floor(ny*dy), dy)

    return S2_r, S2_i, X, Y, kX0, kY0, i, dkx, dky


def surface_from_Efth_noxr(eftn, fren, df_vec, th_vec, dth, seed=0, nx=2048, ny=2048, dx=10, dy=10, D=10000., iswvnb=0, verbose=1):
    [nf, nt] = np.shape(eftn)
    tpi = 2*np.pi
    grav = 9.81
    Hs1 = 4*np.sqrt(np.sum(np.sum(eftn, axis=1) * df_vec)*dth)

    # wraps around directions
    dlast = th_vec[0]+360.
    dirm = np.concatenate([th_vec, [dlast]])
    elast = eftn[:, 0]
    eftm = np.concatenate([eftn.T, [elast]]).T

    kn = (2*np.pi*fren)**2/(grav)   # rad / meter
    kn2 = np.tile(kn.reshape(nf, 1), (1, nt+1))

    # eftn*df*dth = Ek*k*dk*dth -> Ek = efth *df /(k * dk)  =  efth *Cg /k
    Cg2 = np.sqrt(grav/(kn2))*0.5
    Jac = Cg2/(kn2*tpi)
    dirm2 = np.tile(dirm.T, (nf, 1))*np.pi/180.
    kxn = kn2*np.cos(dirm2)
    kyn = kn2*np.sin(dirm2)

    dkx = 2*np.pi/(dx*nx)
    dky = 2*np.pi/(dy*ny)
    kX0 = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))*2*np.pi
    kY0 = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))*2*np.pi
    kX, kY = np.meshgrid(kX0, kY0)  # , indexing='ij')
    Ekxky = griddata((kxn.flatten(), kyn.flatten()),
                     (eftm*Jac).flatten(), (kX, kY), method='nearest')
    Hs2 = 4*np.sqrt(np.sum(np.sum(Ekxky))*dkx*dky)
    Ekxky = Ekxky * (Hs1/Hs2)**2
    E_total = np.sum(Ekxky.flatten())*dkx*dky

    # Note that in DeCarlo et al. Qkk is defined from double-sided spectrum , here we compute from single-sided spectrum, hence the factor 0.5
    Qkk = np.sqrt(np.sum(Ekxky.flatten()**2)*dkx*dky*0.5)/E_total
    Hskk = 4*np.sqrt(E_total)

    # make sure energy is exactly conserved (assuming kmax is consistent with fmax)
    if verbose == 1:
        print('Hs1,Hs2:', Hs1, Hs2, Hskk)
    S2_r, S2_i, X, Y, kX0, kY0, rg, dkx, dky = surface_2D_from_Z1kxky(
        Ekxky, kX, kY, seed)
    return S2_r, S2_i, X, Y, rg, kX0, kY0, Ekxky, dkx, dky, Hskk, Qkk


def surface_from_Efth(Efth, f_vec, df_vec, th_vec, dth, seed=0, nx=2048, ny=2048, dx=10, dy=10, D=10000., iswvnb=0):
    import xarray as xr
    g = 9.81
    #
    # Here we start by adding last and first value at the border of the spectrum in order to deal with the -180/180 gap
    spec = xr.DataArray(Efth,
                        dims=['n_phi', 'nk'],
                        coords={
                            "phi_vector": (["n_phi"], th_vec),
                            "k_vector": (["nk"], f_vec),
                        },
                        )

    spec_bis = xr.concat(
        [spec.isel(n_phi=-1), spec, spec.isel(n_phi=0)], dim="n_phi")
    
    # --- change the first and last new values to have a 2pi revolution ---------------
    A = np.concatenate([[-360], np.zeros((spec.sizes['n_phi'])), [360]])
    factor = xr.DataArray(A, dims="n_phi")
    spec_bis['phi_vector'].values = spec_bis['phi_vector']+factor
    spec_bis = spec_bis.interpolate_na(dim='n_phi')

    # Then, turning the spectrum to cartesian coordinates in order
    # to apply the Inverse Fourier to it and get the surface
    #
    # -- get cartesian spectrum ------

    # -- GET THE INTERPOLATION GRID for the spectrum -----
    # -- get kx,ky values we want from the surface we want ---------
    dkx = 2*np.pi/(dx*nx)
    dky = 2*np.pi/(dy*ny)
    kX0 = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))*2*np.pi
    kY0 = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))*2*np.pi

    kX, kY = np.meshgrid(kX0, kY0)  # , indexing='ij')
    kK = (np.sqrt(kX**2+kY**2))
    kPhi = np.arctan2(kY, kX)*180/np.pi
    kF = kK
    kPhi[kPhi < 0] = kPhi[kPhi < 0]+360

    # create a dataArray with the new (i.e. wanted) values of F written in a cartesian array
    kF2 = xr.DataArray(kF, coords=[("ky", kY0), ("kx", kX0)])

    kPhi2 = xr.DataArray(kPhi, coords=[("ky", kY0), ("kx", kX0)])
    FPhi2s = xr.Dataset(
        {'kF': kF2,
         'kPhi': kPhi2}
    ).stack(flattened=["ky", "kx"])

    if iswvnb:
        Ekxky0, kx, ky = spectrum_from_kth_to_kxky(np.squeeze(spec_bis.compute(
        ).data),  spec_bis['k_vector'].compute().data, spec_bis["phi_vector"].compute().data)
        Ekxky = xr.DataArray(Ekxky0, dims=("nf", "n_phi"), coords={
                             "nf": f_vec, "n_phi": spec_bis['phi_vector']})
        B = Ekxky.interp(nf=FPhi2s.kF, n_phi=FPhi2s.kPhi,
                         kwargs={"fill_value": 0})
        B.name = 'Ekxky_new'
        B0 = B.reset_coords(("nf", "n_phi"))
        Ekxky_for_surf = B0.Ekxky_new.unstack(dim='flattened')
        E_total = np.sum(Ekxky_for_surf.values.flatten())*dkx*dky
        # compared to eq. 16 in De Carlo et al. (2023) the factor 0.5 corrects for single sided spec
        Qkk = np.sqrt(np.sum(Ekxky_for_surf.values.flatten()**2)
                      * dkx*dky*0.5)/E_total
    else:
        Ekxky, kx, ky, kx2, ky2 = wavespec_Efth_to_Ekxky(
            Efth, f_vec, df_vec, th_vec, dth, dkx=dkx/(2*np.pi), dky=dky/(2*np.pi), nkx=ny//2, nky=nx//2, doublesided=0)
        Ekxky_for_surf = Ekxky/(2*np.pi)**2
        E_total = np.sum(Ekxky_for_surf.flatten())*dkx*dky

        # compared to eq. 16 in De Carlo et al. (2023) the factor 0.5 corrects for single sided spec
        Qkk = np.sqrt(np.sum(Ekxky_for_surf.flatten()**2)*dkx*dky*0.5)/E_total

    # --- compute associated Kf, Phi(in deg) ---------
    #        if iswvnb:
    #        else:
    #                kF = f_from_k(kK,D=D)

    # computes Hs and Qkk
    Hskk = 4*np.sqrt(E_total)

    S2_r, S2_i, X, Y, kX0, kY0, rg, dkx, dky = surface_2D_from_Z1kxky(
        Ekxky_for_surf, kX, kY, seed)

    return S2_r, S2_i, X, Y, rg, kX0, kY0, Ekxky_for_surf, dkx, dky, Hskk, Qkk


def surface_from_Z1kxky(Z1, kX, kY, nx=None, ny=None, dx=None, dy=None, dkx=None, dky=None):
    kX0 = np.unique(kX)
    kY0 = np.unique(kY)
    if nx == None:
        nx = Z1.shape[1]
    if ny == None:
        ny = Z1.shape[0]
    shx = np.floor(nx/2-1)
    shy = np.floor(ny/2-1)
    if (dx == None):
        if dkx == None:
            dx = 2*np.pi/((kX0[1] - kX0[0])*nx)
        else:
            dx = 2*np.pi/(dkx*nx)

    if (dy == None):
        if dky == None:
            dy = np.floor(2*np.pi/((kY0[1] - kY0[0])*ny))
        else:
            dy = 2*np.pi/(dky*ny)

    if (dkx == None):
        dkx = 2*np.pi/(dx*nx)
    if (dky == None):
        dky = 2*np.pi/(dy*ny)

    rg = np.random.normal(0, 1, (ny, nx))
    zhats = np.roll(np.sqrt(2*Z1*dkx*dky)*np.exp(1j*2*np.pi*rg),
                    (-int(shy), -int(shx)), axis=(0, 1))
    # checks that ky2D(1,1)=0 ...
    ky2D = np.roll(kY, (-int(shy), -int(shx)), axis=(0, 1))
    # checks that kx2D(1,1)=0 ...
    kx2D = np.roll(kX, (-int(shy), -int(shx)), axis=(0, 1))

    S1 = np.real(np.fft.ifft2(zhats))*(nx*ny)
    X = np.arange(0, nx*dx, dx)  # from 0 to (nx-1)*dx with a dx step
    Y = np.arange(0, ny*dy, dy)

    return S1, X, Y


def surface_from_Z1kxky_uniform_phase(Z1, kX, kY, i, nx=None, ny=None, dx=None, dy=None, dkx=None, dky=None):
    # /!\ Watch out : shape(S) = (ny,nx)
    # usually when doing X,Y=np.meshgrid(x,y) with size(x)=nx and size(y)=ny => size(X)=size(Y)= (ny,nx)
    kX0 = np.unique(kX)
    kY0 = np.unique(kY)
    if nx == None:
        nx = Z1.shape[1]
    if ny == None:
        ny = Z1.shape[0]
    shx = np.floor(nx/2-1)
    shy = np.floor(ny/2-1)
    if (dx == None):
        if dkx == None:
            dx = 2*np.pi/((kX0[1] - kX0[0])*nx)
        else:
            dx = 2*np.pi/(dkx*nx)

    if (dy == None):
        if dky == None:
            dy = np.floor(2*np.pi/((kY0[1] - kY0[0])*ny))
        else:
            dy = 2*np.pi/(dky*ny)

    if (dkx == None):
        dkx = 2*np.pi/(dx*nx)
    if (dky == None):
        dky = 2*np.pi/(dy*ny)

    rng = np.random.default_rng(i)
    rg = rng.uniform(low=0.0, high=1.0, size=(ny, nx))
    # ,(-int(shy),-int(shx)),axis=(0,1))
    zhats = np.fft.ifftshift(np.sqrt(2*Z1*dkx*dky)*np.exp(1j*2*np.pi*rg))
    # ,(-int(shy),-int(shx)),axis=(0,1)) # checks that ky2D(1,1)=0 ...
    ky2D = np.fft.ifftshift(kY)
    # ,(-int(shy),-int(shx)),axis=(0,1)) # checks that kx2D(1,1)=0 ...
    kx2D = np.fft.ifftshift(kX)

    #    real part
    S1 = np.real(np.fft.ifft2(zhats, norm="forward"))
    #     also computes imaginary part (useful for envelope calculations)
    S2 = np.imag(np.fft.ifft2(zhats, norm="forward"))

    X = np.arange(0, nx*dx, dx)  # from 0 to (nx-1)*dx with a dx step
    Y = np.arange(0, ny*dy, dy)

    return S1, S2, X, Y, kX0, rg, dkx, dky
