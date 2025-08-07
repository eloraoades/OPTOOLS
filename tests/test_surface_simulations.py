# tests/test_surface_sim.py  (create this file)
import numpy as np
import PYTHON.surface_simulation_functions as surf

def test_surface_shape_and_finiteness():
    rng = np.random.default_rng(0)
    nx, ny = 64, 64
    kx = rng.normal(size=(nx, ny))
    ky = rng.normal(size=(nx, ny))
    Z1 = rng.normal(size=(nx, ny))
    surf_map = surf.surface_2D_from_Z1kxky(Z1, kx, ky)
    assert surf_map.shape == (nx, ny)
    assert np.isfinite(surf_map).all()
