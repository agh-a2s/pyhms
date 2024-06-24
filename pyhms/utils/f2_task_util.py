from pyhms.core.individual import Individual
from scipy import interpolate
import numpy as np

def f2_translate_genome(ind: Individual):
    xu_ranges = ((0.0, 20.0), (10.0, 100.0))

    xu_shape = (11, 10)
    
    ind_shape = (5, 4)
    ind_size = ind_shape[0]*ind_shape[1]

    genome = ind.genome

    A1_grid = genome[:ind_size].reshape(ind_shape)
    A2_grid = genome[ind_size:2*ind_size].reshape(ind_shape)
    k_grid = genome[2*ind_size:3*ind_size].reshape(ind_shape)
    Td = genome[3*ind_size:]

    x = np.linspace(xu_ranges[0][0], xu_ranges[0][1], ind_shape[0])
    u = np.linspace(xu_ranges[1][0], xu_ranges[1][1], ind_shape[1])

    a1_int = interpolate.RegularGridInterpolator((x,u), A1_grid, method='cubic')
    a2_int = interpolate.RegularGridInterpolator((x,u), A2_grid, method='cubic')
    k_int = interpolate.RegularGridInterpolator((x,u), k_grid, method='cubic')

    xi, ui = np.mgrid[xu_ranges[0][0]:xu_ranges[0][1]:complex(0, xu_shape[0]), xu_ranges[1][0]:xu_ranges[1][1]:complex(0, xu_shape[1])]

    A1_grid_new = np.clip(a1_int((xi, ui)), 0.0, 1.0)
    A2_grid_new = np.clip(a2_int((xi, ui)), 0.0, 1.0)
    k_grid_new = np.clip(k_int((xi, ui)), 0.0, 1.0)

    return np.concatenate([A1_grid_new.flatten(), A2_grid_new.flatten(), k_grid_new.flatten(), Td])
