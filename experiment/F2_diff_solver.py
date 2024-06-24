import scipy.integrate as scint
from scipy import interpolate
import numpy as np

def prepare_costfun(T, xu_shape, xu_ranges, y):
    '''
    Prepare cost function for genetic algorithm optimization.
    t: dict[float: np.ndarray] - dictionary of time vectors loaded from Frantisek data
    xu_shape: tuple - shape of possible state combinations. Exempli gratia (x, u) -> (11, 10)
    xu_ranges: tuple - ranges for 2 state variables. Exempli gratia (x, u) -> ((0.0, 20.0) (10.0, 100.0))
    y: dict[float: np.ndarray] - dictionary of output vectors loaded from Frantisek data and used for comparison
    '''
    x0 =  0.0
    xu_size = xu_shape[0]*xu_shape[1]

    def rhs(t, x, *args):
        u, a1_int, a2_int, k_int, Td = args

        if x[0] > xu_ranges[0][1]:
            xu = np.array([xu_ranges[0][1], u])
        elif x[0] < xu_ranges[0][0]:
            xu = np.array([xu_ranges[0][0], u])
        else:
            xu = np.array([x[0], u])

        if t < Td:
            return [x[1], ((-x[0] - a1_int(xu) * x[1]) / a2_int(xu))[0]]
        else:
            return [x[1], ((-x[0] - a1_int(xu) * x[1] + k_int(xu) * u) / a2_int(xu))[0]]

    def solve_direct(x0, u, a1_int, a2_int, k_int, t, Td):
        solution = scint.solve_ivp(rhs, [t[0], t[-1]], x0, args=[u, a1_int, a2_int, k_int, Td], method='BDF', dense_output=True)
        return solution.sol(t[:-1])[0]

    def costfun(genome):
        A1_grid = genome[:xu_size].reshape(xu_shape)
        A2_grid = genome[xu_size:2*xu_size].reshape(xu_shape)
        k_grid = genome[2*xu_size:3*xu_size].reshape(xu_shape)
        Td = genome[3*xu_size:]
        x = np.linspace(xu_ranges[0][0], xu_ranges[0][1], xu_shape[0])
        u = np.linspace(xu_ranges[1][0], xu_ranges[1][1], xu_shape[1])

        a1_int = interpolate.RegularGridInterpolator((x,u), A1_grid, method='cubic')
        a2_int = interpolate.RegularGridInterpolator((x,u), A2_grid, method='cubic')
        k_int = interpolate.RegularGridInterpolator((x,u), k_grid, method='cubic')
        Td_int = interpolate.interp1d(np.linspace(xu_ranges[1][0], xu_ranges[1][1], len(Td)), Td, kind='cubic')

        cost = 0
        for u0 in u:
            solution = solve_direct([x0, x0], u0, a1_int, a2_int, k_int, T[u0], Td_int(u0))
            xu_s = solution.sol(T[u0][:-1])[0]
            cost += ((xu_s - y[u0][:-1])**2).mean()
        return cost

    return costfun


def prepare_stopping_costfun(T, xu_shape, xu_ranges, y):
    '''
    Prepare cost function for genetic algorithm optimization.
    t: dict[float: np.ndarray] - dictionary of time vectors loaded from Frantisek data
    xu_shape: tuple - shape of possible state combinations. Exempli gratia (x, u) -> (11, 10)
    xu_ranges: tuple - ranges for 2 state variables. Exempli gratia (x, u) -> ((0.0, 20.0) (10.0, 100.0))
    y: dict[float: np.ndarray] - dictionary of output vectors loaded from Frantisek data and used for comparison
    '''
    x0 =  0.0
    xu_size = xu_shape[0]*xu_shape[1]
    u0s = np.linspace(xu_ranges[1][0], xu_ranges[1][1], xu_shape[1])

    def rhs(t, x, *args):
        u, a1_int, a2_int, k_int, Td = args

        if x[0] > xu_ranges[0][1]:
            xu = np.array([xu_ranges[0][1], u])
        elif x[0] < xu_ranges[0][0]:
            xu = np.array([xu_ranges[0][0], u])
        else:
            xu = np.array([x[0], u])

        if t < Td:
            return [x[1], ((-x[0] - a1_int(xu) * x[1]) / a2_int(xu))[0]]
        else:
            return [x[1], ((-x[0] - a1_int(xu) * x[1] + k_int(xu) * u) / a2_int(xu))[0]]
    
    def event(_, x, *args) -> float:
        return max(30.0 - x[1], 0)

    event.terminal = True

    def solve_direct(x0, u, a1_int, a2_int, k_int, t, Td):
        solution = scint.solve_ivp(rhs, [t[0], t[-1]], x0, args=[u, a1_int, a2_int, k_int, Td], method='BDF', dense_output=True, events=event)
        if solution.t[-1] < t[-1]:
            return (t[-1] - solution.t[-1]) * 10e6 + 10e3
        else:
            xu_s = solution.sol(T[u][:-1])[0]
            return ((xu_s - y[u][:-1])**2).mean()

    def costfun(genome):
        A1_grid = genome[:xu_size].reshape(xu_shape)
        A2_grid = genome[xu_size:2*xu_size].reshape(xu_shape)
        k_grid = genome[2*xu_size:3*xu_size].reshape(xu_shape)
        Td = genome[3*xu_size:]
        x = np.linspace(xu_ranges[0][0], xu_ranges[0][1], xu_shape[0])
        u = np.linspace(xu_ranges[1][0], xu_ranges[1][1], xu_shape[1])

        a1_int = interpolate.RegularGridInterpolator((x,u), A1_grid, method='cubic')
        a2_int = interpolate.RegularGridInterpolator((x,u), A2_grid, method='cubic')
        k_int = interpolate.RegularGridInterpolator((x,u), k_grid, method='cubic')
        Td_int = interpolate.interp1d(np.linspace(xu_ranges[1][0], xu_ranges[1][1], len(Td)), Td, kind='cubic')

        return sum([solve_direct([x0, x0], u0, a1_int, a2_int, k_int, T[u0], Td_int(u0)) for u0 in u0s])

    return costfun
