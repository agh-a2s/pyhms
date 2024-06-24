import scipy.integrate as scint
from scipy import interpolate
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt

def plot_solutions(solution_genome, starting_genome, T, xu_shape, xu_ranges, y):
    x0 =  0.0
    xu_size = xu_shape[0]*xu_shape[1]

    def rhs(t, x, *params):
        u, a1_int, a2_int, k_int, Td = params

        if x[0] > xu_ranges[0][1]:
            xu = np.array([xu_ranges[0][1], u])
        elif x[0] < xu_ranges[0][0]:
            xu = np.array([xu_ranges[0][0], u])
        else:
            xu = np.array([x[0], u])

        try:
            a1 = a1_int(xu)
            a2 = a2_int(xu)
            k = k_int(xu)
        except ValueError:
            raise ValueError("Interpolation failed. {} is out of bounds.".format(xu))

        if t < Td:
            u_i = 0.0
        else:
            u_i = u

        if u_i < 0.0 or u_i > xu_ranges[1][1]:
            raise ValueError("Given u is out of bounds.")
        else:
            x0 = x[1]
            x1 = ((-x[0] - a1 * x[1] + k * u_i) / a2)[0]
            if abs(x1) > 1000:
                x1 = 100 * np.sign(x1)
            return [x0, x1]
    
    def solve_direct(x0, u, a1_int, a2_int, k_int, t, Td):
        return scint.solve_ivp(rhs, (t[0], t[-1]), x0, args=[u, a1_int, a2_int, k_int, Td], method='Radau', dense_output=True)
    
    def solve(genome):
        A1_grid = genome[:xu_size].reshape(xu_shape)
        A2_grid = genome[xu_size:2*xu_size].reshape(xu_shape)
        k_grid = genome[2*xu_size:3*xu_size].reshape(xu_shape)
        Td = genome[3*xu_size:]
        x = np.linspace(xu_ranges[0][0], xu_ranges[0][1], xu_shape[0])
        u = np.linspace(xu_ranges[1][0], xu_ranges[1][1], xu_shape[1])

        a1_int = interpolate.RegularGridInterpolator((x,u), A1_grid, method='cubic')
        a2_int = interpolate.RegularGridInterpolator((x,u), A2_grid, method='cubic')
        k_int = interpolate.RegularGridInterpolator((x,u), k_grid, method='cubic')
        Td_int = interpolate.interp1d(np.linspace(xu_ranges[1][0], xu_ranges[1][1], xu_shape[1]), Td, kind='cubic')

        results = {}
        for u0 in np.linspace(xu_ranges[1][0], xu_ranges[1][1], xu_shape[1]):
            solution = solve_direct([x0, x0], u0, a1_int, a2_int, k_int, T[u0], Td_int(u0))
            results[u0] = solution.sol(T[u0][:-1])[0]
        return results

    results_new = solve(solution_genome)
    results_old = solve(starting_genome)

    mat_data_dump = {}

    for u0 in np.linspace(xu_ranges[1][0], xu_ranges[1][1], xu_shape[1]):
        _ = plt.subplots(figsize=(10, 5))
        plt.plot(T[u0][:-1], results_new[u0], label='Simulated output: new solution')
        plt.plot(T[u0][:-1], results_old[u0], label='Simulated output: old solution')
        plt.plot(T[u0][:-1], y[u0][:-1], label="Frantisek's data")
        plt.legend(loc='upper left')
        plt.xlabel('Time')
        plt.ylabel('Pressure')
        plt.title('Comparison of Frantisek data and simulated output for valve opening {}%'.format(u0))
        plt.savefig('plots/comparison_u{}.png'.format(u0))

        mat_data_dump['u{}_new'.format(int(u0))] = results_new[u0]
        mat_data_dump['u{}_old'.format(int(u0))] = results_old[u0]
        mat_data_dump['u{}_y'.format(int(u0))] = y[u0][:-1]
        mat_data_dump['u{}_T'.format(int(u0))] = T[u0][:-1]

    savemat('plots/comparison_data.mat', mat_data_dump)
