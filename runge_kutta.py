import os

import numpy as np
from scipy.optimize import fsolve


def ksystem(k_coeffs: np.array, *args: tuple) -> list:
    """
    Equations system for k1, k2 coefficients of the Implicit Runge-Kutta scheme of 4th order
    :param k_coeffs:
            k_coeffs[0] k1x
            k_coeffs[1] k2x
            k_coeffs[2] k1y
            k_coeffs[3] k2y
            k_coeffs[4] k1z
            k_coeffs[5] k2z
    :return:
    """
    vector_f, h, xyz = args[0], args[1], args[2:]
    coeff_system = [  # equation for k_coeffs[0] k1x
        k_coeffs[0] - h * vector_f[0](xyz[0] + k_coeffs[0] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[1],
                                    xyz[1] + k_coeffs[2] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[3],
                                    xyz[2] + k_coeffs[4] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[5]),
        # equation for k_coeffs[2] k1y
        k_coeffs[2] - h * vector_f[1](xyz[0] + k_coeffs[0] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[1],
                                    xyz[1] + k_coeffs[2] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[3],
                                    xyz[2] + k_coeffs[4] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[5]),
        # equation for k_coeffs[4] k1z
        k_coeffs[4] - h * vector_f[2](xyz[0] + k_coeffs[0] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[1],
                                    xyz[1] + k_coeffs[2] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[3],
                                    xyz[2] + k_coeffs[4] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[5]),
        # equation for k_coeffs[1] k2x
        k_coeffs[1] - h * vector_f[0](xyz[0] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[0] + k_coeffs[1] / 4,
                                    xyz[1] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[2] + k_coeffs[3] / 4,
                                    xyz[2] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[4] + k_coeffs[5] / 4),
        # equation for k_coeffs[3] k2y
        k_coeffs[3] - h * vector_f[1](xyz[0] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[0] + k_coeffs[1] / 4,
                                    xyz[1] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[2] + k_coeffs[3] / 4,
                                    xyz[2] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[4] + k_coeffs[5] / 4),
        # equation for k_coeffs[5] k2z
        k_coeffs[5] - h * vector_f[2](xyz[0] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[0] + k_coeffs[1] / 4,
                                    xyz[1] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[2] + k_coeffs[3] / 4,
                                    xyz[2] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[4] + k_coeffs[5] / 4),
    ]
    return coeff_system


def solve_ksystem(system_to_solve: callable, xyz: np.ndarray, h: float, vector_f: list,
                  k0: np.array = np.array((0, 0, 0, 0, 0, 0))) -> np.array:
    """
    Returns coefficients of the Implicit Runge-Kutta scheme of 4th order
    :param system_to_solve: callable function that returns list of equations to solve
    :param xyz: current dimensional point
    :param h: step size
    :param vector_f: vector of right parts of the ODE system
    :param k0: initial values for optimizing variables
    :return: list of roots (k_coeffs) = [k1x, k2x, k1y, k2y, k1z, k2z]
    """
    args = (vector_f, h, *xyz)
    root = fsolve(system_to_solve, k0, args=args)
    return root


def make_one_step(xyz: np.ndarray, h: float, vector_f: list) -> np.array:
    """"
    Makes one step from the point (x, y, z) with the step size = h
    :param xyz: current dimensional point
    :param h: step size
    :param vector_f: vector of right parts of the ODE system
    :return: np.array with coordinated of next point
    """
    k_coeffs = solve_ksystem(system_to_solve=ksystem, xyz=xyz, h=h, vector_f=vector_f)
    xyz_one_step = np.array([xyz[0] + (k_coeffs[0] + k_coeffs[2]) / 2,
                             xyz[1] + (k_coeffs[2] + k_coeffs[3]) / 2,
                             xyz[2] + (k_coeffs[4] + k_coeffs[5]) / 2
                             ])
    return xyz_one_step


def make_two_steps(xyz: np.ndarray, h: float, vector_f: list) -> np.array:
    """"
    Makes two steps from the point (x, y, z) with the step size = h/2
    :param xyz: current dimensional point
    :param h: step size
    :param vector_f: vector of right parts of the ODE system
    :return: np.array with coordinated of next point
    """
    # k system coeffs (roots) for step = h/2
    h = h / 2
    k_coeffs = solve_ksystem(system_to_solve=ksystem, xyz=xyz, h=h, vector_f=vector_f)

    # values for xyz with one step = h/2
    xyz_one_half_step = np.array([xyz[0] + (k_coeffs[0] + k_coeffs[2]) / 2,
                                  xyz[1] + (k_coeffs[2] + k_coeffs[3]) / 2,
                                  xyz[2] + (k_coeffs[4] + k_coeffs[5]) / 2
                                  ])

    # values for xyz with two steps of step = h/2
    xyz_two_half_step = np.array([xyz_one_half_step[0] + (k_coeffs[0] + k_coeffs[2]) / 2,
                                  xyz_one_half_step[1] + (k_coeffs[2] + k_coeffs[3]) / 2,
                                  xyz_one_half_step[2] + (k_coeffs[4] + k_coeffs[5]) / 2
                                  ])
    return xyz_two_half_step


def calculate_iter_errors(xyz_one_step: np.array, xyz_two_half_step: np.array) -> np.array:
    """"
    Calculates iteration errors for each coordinate (x,y,z)
    """
    iter_errors = np.array([np.abs((xyz_one_step[0] - xyz_two_half_step[0]) / 15),
                            np.abs((xyz_one_step[1] - xyz_two_half_step[1]) / 15),
                            np.abs((xyz_one_step[2] - xyz_two_half_step[2]) / 15)
                            ])
    return iter_errors


def run(t_start, t_end, xyz0, vector_f: list, h0: float = 0.1, epsilon: float = 10e-4, max_iter: int = 10_000):
    xyz = np.ndarray(shape=(1, 3))
    xyz[0] = xyz0
    times = np.array([t_start])

    # arrays for component errors
    errors = np.array([0, 0, 0])

    # array for errors norm
    error_norms = np.array([0])

    # array for step lengths
    steps = np.array([h0])
    h = h0

    curr_iter = 0
    while times[curr_iter] <= t_end:
        if curr_iter > max_iter:
            break

        if times[curr_iter] + steps[curr_iter] > t_end:
            # hxyz = (np.abs(t_end - txyz[curr_iter][0]) + epsilon, hxyz[1], hxyz[2], hxyz[3])
            h = np.abs(t_end - times[curr_iter]) + epsilon
            steps = np.append(steps, h)

        print("CURRENT_ITER: ", curr_iter)
        print("current time point = ", times[curr_iter])
        print("h = ", h)
        print("xyz = ", xyz[curr_iter])

        xyz_one_step = make_one_step(xyz=xyz[curr_iter], h=h, vector_f=vector_f)

        xyz_two_half_step = make_two_steps(xyz=xyz[curr_iter], h=h, vector_f=vector_f)
        print("xyz_two_half_step = ", xyz_two_half_step)

        # Calculates Euclidian norm of errors vector: √(err_x^2 + err_y^2 + err_z^2)
        iter_errors = calculate_iter_errors(xyz_one_step=xyz_one_step, xyz_two_half_step=xyz_two_half_step)

        iter_error_norm = np.linalg.norm(iter_errors)
        print(f"error norm for iter {curr_iter} = ", iter_error_norm)
        if iter_error_norm > epsilon:

            h = h / 2
            steps[curr_iter] = h
            continue

        xyz = np.append(xyz, [xyz_two_half_step], axis=0)
        times = np.append(times, times[curr_iter] + h)
        steps = np.append(steps, h)

        errors = np.append(errors, iter_errors)
        error_norms = np.append(error_norms, iter_error_norm)

        curr_iter += 1

        if iter_error_norm < epsilon / 16:
            h = h * 2


    print(f"Current time point:\n", times[-1])
    print(f"Solution:\n", xyz[curr_iter])
    print(f"Error norm:\n", error_norms[-1])

    return times, xyz, errors


if __name__ == '__main__':
    EXPORT_DIR_PATH = 'data/'

    if not os.path.exists(EXPORT_DIR_PATH):
        os.makedirs(EXPORT_DIR_PATH)

    mu = 80
    a = 5
    func_x = lambda x, y, z: mu * y - a * x
    func_y = lambda x, y, z: x * z - y
    func_z = lambda x, y, z: 1 - x * y - z
    vallis_right_parts = [func_x, func_y, func_z]

    # initial conditions
    t_start = 0
    t_end = 100
    h0 = 0.1
    xyz0 = (1, 1, 1)

    tn, xyz_arr, err_norms = run(t_start=t_start, t_end=t_end, xyz0=xyz0, h0=h0, vector_f=vallis_right_parts)
    np.savetxt(f'{EXPORT_DIR_PATH}tn.csv', tn)
    np.savetxt(f'{EXPORT_DIR_PATH}xyzn.csv', xyz_arr, delimiter=';')
