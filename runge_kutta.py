import os

import numpy as np
from scipy.optimize import fsolve

EXPORT_DIR_PATH = 'data/'

if not os.path.exists(EXPORT_DIR_PATH):
    os.makedirs(EXPORT_DIR_PATH)


def func_x(x, y, z):
    mu = 80
    a = 5
    return mu * y - a * x


def func_y(x, y, z):
    return x * z - y


def func_z(x, y, z):
    return 1 - x * y - z


def ksystem(k_coeffs: np.array, *hxyz: tuple) -> list:
    """
    Equations system for k1, k2 coefficients of the Implicit Runge-Kutta scheme of 4th order
    :param k_coeffs:
            k_coeffs[0] k1x
            k_coeffs[1] k2x
            k_coeffs[2] k1y
            k_coeffs[3] k2y
            k_coeffs[4] k1z
            k_coeffs[5] k2z
    :param xyz:
    :return:
    """
    h, xyz = hxyz[0], hxyz[1:]
    coeff_system = [  # equation for k_coeffs[0] k1x
        k_coeffs[0] - h * func_x(xyz[0] + k_coeffs[0] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[1],
                                 xyz[1] + k_coeffs[2] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[3],
                                 xyz[2] + k_coeffs[4] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[5]),
        # equation for k_coeffs[2] k1y
        k_coeffs[2] - h * func_y(xyz[0] + k_coeffs[0] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[1],
                                 xyz[1] + k_coeffs[2] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[3],
                                 xyz[2] + k_coeffs[4] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[5]),
        # equation for k_coeffs[4] k1z
        k_coeffs[4] - h * func_z(xyz[0] + k_coeffs[0] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[1],
                                 xyz[1] + k_coeffs[2] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[3],
                                 xyz[2] + k_coeffs[4] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[5]),
        # equation for k_coeffs[1] k2x
        k_coeffs[1] - h * func_x(xyz[0] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[0] + k_coeffs[1] / 4,
                                 xyz[1] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[2] + k_coeffs[3] / 4,
                                 xyz[2] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[4] + k_coeffs[5] / 4),
        # equation for k_coeffs[3] k2y
        k_coeffs[3] - h * func_y(xyz[0] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[0] + k_coeffs[1] / 4,
                                 xyz[1] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[2] + k_coeffs[3] / 4,
                                 xyz[2] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[4] + k_coeffs[5] / 4),
        # equation for k_coeffs[5] k2z
        k_coeffs[5] - h * func_z(xyz[0] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[0] + k_coeffs[1] / 4,
                                 xyz[1] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[2] + k_coeffs[3] / 4,
                                 xyz[2] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[4] + k_coeffs[5] / 4),
    ]
    return coeff_system


def solve_ksystem(system_to_solve: callable, xyz: tuple, h: float, k0: np.array = np.array((0, 0, 0, 0, 0, 0))) -> np.array:
    """

    :param system_to_solve: callable function that returns list of equations to solve
    :param k0: initial values for optimizing variables
    :param hxyz: initial conditions for the system
    :return: list of roots (k_coeffs)
    """
    hxyz = (h, *xyz)
    root = fsolve(system_to_solve, k0, args=hxyz)
    return root


def make_one_step(xyz, h):
    """"
    Makes one step with the step size = h
    """
    k_coeffs = solve_ksystem(system_to_solve=ksystem, xyz=xyz, h=h)
    xyz_one_step = np.array([xyz[0] + (k_coeffs[0] + k_coeffs[2]) / 2,
                             xyz[1] + (k_coeffs[2] + k_coeffs[3]) / 2,
                             xyz[2] + (k_coeffs[4] + k_coeffs[5]) / 2
                             ])
    return xyz_one_step


def make_two_steps(xyz: tuple, h: float):
    """"
    Makes two steps with the step size = h/2
    """
    # k system coeffs (roots) for step = h/2
    h = h / 2
    k_coeffs = solve_ksystem(system_to_solve=ksystem, xyz=xyz, h=h)

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


def euclidian_norm(arr: np.array):
    """"
    Calculates Euclidian norm of errors vector: √(err_x^2 + err_y^2 + err_z^2)
    """
    eu_norm = np.sqrt(np.sum(np.power(arr, 2)))
    return eu_norm

def make_iteration():
    pass


def run(t_start, t_end, xyz0, h0: float = 0.1, epsilon: int = 10e-4):

    xyz = np.ndarray(shape=(1, 3))
    xyz[0] = xyz0
    xn = np.array([xyz0[0]])
    yn = np.array([xyz0[1]])
    zn = np.array([xyz0[2]])
    tn = np.array([t_start])

    # arrays for component errors
    errors = np.array([0, 0, 0])
    err_x = np.array([0])
    err_y = np.array([0])
    err_z = np.array([0])

    # array for errors norm
    error_norms = np.array([0])

    # array for step lengths
    steps = np.array([h0])
    h = h0

    curr_iter = 0
    while tn[curr_iter] <= t_end:
        if curr_iter > 10000:
            break

        if tn[curr_iter] + steps[curr_iter] > t_end:
            # hxyz = (np.abs(t_end - txyz[curr_iter][0]) + epsilon, hxyz[1], hxyz[2], hxyz[3])
            h = np.abs(t_end - tn[curr_iter]) + epsilon
            steps = np.append(steps, h)

        print("CURRENT_ITER: ", curr_iter)
        print("current time point = ", tn[curr_iter])
        print("h = ", h)
        print("xyz = ", xyz[curr_iter])

        xyz_one_step = make_one_step(xyz=xyz[curr_iter], h=h)

        xyz_two_half_step = make_two_steps(xyz=xyz[curr_iter], h=h)
        print("xyz_two_half_step = ", xyz_two_half_step)

        iter_errors = calculate_iter_errors(xyz_one_step=xyz_one_step, xyz_two_half_step=xyz_two_half_step)

        iter_error_norm = euclidian_norm(arr=iter_errors)
        print(f"error norm for iter {curr_iter} = ", iter_error_norm)
        if iter_error_norm > epsilon:

            h = h / 2
            steps[curr_iter] = h

        elif iter_error_norm < epsilon / 16:

            xyz = np.append(xyz, [xyz_two_half_step], axis=0)
            tn = np.append(tn, tn[curr_iter] + h)
            xn = np.append(xn, xyz_two_half_step[0])
            yn = np.append(yn, xyz_two_half_step[1])
            zn = np.append(zn, xyz_two_half_step[2])
            steps = np.append(steps, h)

            error_norms = np.append(error_norms, iter_error_norm)
            err_x = np.append(err_x, iter_errors[0])
            err_y = np.append(err_y, iter_errors[1])
            err_z = np.append(err_z, iter_errors[2])

            h = h * 2
            curr_iter += 1

        elif epsilon / 16 < iter_error_norm <= epsilon:

            xyz = np.append(xyz, [xyz_two_half_step], axis=0)
            tn = np.append(tn, tn[curr_iter] + h)
            xn = np.append(xn, xyz_two_half_step[0])
            yn = np.append(yn, xyz_two_half_step[1])
            zn = np.append(zn, xyz_two_half_step[2])
            steps = np.append(steps, h)

            error_norms = np.append(error_norms, iter_error_norm)
            err_x = np.append(err_x, iter_errors[0])
            err_y = np.append(err_y, iter_errors[1])
            err_z = np.append(err_z, iter_errors[2])

            curr_iter += 1

    print(f"Current time point:\n", tn[-1])
    print(f"Solution:\n", xyz[curr_iter])
    print(f"Error norm:\n", error_norms[-1])

    return np.array([tn, xn, yn, zn]), error_norms


# initial conditions
t_start = 0
t_end = 100
h0 = 0.1
xyz0 = (1, 1, 1)

txyz_arr, err_norms = run(t_start=t_start, t_end=t_end, xyz0=xyz0, h0=h0)
np.savetxt(f'{EXPORT_DIR_PATH}tn.csv', txyz_arr[0])
np.savetxt(f'{EXPORT_DIR_PATH}xn.csv', txyz_arr[1])
np.savetxt(f'{EXPORT_DIR_PATH}yn.csv', txyz_arr[2])
np.savetxt(f'{EXPORT_DIR_PATH}zn.csv', txyz_arr[3])
