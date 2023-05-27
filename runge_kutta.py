import os

import numpy as np
from scipy.optimize import fsolve

EXPORT_DIR_PATH = 'data/'

if not os.path.exists(EXPORT_DIR_PATH):
    os.makedirs(EXPORT_DIR_PATH)

EPSILON = 10e-4


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
    :param hxyz:
    :return:
    """
    coeff_system = [  # equation for k_coeffs[0] k1x
        k_coeffs[0] - hxyz[0] * func_x(hxyz[1] + k_coeffs[0] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[1],
                                       hxyz[2] + k_coeffs[2] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[3],
                                       hxyz[3] + k_coeffs[4] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[5]),
        # equation for k_coeffs[2] k1y
        k_coeffs[2] - hxyz[0] * func_y(hxyz[1] + k_coeffs[0] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[1],
                                       hxyz[2] + k_coeffs[2] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[3],
                                       hxyz[3] + k_coeffs[4] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[5]),
        # equation for k_coeffs[4] k1z
        k_coeffs[4] - hxyz[0] * func_z(hxyz[1] + k_coeffs[0] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[1],
                                       hxyz[2] + k_coeffs[2] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[3],
                                       hxyz[3] + k_coeffs[4] / 4 + (1 / 4 - np.sqrt(3) / 6) * k_coeffs[5]),
        # equation for k_coeffs[1] k2x
        k_coeffs[1] - hxyz[0] * func_x(hxyz[1] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[0] + k_coeffs[1] / 4,
                                       hxyz[2] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[2] + k_coeffs[3] / 4,
                                       hxyz[3] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[4] + k_coeffs[5] / 4),
        # equation for k_coeffs[3] k2y
        k_coeffs[3] - hxyz[0] * func_y(hxyz[1] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[0] + k_coeffs[1] / 4,
                                       hxyz[2] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[2] + k_coeffs[3] / 4,
                                       hxyz[3] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[4] + k_coeffs[5] / 4),
        # equation for k_coeffs[5] k2z
        k_coeffs[5] - hxyz[0] * func_z(hxyz[1] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[0] + k_coeffs[1] / 4,
                                       hxyz[2] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[2] + k_coeffs[3] / 4,
                                       hxyz[3] + (1 / 4 + np.sqrt(3) / 6) * k_coeffs[4] + k_coeffs[5] / 4),
    ]
    return coeff_system


def solve_ksystem(system_to_solve: callable, k0: np.array, hxyz: tuple) -> np.array:
    """

    :param system_to_solve: callable function that returns list of equations to solve
    :param k0: initial values for optimizing variables
    :param hxyz: initial conditions for the system
    :return: list of roots (k_coeffs)
    """
    root = fsolve(system_to_solve, k0, args=hxyz)
    return root

def make_one_step(hxyz: tuple):
    """"
    Makes one step with the step size = h
    """
    k_coeffs = solve_ksystem(system_to_solve=ksystem, k0=k0, hxyz=hxyz)
    xyz_one_step = np.array([hxyz[1] + (k_coeffs[0] + k_coeffs[2]) / 2,
                             hxyz[2] + (k_coeffs[2] + k_coeffs[3]) / 2,
                             hxyz[3] + (k_coeffs[4] + k_coeffs[5]) / 2
                             ])
    return xyz_one_step


def make_two_steps(hxyz: tuple):
    """"
    Makes two steps with the step size = h/2
    """
    # k system coeffs (roots) for step = h/2
    k_coeffs = solve_ksystem(system_to_solve=ksystem, k0=k0, hxyz=(hxyz[0] / 2, hxyz[1], hxyz[2], hxyz[3]))

    # values for xyz with one step = h/2
    xyz_one_half_step = np.array([hxyz[1] + (k_coeffs[0] + k_coeffs[2]) / 2,
                                  hxyz[2] + (k_coeffs[2] + k_coeffs[3]) / 2,
                                  hxyz[3] + (k_coeffs[4] + k_coeffs[5]) / 2
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

# initial conditions
t_start = 0
t_end = 100
hxyz = (0.1, 1, 1, 1)

xn = np.array([hxyz[1]])
yn = np.array([hxyz[2]])
zn = np.array([hxyz[3]])
tn = np.array([t_start])

# arrays for component errors
err_x = np.array([0])
err_y = np.array([0])
err_z = np.array([0])

# array for errors norm
error_norms = np.array([0])

# array for step lengths
steps = np.array([hxyz[0]])

curr_iter = 0

# initial conditions for k system
k0 = np.array((0, 0, 0, 0, 0, 0))


while tn[curr_iter] <= t_end:
    if tn[curr_iter] + hxyz[0] > t_end:
        hxyz = (np.abs(t_end - tn[curr_iter]) + EPSILON, hxyz[1], hxyz[2], hxyz[3])

    print("CURRENT_ITER: ", curr_iter)
    print("current time point = ", tn[curr_iter])
    print("hxyz = ", hxyz)

    """
    k_coeffs[0] k1x
    k_coeffs[1] k2x
    k_coeffs[2] k1y
    k_coeffs[3] k2y
    k_coeffs[4] k1z
    k_coeffs[5] k2z
    """

    # k_coeffs = solve_ksystem(system_to_solve=ksystem, k0=k0, hxyz=hxyz)
    # # values for xyz with step = h
    # xyz_one_step = np.array([hxyz[1] + (k_coeffs[0] + k_coeffs[2]) / 2,
    #                          hxyz[2] + (k_coeffs[2] + k_coeffs[3]) / 2,
    #                          hxyz[3] + (k_coeffs[4] + k_coeffs[5]) / 2
    #                          ])
    xyz_one_step = make_one_step(hxyz=hxyz)

    # k_coeffs_h2 = solve_ksystem(system_to_solve=ksystem, k0=k0, hxyz=(hxyz[0] / 2, hxyz[1], hxyz[2], hxyz[3]))
    #
    # # values for xyz with step = h/2
    # xyz_one_half_step = np.array([hxyz[1] + (k_coeffs_h2[0] + k_coeffs_h2[2]) / 2,
    #                               hxyz[2] + (k_coeffs_h2[2] + k_coeffs_h2[3]) / 2,
    #                               hxyz[3] + (k_coeffs_h2[4] + k_coeffs_h2[5]) / 2
    #                               ])
    #
    # # values for xyz with two steps of step = h/2
    # xyz_two_half_step = np.array([xyz_one_half_step[0] + (k_coeffs_h2[0] + k_coeffs_h2[2]) / 2,
    #                               xyz_one_half_step[1] + (k_coeffs_h2[2] + k_coeffs_h2[3]) / 2,
    #                               xyz_one_half_step[2] + (k_coeffs_h2[4] + k_coeffs_h2[5]) / 2
    #                               ])
    xyz_two_half_step = make_two_steps(hxyz=hxyz)

    # iter_errors = np.array([np.abs((xyz_one_step[0] - xyz_two_half_step[0]) / 15),
    #                         np.abs((xyz_one_step[1] - xyz_two_half_step[1]) / 15),
    #                         np.abs((xyz_one_step[2] - xyz_two_half_step[2]) / 15)
    #                         ])
    iter_errors = calculate_iter_errors(xyz_one_step=xyz_one_step, xyz_two_half_step=xyz_two_half_step)

    # Euclidian norm of errors vector: √(err_x^2 + err_y^2 + err_z^2)
    # iter_error_norm = np.sqrt(np.sum(np.power(iter_errors, 2)))
    iter_error_norm = euclidian_norm(arr=iter_errors)
    print(f"error norm for iter {curr_iter} = ", iter_error_norm)
    if iter_error_norm > EPSILON:

        hxyz = (hxyz[0] / 2, hxyz[1], hxyz[2], hxyz[3])

    elif iter_error_norm < EPSILON / 16:

        tn = np.append(tn, tn[curr_iter] + hxyz[0])
        xn = np.append(xn, xyz_two_half_step[0])
        yn = np.append(yn, xyz_two_half_step[1])
        zn = np.append(zn, xyz_two_half_step[2])
        steps = np.append(steps, hxyz[0])

        error_norms = np.append(error_norms, iter_error_norm)
        err_x = np.append(err_x, iter_errors[0])
        err_y = np.append(err_y, iter_errors[1])
        err_z = np.append(err_z, iter_errors[2])

        hxyz = (hxyz[0] * 2, xyz_two_half_step[0], xyz_two_half_step[1], xyz_two_half_step[2])

        curr_iter += 1

    elif EPSILON / 16 < iter_error_norm <= EPSILON:

        tn = np.append(tn, tn[curr_iter] + hxyz[0])
        xn = np.append(xn, xyz_two_half_step[0])
        yn = np.append(yn, xyz_two_half_step[1])
        zn = np.append(zn, xyz_two_half_step[2])
        steps = np.append(steps, hxyz[0])

        error_norms = np.append(error_norms, iter_error_norm)
        err_x = np.append(err_x, iter_errors[0])
        err_y = np.append(err_y, iter_errors[1])
        err_z = np.append(err_z, iter_errors[2])

        hxyz = (hxyz[0], xyz_two_half_step[0], xyz_two_half_step[1], xyz_two_half_step[2])

        curr_iter += 1

np.savetxt(f'{EXPORT_DIR_PATH}tn.csv', tn)
np.savetxt(f'{EXPORT_DIR_PATH}xn.csv', xn)
np.savetxt(f'{EXPORT_DIR_PATH}yn.csv', yn)
np.savetxt(f'{EXPORT_DIR_PATH}zn.csv', zn)

print(f"Current time point:\n", tn[-1])
print(f"Solution:\n", hxyz)
print(f"Error norm:\n", error_norms[-1])
