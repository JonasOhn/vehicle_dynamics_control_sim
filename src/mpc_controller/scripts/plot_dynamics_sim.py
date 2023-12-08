import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot
import math


def plot_dynamics(shooting_nodes, idx_b_x, lb_x, ub_x, idx_b_u, lb_u, ub_u, U, X_true, latexify=False, plt_show=True, plot_constraints=True):
    """
    Params:
        shooting_nodes: time values of the discretization
        U_con: input constraints
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    N_sim_u = U.shape[0]
    nu = U.shape[1]

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    Ts = t[1] - t[0]

    plotting_idx = 1
    plt.figure()
    n_rows_plot = math.ceil((nx+nu)/2)

    inputs_lables = [r'$Fx_m$', r'$\delta_s$']
    idx_u = 0
    for i in range(nu):
        plt.subplot(n_rows_plot, 2, plotting_idx)
        line, = plt.step(t, np.append([U[0, i]], U[:, i]))
        line.set_color('r')

        plt.ylabel(inputs_lables[i])
        plt.xlabel('$t$')
        if plot_constraints:
            if i in idx_b_u:
                plt.hlines(ub_u[idx_u], t[0], t[-1], linestyles='dashed', alpha=0.7)
                plt.hlines(lb_u[idx_u], t[0], t[-1], linestyles='dashed', alpha=0.7)
                plt.ylim([1.2*lb_u[idx_u], 1.2*ub_u[idx_u]])
                idx_u += 1
        plt.xlim(t[0], t[-1])
        plt.grid('both')

        plotting_idx += 1

    states_lables = ['$s$', '$n$', r'$\mu$', '$v_x$', '$v_y$', r'$\dot{\psi}$']
    idx_x = 0
    for i in range(nx):
        plt.subplot(n_rows_plot, 2, plotting_idx)
        line, = plt.plot(t, X_true[:, i], label='true')

        plt.ylabel(states_lables[i])
        plt.xlabel('$t$')
        if plot_constraints:
            if i in idx_b_x:
                plt.hlines(ub_x[idx_x], t[0], t[-1], linestyles='dashed', alpha=0.7)
                plt.hlines(lb_x[idx_x], t[0], t[-1], linestyles='dashed', alpha=0.7)
                plt.ylim([-1.2*np.abs(lb_x[idx_x]), 1.2*ub_x[idx_x]])
                idx_x += 1
        plt.grid('both')
        plt.legend(loc=1)
        plt.xlim(t[0], t[-1])

        plotting_idx += 1

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    if plt_show:
        plt.show()