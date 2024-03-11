import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot
import math


def plot_nlp_dynamics(shooting_nodes, 
                      idx_b_x, lb_x, ub_x, idx_b_u, lb_u, ub_u, 
                      U, X_sim,
                      U_predict=None, X_predict=None, t_predict=None,
                      latexify=False):
    """
    Params:
        shooting_nodes: time values of the discretization

        idx_b_x: Indices of constraints within state vector
        lb_x: lower bounds on respective states
        ub_x: upper bounds on respective states

        idx_b_u: Indices of constraints within input vector
        lb_u: lower bounds on respective inputs
        ub_u: upper bounds on respective inputs

        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_sim: arrray with shape (N_sim, nx)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    # Get simulation length and input/state dimensions
    N_sim = X_sim.shape[0]
    nx = X_sim.shape[1]
    N_sim_u = U.shape[0]
    nu = U.shape[1]

    # get final time and time vector
    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes
    # get time step size
    Ts = t[1] - t[0]

    # running index for plotting subplots
    plotting_idx = 1

    plt.figure()
    n_rows_plot = math.ceil((nx+nu)/2)

    inputs_lables = [r'$da_{x,m}$', r'$d\delta_s$']
    idx_u = 0
    for i in range(nu):
        plt.subplot(n_rows_plot, 2, plotting_idx)
        line, = plt.step(t, np.append([U[0, i]], U[:, i]), linewidth=2.0)
        line.set_color('r')

        if U_predict is not None:
            line, = plt.step(t_predict, np.append([U_predict[0, i]], U_predict[:, i]))
            line.set_color('m')

        plt.ylabel(inputs_lables[i])
        plt.xlabel('$t$')
        if i in idx_b_u:
            plt.hlines(ub_u[idx_u], t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(lb_u[idx_u], t[0], t[-1], linestyles='dashed', alpha=0.7)            
            idx_u += 1

        plt.grid('both')

        plotting_idx += 1

    states_lables = ['$s$', '$n$', r'$\mu$', '$v_x$', '$a_x$', r'$\delta_s$']
    idx_x = 0
    for i in range(nx):
        plt.subplot(n_rows_plot, 2, plotting_idx)
        line, = plt.plot(t, X_sim[:, i], linewidth=2.0)
        line.set_color('b')

        if X_predict is not None:
            line, = plt.plot(t_predict, X_predict[:, i])
            line.set_color('c')

        plt.ylabel(states_lables[i])
        plt.xlabel('$t$')
        if i in idx_b_x:
            plt.hlines(ub_x[idx_x], t[0], t[-1], linestyles='dashed', alpha=0.7)
            plt.hlines(lb_x[idx_x], t[0], t[-1], linestyles='dashed', alpha=0.7)

            idx_x += 1

        plt.grid('both')

        plotting_idx += 1

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.show()