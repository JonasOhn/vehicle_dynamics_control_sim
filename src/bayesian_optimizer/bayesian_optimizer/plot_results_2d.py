"""
 * Simple Vehicle Dynamics Simulator Project
 *
 * Copyright (c) 2023-2024 Authors:
 *   - Jonas Ohnemus <johnemus@ethz.ch>
 *
 * All rights reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
"""

import numpy as np
from bo import BayesianOptimizer
from gp import GaussianProcess
import matplotlib.pyplot as plt
import yaml
import matplotlib

def main():
    results_csv_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/results/results.csv"
    config_bo_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/config/optimizer_params.yaml"

    with open(config_bo_filepath, "r") as file:
        config_bo = yaml.safe_load(file)
        config_bo = config_bo["/bayesian_optimizer_node"]["ros__parameters"]
    
    # --- initialize parameters
    lb_q_n = 0.01
    ub_q_n = 2.0
    lb_q_mu = 0.01
    ub_q_mu = 2.0

    gp_noise_covariance = config_bo["gp_noise_covariance"]
    gp_lengthscale = config_bo["gp_lengthscale"]
    gp_output_variance = config_bo["gp_output_variance"]
    gp_mean = config_bo["gp_mean"]
    beta = config_bo["bo_beta"]
    max_laptime = config_bo["max_lap_time"]

    gp = GaussianProcess(
        noise_covariance=gp_noise_covariance,
        lengthscale=gp_lengthscale,
        output_variance=gp_output_variance,
    )

    bayesian_optimizer = BayesianOptimizer(gp=gp, beta=beta)

    # load data from .csv file
    print(results_csv_filepath)
    b_data = load_data(bayesian_optimizer, results_csv_filepath)
    if not b_data:
        print("No data found in .csv file. Exiting.")
        return
    
    # plot the GP in 2d
    plot_gp_2d(bayesian_optimizer, gp_mean, np.array([[lb_q_n, ub_q_n], [lb_q_mu, ub_q_mu]]))

def load_data(bayes_opt, csv_filepath):
    """
        Load data from .csv file.
    """
    print("Loading data from .csv file.")
    # try to load data from .csv file, if it exists. else, start with empty data and print to console
    try:
        data = np.loadtxt(csv_filepath, delimiter=",")
    except:
        print("Could not load data from .csv file.")
        return False
    print(data)
    bayes_opt.add_data(data[:, 1:3], data[:, 3])
    print("Loaded data from .csv file.")
    return True

def plot_gp_2d(bo, gp_mean, q_bounds):
    """
    Plot the GP in 2d

    Args:
        bo (BayesianOptimizer): the Bayesian optimizer
        gp_mean (float): mean of the GP
        q_bounds (np.array, shape=[2, 2]): bounds of the decision variables
    """

    assert len(q_bounds.shape) == 2, "q_bounds must be of shape [2,2]"
    assert q_bounds.shape[0] == 2, "q_bounds must be of shape [2,2]"
    assert q_bounds.shape[1] == 2, "q_bounds must be of shape [2,2]"

    # --- generate parameter meshgrid
    n_samples = 20

    Q = np.zeros((n_samples, 3))
    q_n = np.linspace(start=q_bounds[0][0], stop=q_bounds[0][1], num=n_samples)
    q_mu = np.linspace(start=q_bounds[1][0], stop=q_bounds[1][1], num=n_samples)

    # Generate meshgrid
    X, Y = np.meshgrid(q_n, q_mu, indexing="ij")

    # Reshape meshgrid to form a matrix where each column represents one parameter dimension
    Q = np.vstack((X.flatten(), Y.flatten())).T

    # get data from the Bayesian optimizer
    X_data, Y_data = bo.get_data()

    # plot the raw data in 2d using colormap for y
    fig = plt.figure(1)

    # plot the raw data only
    ax = fig.add_subplot(111)

    idx_last_no_bo = 232

    # plot predictions using filled contours
    mi = gp_mean + Y_data.min()
    ma = gp_mean + Y_data.max()
    norm = matplotlib.colors.Normalize(vmin=mi,vmax=ma)
    sc = ax.scatter(X_data[:idx_last_no_bo, 0], X_data[:idx_last_no_bo, 1], c=gp_mean + Y_data[:idx_last_no_bo], norm=norm, cmap="viridis", label="Data", marker="o")
    plt.colorbar(sc)
    ax.scatter(X_data[idx_last_no_bo:, 0], X_data[idx_last_no_bo:, 1], c='r', label="BO chosen data", marker="x")

    plt.xlim(q_bounds[0][0], q_bounds[0][1])
    plt.ylim(q_bounds[1][0], q_bounds[1][1])

    plt.xlabel("q_n")
    plt.ylabel("q_mu")
    plt.title("Raw data")


    fig = plt.figure(2)

    # plot the GP mean in 2d
    ax = fig.add_subplot(111)

    y_pred, y_ucb, y_lcb = bo.sample_gp(Q)
    confidence = y_ucb - y_lcb

    # plot predictions using filled contours
    # mi = np.min((Y_data.min(), Y.min()))
    # ma = np.max((Y_data.max(), Y.max()))
    mi = Y_data.min()
    ma = Y_data.max()
    norm = matplotlib.colors.Normalize(vmin=mi,vmax=ma)
    # plot contour plot of the GP mean
    cp = ax.contourf(X, Y, y_pred.reshape(n_samples, n_samples), cmap="viridis", alpha=0.5, norm=norm)
    sc = ax.scatter(X_data[:idx_last_no_bo, 0], X_data[:idx_last_no_bo, 1], c=Y_data[:idx_last_no_bo], norm=norm, cmap="viridis", label="Data", marker="x")
    plt.colorbar(sc)
    ax.scatter(X_data[idx_last_no_bo:, 0], X_data[idx_last_no_bo:, 1], c='r', label="BO chosen data", marker="x")

    plt.xlim(q_bounds[0][0], q_bounds[0][1])
    plt.ylim(q_bounds[1][0], q_bounds[1][1])

    plt.xlabel("q_n")
    plt.ylabel("q_mu")
    plt.title("Predictions and raw data, mean laptime subtracted")


    fig = plt.figure(3)

    # plot the uncertainty in 2d using colormap for y
    ax = fig.add_subplot(111)

    mi = confidence.min()
    ma = confidence.max()
    norm = matplotlib.colors.Normalize(vmin=mi,vmax=ma)
    cp = ax.contourf(X, Y, confidence.reshape(n_samples, n_samples), cmap="viridis", alpha=0.5, norm=norm)
    plt.colorbar(cp)
    ax.scatter(X_data[:, 0], X_data[:, 1], c='k', label="Data", marker="x")

    plt.xlim(q_bounds[0][0], q_bounds[0][1])
    plt.ylim(q_bounds[1][0], q_bounds[1][1])
    
    plt.xlabel("q_n")
    plt.ylabel("q_mu")
    plt.title("Uncertainty (std) and raw data")

    # plot acquisition function
    fig = plt.figure(4)

    ax = fig.add_subplot(111)

    xmin, _, y_acq = bo.aquisition_function(Q)

    # plot predictions using filled contours
    mi = y_acq.min()
    ma = y_acq.max()
    norm = matplotlib.colors.Normalize(vmin=mi,vmax=ma)
    cp = ax.contourf(X, Y, y_acq.reshape(n_samples, n_samples), cmap="viridis", alpha=0.5, norm=norm)
    plt.colorbar(cp)
    ax.scatter(X_data[:, 0], X_data[:, 1], c='k', label="Data", marker="x")
    ax.scatter(xmin[0], xmin[1], c='r', label="Minimum", marker="x")

    plt.xlim(q_bounds[0][0], q_bounds[0][1])
    plt.ylim(q_bounds[1][0], q_bounds[1][1])

    plt.xlabel("q_n")
    plt.ylabel("q_mu")
    plt.title("Acquisition function")


    plt.show()

if __name__ == "__main__":
    main()