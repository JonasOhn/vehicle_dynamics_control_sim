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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt
import matplotlib
import yaml

def main():
    # get the data
    results_csv_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/results/results_sd05_max35_mean30.csv"
    config_bo_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/config/optimizer_params.yaml"

    with open(config_bo_filepath, "r") as file:
        config_bo = yaml.safe_load(file)
        config_bo = config_bo["/bayesian_optimizer_node"]["ros__parameters"]
    
    # --- initialize parameters
    lb_q_n = 0.01
    ub_q_n = 2.0
    lb_q_mu = 0.01
    ub_q_mu = 2.0

    # load the data
    data = np.loadtxt(results_csv_filepath, delimiter=",")

    gp_mean = config_bo["gp_mean"]

    # get the data
    X_data = data[:, 1:3]
    Y_data = data[:, 3]

    print(X_data)
    print(Y_data)

    max_log_marginal_likelihood = -np.inf

    # specify kernel
    kernels = [RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
               ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
                WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-5, 10.0)) + ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
               ]
    
    for kernel in kernels:
        # create the model
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=20, normalize_y=False, random_state=42)

        # fit the model
        gpr.fit(X_data, Y_data)

        log_marginal_likelihood = gpr.log_marginal_likelihood(gpr.kernel_.theta)

        # print the score
        print("Log marginal likelihood: ", log_marginal_likelihood)
        print("Score (best is 1.0, the smaller the worse): ", gpr.score(X_data, Y_data))
        print("Kernel: ", gpr.kernel_)

        if log_marginal_likelihood > max_log_marginal_likelihood:
            max_log_marginal_likelihood = log_marginal_likelihood
            best_kernel = gpr.kernel_
            print("New best kernel found: ", best_kernel)

    print("Best kernel: ", best_kernel)
    print("Max log marginal likelihood: ", max_log_marginal_likelihood)
    
    gpr = GaussianProcessRegressor(kernel=best_kernel, alpha=0.01, n_restarts_optimizer=20, normalize_y=False, random_state=42)

    # fit the model
    gpr.fit(X_data, Y_data)

    # print the score
    print("Score (best is 1.0, the smaller the worse): ", gpr.score(X_data, Y_data))

    # get the predictions over bounded parameter space
    n_samples = 50
    q_n = np.linspace(start=lb_q_n, stop=ub_q_n, num=n_samples)
    q_mu = np.linspace(start=lb_q_mu, stop=ub_q_mu, num=n_samples)

    # Generate meshgrid
    X, Y = np.meshgrid(q_n, q_mu, indexing="ij")

    # Reshape meshgrid to form a matrix where each column represents one parameter dimension
    Q = np.vstack((X.flatten(), Y.flatten())).T

    # get the predictions
    y_pred, sigma = gpr.predict(Q, return_std=True)

    # plot the raw data in 2d using colormap for y
    fig = plt.figure(1)


    # plot the raw data only
    ax = fig.add_subplot(111)

    # plot predictions using filled contours
    mi = gp_mean + Y_data.min()
    ma = gp_mean + Y_data.max()
    norm = matplotlib.colors.Normalize(vmin=mi,vmax=ma)
    sc = ax.scatter(X_data[:, 0], X_data[:, 1], c=gp_mean + Y_data, norm=norm, cmap="viridis", label="Data", marker="o")
    plt.colorbar(sc)

    plt.xlim(lb_q_n, ub_q_n)
    plt.ylim(lb_q_mu, ub_q_mu)

    plt.xlabel("q_n")
    plt.ylabel("q_mu")
    plt.title("Raw data")


    fig = plt.figure(2)

    # plot the GP mean in 2d
    ax = fig.add_subplot(111)

    # plot predictions using filled contours
    # mi = np.min((Y_data.min(), Y.min()))
    # ma = np.max((Y_data.max(), Y.max()))
    mi = Y_data.min()
    ma = Y_data.max()
    norm = matplotlib.colors.Normalize(vmin=mi,vmax=ma)
    ax.contourf(X, Y, y_pred.reshape(n_samples, n_samples), norm=norm, cmap="viridis", alpha=0.5)
    sc = ax.scatter(X_data[:, 0], X_data[:, 1], c=Y_data, norm=norm, cmap="viridis", label="Data", marker="x")
    plt.colorbar(sc)

    plt.xlim(lb_q_n, ub_q_n)
    plt.ylim(lb_q_mu, ub_q_mu)

    plt.xlabel("q_n")
    plt.ylabel("q_mu")
    plt.title("Predictions and raw data, mean laptime subtracted")


    fig = plt.figure(3)

    # plot the uncertainty in 2d using colormap for y
    ax = fig.add_subplot(111)

    mi = sigma.min()
    ma = sigma.max()
    norm = matplotlib.colors.Normalize(vmin=mi,vmax=ma)
    cp = ax.contourf(X, Y, sigma.reshape(n_samples, n_samples), cmap="viridis", alpha=0.5, norm=norm)
    plt.colorbar(cp)
    ax.scatter(X_data[:, 0], X_data[:, 1], c='k', label="Data", marker="x")

    plt.xlim(lb_q_n, ub_q_n)
    plt.ylim(lb_q_mu, ub_q_mu)
    
    plt.xlabel("q_n")
    plt.ylabel("q_mu")
    plt.title("Uncertainty (std) and raw data")

    plt.show()

    # print the resulting optimal hyperparameters of the GP kernel
    print("Optimal kernel hyperparameters:")
    print(gpr.kernel_)


if __name__ == "__main__":
    main()
