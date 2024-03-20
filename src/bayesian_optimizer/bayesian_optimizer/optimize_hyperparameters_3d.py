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
    results_csv_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/results/results.csv"
    config_bo_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/config/optimizer_params.yaml"

    with open(config_bo_filepath, "r") as file:
        config_bo = yaml.safe_load(file)
        config_bo = config_bo["/bayesian_optimizer_node"]["ros__parameters"]
    
    # --- initialize parameters
    lb_q_sd = 0.01
    ub_q_sd = 2.0
    lb_q_n = 0.01
    ub_q_n = 2.0
    lb_q_mu = 0.01
    ub_q_mu = 2.0

    # load the data
    data = np.loadtxt(results_csv_filepath, delimiter=",")

    gp_mean = config_bo["gp_mean"]

    # get the data
    X_data = data[:, 0:3]
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
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=10, normalize_y=False, random_state=42)

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
    n_samples = 10
    q_sd = np.linspace(start=lb_q_sd, stop=ub_q_sd, num=n_samples)
    q_n = np.linspace(start=lb_q_n, stop=ub_q_n, num=n_samples)
    q_mu = np.linspace(start=lb_q_mu, stop=ub_q_mu, num=n_samples)

    q_bounds = np.array([[lb_q_sd, ub_q_sd], [lb_q_n, ub_q_n], [lb_q_mu, ub_q_mu]])

    # Generate meshgrid
    X, Y, Z = np.meshgrid(q_sd, q_n, q_mu, indexing="ij")

    # Reshape meshgrid to form a matrix where each column represents one parameter dimension
    Q = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    # get the predictions
    y_pred, sigma = gpr.predict(Q, return_std=True)

    # plot the raw data in 3d using colormap for y
    fig = plt.figure(1)

    # plot the raw data only
    ax = fig.add_subplot(111, projection="3d")

    # plot predictions using filled contours
    sc = ax.scatter(X_data[:, 0], X_data[:, 1], X_data[:, 2], c=gp_mean + Y_data, cmap="viridis", label="Data")
    plt.colorbar(sc)

    plt.xlim(q_bounds[0][0], q_bounds[0][1])
    plt.ylim(q_bounds[1][0], q_bounds[1][1])
    ax.set_zlim(q_bounds[2][0], q_bounds[2][1])

    plt.xlabel("q_s")
    plt.ylabel("q_n")
    ax.set_zlabel("q_mu")
    plt.title("Raw data")



    fig = plt.figure(2)

    # plot the GP mean in 3d
    ax = fig.add_subplot(111, projection="3d")

    # plot predictions using filled contours
    # plot contour plot of the GP mean
    sc = ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=y_pred + gp_mean, cmap="viridis", label="Mean Prediction", alpha=0.5)
    plt.colorbar(sc)

    plt.xlim(q_bounds[0][0], q_bounds[0][1])
    plt.ylim(q_bounds[1][0], q_bounds[1][1])
    ax.set_zlim(q_bounds[2][0], q_bounds[2][1])

    plt.xlabel("q_s")
    plt.ylabel("q_n")
    ax.set_zlabel("q_mu")
    plt.title("Predictions")


    fig = plt.figure(3)

    # plot the uncertainty in 2d using colormap for y
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=sigma, cmap="viridis", label="Prediction Uncertainty", alpha=0.5)
    plt.colorbar(sc)

    plt.xlim(q_bounds[0][0], q_bounds[0][1])
    plt.ylim(q_bounds[1][0], q_bounds[1][1])
    ax.set_zlim(q_bounds[2][0], q_bounds[2][1])
    
    plt.xlabel("q_s")
    plt.ylabel("q_n")
    ax.set_zlabel("q_mu")
    plt.title("Uncertainty (std)")

    plt.show()

    # print the resulting optimal hyperparameters of the GP kernel
    print("Optimal kernel hyperparameters:")
    print(gpr.kernel_)


if __name__ == "__main__":
    main()
