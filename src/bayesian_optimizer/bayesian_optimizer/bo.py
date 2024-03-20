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


class BayesianOptimizer:

    def __init__(self, gp, beta):
        self.gp = gp
        self.beta = beta

    def aquisition_function(self, X):
        """
        Sample acquisition function at X

        Args:
        X: sampling points of shape (N, n_x)

        Returns:
        x_min_acq: minimum of the acquisition function for the sampled points of shape (n_x,)
        y_min: minimum value of the acquisition function
        y_hat: sampled values of the acquisition function of shape (N,)
        """

        # predict mean and standard deviation of the GP at the sampling points
        y, std = self.gp.predict(X)

        # compute the acquisition function
        y_hat = y - self.beta * std

        # minimize the acquisition function
        idx_min = np.argmin(y_hat)
        x_min_acq = X[idx_min, :].reshape(-1, 1)
        y_min = y_hat[idx_min]

        return x_min_acq, y_min, y_hat

    def get_estimate(self, X):
        """
        Sample GP at provided sampling points X and return best parameter estimate,
        expected value of learned function, and 95% confidence interval

        Args:
        X: sampling points of shape (N, n_x)
        """
        y, std = self.gp.predict(X)

        mean_min, min_idx = (np.min(y), np.argmin(y))
        theta = X[min_idx]
        conf = [mean_min - 1.96 * std[min_idx], mean_min + 1.96 * std[min_idx]]

        return theta, mean_min, conf

    def sample_gp(self, X):
        """
        Sample GP at sampling points X and return y, y_ucb, y_lcb

        Args:
        X: sampling points of shape (N, n_x)

        Returns:
        y: sampled values of the GP of shape (N,)
        y_ucb: sampled values of the GP upper confidence bound of shape (N,)
        y_lcb: sampled values of the GP lower confidence bound of shape (N,)
        """
        y, std = self.gp.predict(X)
        y_ucb = y + 1.96 * std
        y_lcb = y - 1.96 * std
        return y, y_ucb, y_lcb

    def add_data(self, x, y):
        """
        Args:
        x: input data of shape (N, n_x)
        y: output data of shape (N,)
        """
        self.gp.add_data(x, y)

    def get_data(self):
        return self.gp.get_data()

    @property
    def has_data(self):
        return self.gp.has_data
