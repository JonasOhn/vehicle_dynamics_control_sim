'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2023, Alexandre Didier, Jérôme Sieber, Rahel Rickenbach and Shao (Mike) Zhang, ETH Zurich,
% {adidier,jsieber, rrahel}@ethz.ch
%
% All rights reserved.
%
% This code is only made available for students taking the advanced MPC 
% class in the fall semester of 2023 (151-0371-00L) and is NOT to be 
% distributed.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

import numpy as np

class BayesianOptimizer():

    def __init__(self, gp):
        self.gp = gp

    def aquisition_function(self, X):
        '''
        Sample acquisition function at provided sampling points X

        Args:
        X: sampling points of shape (N, n_x)

        Returns:
        theta: minimum of the acquisition function for the sampled points of shape (n_x,)
        y_hat: sampled values of the acquisition function of shape (N,)
        '''
        # --- start inserting code here ---
        ### TODO: implement acquisition function
        # Hint: use the GP class stored in self.gp

        #print(X.shape)

        y, std = self.gp.predict(X)

        beta = 1.0

        y_hat = y - beta * np.sqrt(std)
        #print(y_hat.shape)

        idx_min = np.argmin(y_hat)
        theta = X[idx_min, :].reshape(-1, 1)
        #print(theta.shape)

        # --- end inserting code here ---
        return theta, y_hat

    def get_estimate(self, X):
        '''
        Sample GP at provided sampling points X and return best parameter estimate,
        expected value of learned function, and 95% confidence interval

        Args:
        X: sampling points of shape (N, n_x)
        '''
        y, std = self.gp.predict(X)

        mean_min, min_idx = (np.min(y), np.argmin(y))
        theta = X[min_idx]
        conf = [mean_min - 1.96*std[min_idx], mean_min + 1.96*std[min_idx]]

        return theta, mean_min, conf

    def sample_gp(self, X):
        '''
        Sample GP at sampling points X

        Args:
        X: sampling points of shape (N, n_x)
        '''
        return self.gp.predict(X)

    def add_data(self, x, y):
        '''
        Args:
        x: input data of shape (N, n_x)
        y: output data of shape (N,)
        '''
        self.gp.add_data(x,y)

    def get_data(self):
        return self.gp.get_data()

    @property
    def has_data(self):
        return self.gp.has_data
