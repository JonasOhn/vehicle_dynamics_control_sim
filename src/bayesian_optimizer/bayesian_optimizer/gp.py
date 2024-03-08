

import numpy as np

class GaussianProcess():
    '''
    Gaussian Process Regressor

    Internally stores the input data_x as an array of shape (N,n_x)
    and the output data_y as an array of shape (N,), i.e., the output
    dimension cannot be greater than 1.

    with N the number of data points
    '''
    def __init__(self, noise_covariance, lengthscale, output_variance):
        self.noise_covariance_ = noise_covariance
        self.lengthscale_ = lengthscale
        self.output_variance_ = output_variance

    def squared_exponential_kernel(self, X1, X2):
      '''
      Args:
      X1 (N1, n_x) first set of feature vectors
      X2 (N2, n_x) second set of feature vectors
      lam scaling factor
      sigma lengthscale

      Returns:
      K -- numpy array of shape (N1, N2) representing the kernel matrix
      where K[i,j] is the kernel evaluated for X1[i,:] and X2[j,:]
      '''
      # --- start inserting code here ---

      # get number of samples
      N1 = X1.shape[0]
      N2 = X2.shape[0]

      K = np.zeros((N1, N2))
      for i in range(N1):
          for j in range(N2):
              K[i, j] = self.output_variance_ * np.exp(-1/( 2 * (self.lengthscale_**2)) * np.linalg.norm(X1[i, :] - X2[j, :], 2)**2)

      # print(K)
      return K

    def add_data(self, x:np.array, y:np.array):
        '''
        Args:
        x: input data of shape (N, n_x)
        y: output data of shape (N, 1)
        where N is the number of data points
        '''
        assert x.shape[0] == y.shape[0], \
            'Number of data points in input x ({0}) and output y ({1}) must match!'.format(x.shape[0], y.shape[0])
        if not self.has_data:
            assert len(x.shape) == 2, 'input data must be of shape (N,n_x)'
            self.x_dim = x.shape[1]
            self.data_x_ = x
            self.data_y_ = y.reshape(-1,1)
        else:
            assert len(x.shape) == 2, 'input data must be of shape (N,n_x)'
            assert x.shape[1] == self.x_dim, \
                'Trying to add input data of dimension {0} to a GP with input data of dimemsion {1}'.format(x.shape[1], self.x_dim)
            self.data_x_ = np.vstack([self.data_x_, x])
            self.data_y_ = np.vstack([self.data_y_, y.reshape(-1,1)])

    def predict(self, x):
        '''
        Args:
        x: input data of shape (N, n_x)
        where N is the number of data points

        Returns:
        y: mean output at the input data points of shape (N,1)
        std: standard deviation of the output at the input data points of shape (N,1)
        '''
        if not self.has_data:
            raise RuntimeError('Cannot make prediction because GP has no data!')

        assert len(x.shape) == 2, 'input data must be of shape (N,n_x)'
        assert x.shape[1] == self.x_dim, \
            'input data dimension {0} does not match the GP input data dimension {1}'.format(x.shape[1], self.x_dim)

        inv_K_noise = np.linalg.inv(
            self.squared_exponential_kernel(self.data_x_, self.data_x_) + self.noise_covariance_**2 * np.eye(len(self.data_x_))
        )
        k = self.squared_exponential_kernel(x, self.data_x_)

        y = k @ inv_K_noise @ self.data_y_

        # ignore cross correlation terms, only get point estimate uncertainty
        std = np.sqrt(np.diagonal(self.squared_exponential_kernel(x,x) - k @ inv_K_noise @ k.T)).reshape(-1,1)

        return y, std

    def get_data(self):
        return self.data_x_, self.data_y_

    @property
    def has_data(self):
        return hasattr(self, 'data_x_') and hasattr(self, 'data_y_')
