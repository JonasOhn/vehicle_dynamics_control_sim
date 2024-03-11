import numpy as np
from bo import BayesianOptimizer
from gp import GaussianProcess
import matplotlib.pyplot as plt


def plot_gp(bayesian_optimizer, gp_mean, lb_q_sd, ub_q_sd, lb_q_n, ub_q_n, lb_q_mu, ub_q_mu):
  X, y = bayesian_optimizer.get_data()
  s = y + gp_mean
  fig = plt.figure(1)
  ax = fig.add_subplot(111, projection='3d')

  # Scatter plot
  sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=s, cmap='rainbow')

  # Set labels and title
  ax.set_xlabel('q_sd')
  ax.set_ylabel('q_n')
  ax.set_zlabel('q_mu')
  ax.set_title('Cost Function Scatter Plot')

  ax.set_xlim(lb_q_sd, ub_q_sd)
  ax.set_ylim(lb_q_n, ub_q_n)
  ax.set_zlim(lb_q_mu, ub_q_mu)

  # Add color bar
  cbar = fig.colorbar(sc)
  cbar.set_label('cone-penalized lap time')

  plt.tight_layout()

def plot_acquisition_function(bayesian_optimizer, Q, lb_q_sd, ub_q_sd, lb_q_n, ub_q_n, lb_q_mu, ub_q_mu):

  fig = plt.figure(2)
  ax = fig.add_subplot(111, projection='3d')

  # Scatter plot
  q_star, q_hat = bayesian_optimizer.aquisition_function(Q)
  sc = ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=q_hat, cmap='rainbow')
  sc_star = ax.scatter(q_star[0], q_star[1], q_star[2], color='red', marker='*', s=100)  # Red star marker at q_star

  # Optionally, you may want to add a legend to distinguish the red star marker
  ax.legend([sc, sc_star], ['Points', 'q_star'], loc='upper right')


  # Set labels and title
  ax.set_xlabel('q_sd')
  ax.set_ylabel('q_n')
  ax.set_zlabel('q_mu')
  ax.set_title('Acquisition Function Scatter Plot')

  ax.set_xlim(lb_q_sd, ub_q_sd)
  ax.set_ylim(lb_q_n, ub_q_n)
  ax.set_zlim(lb_q_mu, ub_q_mu)

  # Add color bar
  cbar = fig.colorbar(sc)
  cbar.set_label('acquisition function')

  plt.tight_layout()

def load_data(bayesian_optimizer, results_csv_filepath):
  # try to load data from .csv file, if it exists. else, start with empty data and print to console
  try:
    data = np.loadtxt(results_csv_filepath, delimiter=",")
    bayesian_optimizer.add_data(data[:, 0:3], data[:, 3])
  except:
    print("No data found in .csv file. Starting with empty data.")




results_csv_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/results/results.csv"

# define bounds on the decision variables
lb_q_sd = 0.01
ub_q_sd = 2.0

lb_q_n = 0.01
ub_q_n = 2.0

lb_q_mu = 0.01
ub_q_mu = 2.0

num_acqfct_samples_perdim = 10

Q = np.zeros((num_acqfct_samples_perdim, 3))
q_sd = np.linspace(start=lb_q_sd, stop=ub_q_sd, num=num_acqfct_samples_perdim)
q_n = np.linspace(start=lb_q_n, stop=ub_q_n, num=num_acqfct_samples_perdim)
q_mu = np.linspace(start=lb_q_mu, stop=ub_q_mu, num=num_acqfct_samples_perdim)

# Generate meshgrid
X, Y, Z = np.meshgrid(q_sd, q_n, q_mu, indexing='ij')

# Reshape meshgrid to form a matrix where each column represents one parameter dimension
Q = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

gp_noise_covariance = 0.001
gp_lengthscale = 5.0
gp_output_variance = 1.0
gp_mean = 40.0
beta = 0.5

gp = GaussianProcess(noise_covariance=gp_noise_covariance,
                      lengthscale=gp_lengthscale,
                      output_variance=gp_output_variance)

bayesian_optimizer = BayesianOptimizer(gp=gp, beta=beta)

load_data(bayesian_optimizer, results_csv_filepath)

plot_acquisition_function(bayesian_optimizer, Q, lb_q_sd, ub_q_sd, lb_q_n, ub_q_n, lb_q_mu, ub_q_mu)

plot_gp(bayesian_optimizer, gp_mean, lb_q_sd, ub_q_sd, lb_q_n, ub_q_n, lb_q_mu, ub_q_mu)

plt.show()
