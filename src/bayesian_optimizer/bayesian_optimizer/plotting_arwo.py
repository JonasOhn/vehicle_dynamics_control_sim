import bo
import gp
import matplotlib.pyplot as plt
import numpy as np


def h(x):
    return -x**2 * np.sin(5 * np.pi * x)**6

def eps(mean, std):
    return np.random.normal(mean, std)

def plot_1d_bo(X_data, Y_data, X_true, Y_true, X_acq, Y_acq, x_acq_opt, y_acq_opt, X_fit_gp, Y_fit_gp, Y_fit_gp_ucb, Y_fit_gp_lcb, savefig_path=None):
    plt.figure()

    if X_fit_gp is not None and Y_fit_gp is not None:
      plt.plot(X_fit_gp, Y_fit_gp, 'blue', label='GP fit')
    if X_acq is not None and Y_acq is not None:
      plt.plot(X_acq, Y_acq, 'red', label='acquisition function')
    if X_true is not None and Y_true is not None:
      plt.plot(X_true, Y_true, 'green', label='true function')
    if X_data is not None and Y_data is not None:
      # plot the data in teal color
      plt.scatter(X_data, Y_data, c='green', label='data')
    if x_acq_opt is not None and y_acq_opt is not None:
      plt.scatter(x_acq_opt, y_acq_opt, c='red', label='next sample')
    
    # if both upper and lower confidence bounds are provided, plot them
    # using a shaded area
    if Y_fit_gp_ucb is not None and Y_fit_gp_lcb is not None:
        plt.fill_between(X_fit_gp.flatten(), Y_fit_gp_ucb.flatten(), Y_fit_gp_lcb.flatten(), alpha=0.3, color='blue')

    if X_fit_gp is not None and Y_fit_gp_ucb is not None:
      plt.plot(X_fit_gp, Y_fit_gp_ucb, 'b--', label='GP UCB (95%)')
    if X_fit_gp is not None and Y_fit_gp_lcb is not None:
      plt.plot(X_fit_gp, Y_fit_gp_lcb, 'b--', label='GP LCB (95%)')
    
    plt.legend()

    plt.xlabel('x')
    plt.ylabel('y')

    plt.ylim(-1.5, 1.5)
    plt.xlim(0, 1)

    plt.title('Bayesian Optimization')

    if savefig_path is not None:
        plt.savefig(savefig_path)

    plt.show()


def main():
    
    real_noise_var = 0.001
    real_noise_mean = 0

    # initialize the GP and the Bayesian optimizer
    gp_noise_var = 0.005
    lengthscale = 0.1
    output_variance = 1
    gauss_process = gp.GaussianProcess(gp_noise_var, lengthscale, output_variance)
    bopt = bo.BayesianOptimizer(gauss_process, 1)

    x_lb = 0
    x_ub = 1

    # ground truth (not known)
    X_true = np.linspace(x_lb, x_ub, 110).reshape(-1,1)
    Y_true = h(X_true)

    plot_1d_bo(None, None, 
               X_true, Y_true, 
               None, None, None, None, 
               None, None, None, None, 
               savefig_path='bopt_{}.png'.format(0))

    # first sample uniformly at random
    x_to_sample = np.random.uniform(x_lb, x_ub, 1).reshape(-1,1)
    
    X_data = None
    Y_data = None

    # simulate data acquisition process and save the figure
    for i in range(20):
        
        # sample true function
        y = h(x_to_sample) + eps(real_noise_mean, real_noise_var)

        # add data to the GP
        bopt.add_data(x_to_sample, y)

        # fit GP
        X_fit_gp = np.linspace(x_lb, x_ub, 95).reshape(-1,1)
        Y_fit_gp, Y_fit_gp_ucb, Y_fit_gp_lcb = bopt.sample_gp(X_fit_gp)

        # find next sample
        X_acq = np.linspace(x_lb, x_ub, 100).reshape(-1,1)
        x_acq_opt, y_acq_opt, Y_acq = bopt.aquisition_function(X_acq)
        x_to_sample = x_acq_opt
        
        X_data, Y_data = bopt.get_data()

        plot_1d_bo(X_data, Y_data, 
                  X_true, Y_true, 
                  X_acq, Y_acq, x_acq_opt, y_acq_opt, 
                  X_fit_gp, Y_fit_gp, Y_fit_gp_ucb, Y_fit_gp_lcb, 
                  savefig_path='bopt_{}.png'.format(i+1))



if __name__ == '__main__':
    main()
