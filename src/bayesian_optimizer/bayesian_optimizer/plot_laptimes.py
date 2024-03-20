import matplotlib.pyplot as plt
import numpy as np
import yaml

# load config
config_bo_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/config/optimizer_params.yaml"

with open(config_bo_filepath, "r") as file:
    config_bo = yaml.safe_load(file)
    config_bo = config_bo["/bayesian_optimizer_node"]["ros__parameters"]

gp_mean = config_bo["gp_mean"]

# load the data
results_csv_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/results/results.csv"
data = np.loadtxt(results_csv_filepath, delimiter=",")
laptimes_normalized = data[:, 3]

# # remove all laptimes that are larger than -3
# laptimes_normalized = laptimes_normalized[laptimes_normalized < -3]

# plot the laptimes
plt.scatter(range(len(laptimes_normalized)), laptimes_normalized + gp_mean, label="Laptimes")

# emphasize the best laptimes
best_laptime_idx = np.argmin(laptimes_normalized)
# print best laptime
print("Best laptime: ", laptimes_normalized[best_laptime_idx] + gp_mean, " s")
plt.scatter(best_laptime_idx, laptimes_normalized[best_laptime_idx] + gp_mean, label="Best laptime", color="red")

plt.xlabel("Iteration")
plt.ylabel("Laptime [s]")
plt.title("Laptimes")

# plot the mean and std of the laptimes by moving a window over the laptimes
window_size = 10
mean_laptimes = []
std_laptimes = []
for i in range(len(laptimes_normalized) - window_size):
    mean_laptimes.append(np.mean(laptimes_normalized[i:i+window_size]))
    std_laptimes.append(np.std(laptimes_normalized[i:i+window_size]))

mean_laptimes = np.array(mean_laptimes)
std_laptimes = np.array(std_laptimes)

plt.plot(range(len(mean_laptimes)), mean_laptimes + gp_mean, label="Mean laptimes", color="green")
plt.fill_between(range(len(mean_laptimes)), np.array(mean_laptimes) + np.array(std_laptimes) + gp_mean, np.array(mean_laptimes) - np.array(std_laptimes) + gp_mean, alpha=0.5, label="Std laptimes", color="green")

plt.legend()



plt.show()