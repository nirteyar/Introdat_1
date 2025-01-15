import numpy as np
import matplotlib.pyplot as plt

def confidence_interval_length(alpha, n_values):
    lengths = []
    for n in n_values:
        # Calculate epsilon_n
        epsilon_n = np.sqrt(1 / (2 * n) * np.log(2 / alpha))
        # Length of the confidence interval is 2 * epsilon_n
        length = 2 * epsilon_n
        lengths.append(length)
    return lengths

# Parameters for the plot
alpha = 0.05
n_values = [10, 100, 1000, 10000]

# Calculate the lengths of the confidence intervals
interval_lengths = confidence_interval_length(alpha, n_values)

# Plot the lengths of the confidence intervals as a function of n
plt.plot(n_values, interval_lengths, marker='o')
plt.xscale('log')
plt.xlabel('Sample Size (n)')
plt.ylabel('Length of Confidence Interval')
plt.title('Length of Confidence Interval as a Function of Sample Size')
plt.grid(True)
plt.show()

# Output the lengths of the confidence intervals
print("Lengths of the confidence intervals:", interval_lengths)
