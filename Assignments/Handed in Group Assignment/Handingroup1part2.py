import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.05
p_true = 0.4
n_values = [10, 100, 1000, 10000]
num_simulations = 10000

def coverage_simulation(alpha, p_true, n_values, num_simulations):
    coverage_probabilities = []
    
    for n in n_values:
        coverage_count = 0
        
        for _ in range(num_simulations):
            # Generate n IID Bernoulli samples with true proportion p_true
            samples = np.random.binomial(1, p_true, n)
            # Compute the sample mean
            p_hat = np.mean(samples)
            # Compute epsilon_n
            epsilon_n = np.sqrt(1 / (2 * n) * np.log(2 / alpha))
            # Define the confidence interval
            lower_bound = p_hat - epsilon_n
            upper_bound = p_hat + epsilon_n
            # Check if the true proportion p_true is within the confidence interval
            if lower_bound <= p_true <= upper_bound:
                coverage_count += 1
        
        # Compute the coverage probability
        coverage_probability = coverage_count / num_simulations
        coverage_probabilities.append(coverage_probability)
    
    return coverage_probabilities

# Run the simulation
coverage_results = coverage_simulation(alpha, p_true, n_values, num_simulations)

# Plot the coverage as a function of n
plt.plot(n_values, coverage_results, marker='o')
plt.xscale('log')
plt.xlabel('Sample Size (n)')
plt.ylabel('Coverage Probability')
plt.title('Coverage Probability of Confidence Intervals as a Function of Sample Size')
plt.grid(True)
plt.show()

coverage_results  # Display the coverage results
