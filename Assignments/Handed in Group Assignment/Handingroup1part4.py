import numpy as np
import matplotlib.pyplot as plt

def decision_correct_simulation(alpha, p_new, n_values, num_simulations):
    correct_decision_probabilities = []
    
    for n in n_values:
        correct_count = 0
        
        for _ in range(num_simulations):
            # Generate n IID Bernoulli samples with the new proportion p_new
            samples = np.random.binomial(1, p_new, n)
            # Compute the sample mean
            p_hat = np.mean(samples)
            # Compute epsilon_n
            epsilon_n = np.sqrt(1 / (2 * n) * np.log(2 / alpha))
            # Define the confidence interval
            lower_bound = p_hat - epsilon_n
            upper_bound = p_hat + epsilon_n
            # Check if the new proportion p_new is within the confidence interval
            if lower_bound <= p_new <= upper_bound:
                correct_count += 1
        
        # Compute the probability that the decision is correct
        correct_decision_probability = correct_count / num_simulations
        correct_decision_probabilities.append(correct_decision_probability)
    
    return correct_decision_probabilities

# Parameters for the changed proportion
alpha = 0.05
p_changed = 0.5
n_values = [10, 100, 1000, 10000]
num_simulations = 10000

# Run the simulation for the changed proportion
correct_decision_results = decision_correct_simulation(alpha, p_changed, n_values, num_simulations)

# Plot the probability of a correct decision as a function of n
plt.plot(n_values, correct_decision_results, marker='o')
plt.xscale('log')
plt.xlabel('Sample Size (n)')
plt.ylabel('Probability of Correct Decision')
plt.title('Probability of Correct Decision as a Function of Sample Size (p = 0.5)')
plt.grid(True)
plt.show()

# Output the probabilities of correct decisions
print("Probabilities of correct decisions:", correct_decision_results)
