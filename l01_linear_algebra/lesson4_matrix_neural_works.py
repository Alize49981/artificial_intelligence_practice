import numpy as np

# Inputs (2 features)
inputs = np.array([1,2])

# Weights (2 features × 3 neurons)
weights = np.array([[0.5,0.3,0.1],
                    [0.2,0.7,0.9]])

# Bias (optional)
bias = np.array([0,0,0])

# Compute output
output = np.dot(inputs, weights) + bias
print("Layer output:", output)