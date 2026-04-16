import numpy as np

# Input matrix: 3 samples, 2 features
X = np.array([[1,2],
              [3,4],
              [5,6]])

# Weights vector: 2 features → 1 output
W = np.array([0.5, 1.0])

# Bias
b = 0.2

# Compute predictions
Y = np.dot(X, W) + b
print("Predicted Y:", Y)

# Transpose example
X_T = X.T
print("Transpose of X:\n", X_T)