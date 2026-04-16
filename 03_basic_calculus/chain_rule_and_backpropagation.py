import numpy as np

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input data (2 samples, 2 features)
X = np.array([[0,0],[1,1]])
# True output
y = np.array([[0],[1]])

# Initialize weights randomly
np.random.seed(1)
weights = np.random.rand(2,1)

# Learning rate
lr = 0.1

# Forward + Backward Pass
for epoch in range(10):
    # Forward
    z = np.dot(X, weights)
    y_pred = sigmoid(z)
    
    # Loss derivative
    error = y - y_pred
    gradient = error * sigmoid_derivative(y_pred)
    
    # Weight update
    weights += np.dot(X.T, gradient) * lr

print("Updated Weights:\n", weights)