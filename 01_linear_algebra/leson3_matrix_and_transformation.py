import numpy as np

# Create matrices
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

# Transpose
print("A Transpose:\n", A.T)

# Addition
print("A + B:\n", A + B)

# Scalar multiplication
print("2 * A:\n", 2*A)

# Matrix multiplication
C = np.dot(A, B)
print("A * B:\n", C)

# Identity matrix
I = np.eye(2)
print("Identity Matrix:\n", I)

# Inverse matrix
A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)

# Check multiplication
print("A * A_inv:\n", np.dot(A, A_inv))

inputs = np.array([1,2])
weights = np.array([[0.5,0.3],
                    [0.2,0.7]])

output = np.dot(inputs, weights)
print("Layer output:", output)