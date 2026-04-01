import numpy as np

# Define a matrix
A = np.array([[2,0],
              [0,3]])

# Compute eigenvalues and eigenvectors
eig_values, eig_vectors = np.linalg.eig(A)

print("Eigenvalues:", eig_values)
print("Eigenvectors:\n", eig_vectors)