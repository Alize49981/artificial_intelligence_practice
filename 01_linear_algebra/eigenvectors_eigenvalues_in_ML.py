import numpy as np

# Sample dataset: 3 samples, 2 features
X = np.array([[2,0],
              [0,1],
              [3,1]])

# Center data
X_centered = X - np.mean(X, axis=0)

# Compute covariance matrix
cov_matrix = np.cov(X_centered.T)

# Eigenvalues and eigenvectors
eig_values, eig_vectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:", eig_values)
print("Eigenvectors:\n", eig_vectors)

# Project data onto first principal component
X_reduced = X_centered @ eig_vectors[:,0].reshape(-1,1)
print("Data projected onto first PC:\n", X_reduced)