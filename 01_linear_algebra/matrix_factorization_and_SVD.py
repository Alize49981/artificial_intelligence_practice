import numpy as np

# Sample matrix
A = np.array([[1,2],
              [3,4],
              [5,6]])

# Perform SVD
U, S, Vt = np.linalg.svd(A)

print("U matrix:\n", U)
print("Singular values:", S)
print("V^T matrix:\n", Vt)

# Reconstruct matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = np.diag(S)
A_reconstructed = U @ Sigma @ Vt
print("Reconstructed A:\n", A_reconstructed)