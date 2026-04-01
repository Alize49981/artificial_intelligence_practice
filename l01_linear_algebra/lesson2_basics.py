import numpy as np

# Vectors
v1 = np.array([80, 90, 70])
weights = np.array([0.3, 0.4, 0.3])

# Dot product (AI prediction)
score = np.dot(v1, weights)
print("Predicted Score:", score)