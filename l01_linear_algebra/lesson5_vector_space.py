import numpy as np

# Define vectors
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

# Vector addition
v_sum = v1 + v2
print("v1 + v2 =", v_sum)

# Scalar multiplication
v_scaled = 2 * v1
print("2 * v1 =", v_scaled)

# Dot product
dot = np.dot(v1, v2)
print("v1 • v2 =", dot)

# Magnitude
magnitude = np.linalg.norm(v1)
print("||v1|| =", magnitude)

# Unit vector
unit = v1 / magnitude
print("Unit vector of v1 =", unit)
# Example vectors for two words
word1 = np.array([1,0,1])
word2 = np.array([0,1,1])

# Cosine similarity
cos_sim = np.dot(word1, word2) / (np.linalg.norm(word1)*np.linalg.norm(word2))
print("Cosine similarity:", cos_sim)