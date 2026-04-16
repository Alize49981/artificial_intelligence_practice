import numpy as np

# Example dataset: email lengths
emails = [120, 85, 100, 200, 150, 90]

# Mean
mean_length = np.mean(emails)
print("Mean email length:", mean_length)

# Median
median_length = np.median(emails)
print("Median email length:", median_length)

# Variance
variance_length = np.var(emails)
print("Variance:", variance_length)

# Standard deviation
std_length = np.std(emails)
print("Standard Deviation:", std_length)