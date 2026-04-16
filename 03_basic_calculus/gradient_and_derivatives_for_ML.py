import numpy as np

# Example function f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Partial derivatives
def grad_f(x, y):
    df_dx = 2*x
    df_dy = 2*y
    return np.array([df_dx, df_dy])

# Compute gradient at point (3,4)
gradient = grad_f(3,4)
print("Gradient at (3,4):", gradient)