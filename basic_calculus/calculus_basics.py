#Functions
# Simple "AI model"
def model(x):
    return 2*x + 3

print(model(5))  

#Plotting Functions
import numpy as np
import matplotlib.pyplot as plt

# Create x values
x = np.linspace(-10, 10, 100)

# Function
y = 2*x + 3

# Plot
plt.plot(x, y)
plt.title("y = 2x + 3")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

plt.show()
#Limits
import numpy as np

def f(x):
    return x**2

# Approach 2
for x in [1.9, 1.99, 1.999]:
    print(x, f(x))

#Derivatives
def f(x):
    return x**2

def derivative(x):
    h = 0.0001
    return (f(x + h) - f(x)) / h

print(derivative(2))  
#artial Derivatives (Multivariable AI)
def f(x, y):
    return x**2 + y**2

def partial_x(x, y):
    h = 0.0001
    return (f(x+h, y) - f(x, y)) / h

print(partial_x(2, 3))  


#gradient
def grad(x, y):
    return (2*x, 2*y)

print(grad(2, 3))  

#Chain Rule
def f(x):
    return (x**2 + 1)**3

def derivative(x):
    h = 0.0001
    return (f(x + h) - f(x)) / h

print(derivative(2))