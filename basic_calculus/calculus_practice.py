#DERIVATIVES
import sympy as sp

x = sp.symbols('x')
f = 3*x**2 + 2*x + 1

print(sp.diff(f, x))  

#NUMERICAL DERIVATIVE
def f(x):
    return x**2

def derivative(x):
    h = 0.0001
    return (f(x + h) - f(x)) / h

print(derivative(2)) 

#PARTIAL DERIVATIVES
x, y = sp.symbols('x y')
f = x**2 + 3*y**2

print(sp.diff(f, x))  
print(sp.diff(f, y)) 

#gradient_descent
x = 8   # start far
lr = 0.1

def grad(x):
    return 2*x

for i in range(10):
    x = x - lr * grad(x)
    print(x)