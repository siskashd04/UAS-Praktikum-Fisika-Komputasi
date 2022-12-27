import numpy as np
import matplotlib.pyplot as plt

def trapezoid(f,a,b,n):
    h = (b-a)/n
    sum = 0.0

    for i in range (1,n):
        x = a+i*h
        sum = sum+f(x)
        integral = (h/2)*(f(a)+2*sum+f(b))
        return integral


def f(x):
    return (x**3)*np.sin(-2*x)

a = 1.0
b = 10.0
n = 100

integral=trapezoid(f,a,b,n)

print('integral =',integral)
