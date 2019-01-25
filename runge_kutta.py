
# The Runge-Kutta Method is a 1-step, 4th order accurate method
# Solving:  y' = 2x - y + 1		y(0) = 0

import math
import numpy as np
import matplotlib.pyplot as plt

dx = 0.01
xmin = 0
xmax = 1
nx = int((xmax-xmin)/dx)
x  = np.linspace(0, 1, nx)
y = np.zeros_like(x)
y[0] = 0

# Function defining the ODE: f(x,y) = y'
def f(x,y):
	return 2*x - y + 1

# Runge-Kutta Method
for n in range(0, nx-1):
	k1 = dx*f(x[n], y[n])
	k2 = dx*f(x[n]+.5*dx, y[n]+.5*k1)
	k3 = dx*f(x[n]+.5*dx, y[n]+.5*k2)
	k4 = dx*f(x[n]+dx, y[n]+k3)
	y[n+1] = y[n] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
	
# Actual Solution: y(x) = e^(-x)+2x-1
y_actual = np.zeros_like(x)
for i in range(0, len(y_actual)):
	y_actual[i] = math.exp(-x[i])+2*x[i]-1

# Plot the 
plt.plot(x, y, 'r', x, y_actual, 'b')
plt.title("Runge-Kutta Method Solution to: y'=2x-y+1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['R-K', 'Actual'])
plt.show()

print("Percent Error at x=1: "+str( abs( (y[-1]-y_actual[-1])/y_actual[-1])*100)+" %")


