
# The Improved Euler Method is a 2-step method
# Solving:  y' = 2x*y  ,  y(0) = 1

import math
import numpy as np
import matplotlib.pyplot as plt

dx = 0.01
xmin = 0
xmax = 1
nx = int((xmax-xmin)/dx)
x  = np.linspace(0, 1, nx)
y = np.zeros_like(x)
y[0] = 1

# Function defining the ODE: f(x,y) = y'
def f(x,y):
	return 2*x*y

# Improved Euler Method
for n in range(0, nx-1):
	k1 = dx*f(x[n], y[n])
	k2 = dx*f(x[n+1], y[n]+k1)
	y[n+1] = y[n] + (1/2)*(k1 + k2)
	
# Calculate Actual Solution
y_actual = np.zeros_like(x)
for i in range(0, len(y_actual)):
	y_actual[i] = math.exp(x[i]**2)

# Plot the Numerical and Actual Solutions
plt.plot(x, y, 'r', x, y_actual, 'b')
plt.title("Improved Euler Method Solution to: y'=2xy")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['Improved Euler', 'Actual'])
plt.show()

print("Percent Error at x=1: "+str( abs( (y[-1]-y_actual[-1])/y_actual[-1])*100)+" %")

