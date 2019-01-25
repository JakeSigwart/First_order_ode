# The Adams-Moulton Method of 4th order is a predictor-corrector method
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
y_star = np.zeros_like(x)
y[0] = 0

# Function defining the ODE: f(x,y) = y'
def f(x,y):
	return 2*x - y + 1

# The Runge-Kutta Method is used to generate the 1st 4 terms
for n in range(0, 3):
	k1 = dx*f(x[n], y[n])
	k2 = dx*f(x[n]+.5*dx, y[n]+.5*k1)
	k3 = dx*f(x[n]+.5*dx, y[n]+.5*k2)
	k4 = dx*f(x[n]+dx, y[n]+k3)
	y[n+1] = y[n] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
	
# Adams-Bashforth Method
for n in range(3, nx-1):
	y_star[n+1] = y[n] + (1/24)*dx*(55*f(x[n],y[n]) - 59*f(x[n-1],y[n-1]) + 37*f(x[n-2],y[n-2]) - 9*f(x[n-3],y[n-3]))
	y[n+1] = y[n] + (1/24)*dx*(9*f(x[n+1],y_star[n+1]) + 19*f(x[n],y[n]) - 5*f(x[n-1],y[n-1]) + f(x[n-2],y[n-2]))

# Actual Solution: y(x) = e^(-x)+2x-1
y_actual = np.zeros_like(x)
for i in range(0, len(y_actual)):
	y_actual[i] = math.exp(-x[i])+2*x[i]-1

# Plot the approximated and exact solutions
plt.plot(x, y, 'r', x, y_actual, 'b')
plt.title("Adams-Moulton Method Solution to: y'=2x-y+1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['A-M', 'Actual'])
plt.show()

print("Percent Error at x=1: "+str( abs( (y[-1]-y_actual[-1])/y_actual[-1])*100)+" %")

