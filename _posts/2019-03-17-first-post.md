import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#First-order Taylor approximation for cubic functions

def f(x):
    return 2*x**3 + x**2 -7*x + 5  #Function definition
 
x = np.linspace(-7,7,24)   #Set range
plt.plot(x,f(x))    #Create Graph
plt.show()


plt.plot(x,f(x))      # (4,121) Coordinates
plt.plot(4, 121, "o")
plt.show()

def f2(x):
    return 97*x - 267  #(4,121) Definition of First - order Talyor approximation
    
    
plt.plot(x,f(x))
plt.plot(x,f2(x))
plt.plot(4, 121, "o")
plt.show()
