import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#First-order Taylor approximation for cubic functions

def f(x):
    return 2*x**3 + x**2 -7*x + 5  #Function definition
 
x = np.linspace(-7,7,24)   #Set range
plt.plot(x,f(x))    #Create Graph
plt.show()


<img width="262" alt="캡처" src="https://user-images.githubusercontent.com/28971360/54863635-7ed39100-4d8e-11e9-86cd-83db0cc7f8c1.PNG">



plt.plot(x,f(x))      # (4,121) Coordinates
plt.plot(4, 121, "o")
plt.show()


<img width="257" alt="캡처1" src="https://user-images.githubusercontent.com/28971360/54863638-87c46280-4d8e-11e9-9edd-4daf1513c733.PNG">


def f2(x):
    return 97*x - 267  #(4,121) Definition of First - order Talyor approximation
    
    
plt.plot(x,f(x))
plt.plot(x,f2(x))
plt.plot(4, 121, "o")
plt.show()


<img width="238" alt="캡처2" src="https://user-images.githubusercontent.com/28971360/54863643-8f840700-4d8e-11e9-8a09-7d475794ad0e.PNG">


