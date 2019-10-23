'''
Brian Kaine Margretta
10/22/19
CAP4630 Artificial Intelligence
Homework #1 Part A
I chose to one function that has all the required features. 
Local Maximum @ [1.59,1.56]  = dark black
Local Minimum @ [-1.59,-1.56] = dark red
Neither Min nor Max but gradient is 0 @ [-1.47,1.49]  = white
'''
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
from mpl_toolkits import mplot3d


def f1(x,y): #r2 -> r1 function
  return np.sin(x) + np.sin(y)

x = np.linspace(-3, 3, 50) # X Linespace for density/contour plot
y = np.linspace(-3, 3, 50) # Y Linespace for density/contour plot
X, Y = np.meshgrid(x, y) #Create a rectangular grid out of array of x and y values
Z = f1(X, Y) # Call function for density/contour

contours = plt.contour(X, Y, Z, 3, colors='black') # plot X,Y,Z
plt.clabel(contours, inline=True, fontsize=8) # Label values within plot
plt.imshow(Z, extent=[-3, 3, -3, 3], origin='lower', cmap='RdGy', alpha=1) #Add color gradientw
plt.colorbar(); #Add key on side; Red = minimum, Black = maximum
