'''
Brian Kaine Margretta
10/22/19
CAP4630 Artificial Intelligence
Homework #1 Part B
'''
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(25, 45)
mpl.rc('axes', labelsize=9)
mpl.rc('xtick', labelsize=9)
mpl.rc('ytick', labelsize=9)

np.random.seed(42)

xs = 2 * np.random.rand(100, 1)
ys = 4 + 3 * xs + np.random.rand(100, 1)


plt.plot(xs, ys, "b.")
plt.xlabel("$x_1$", fontsize=1)
plt.ylabel("$y$", rotation=0, fontsize=1)
plt.axis([0, 2, 6, 10])
plt.show()



# split the data into training and test sets
# train set
train_xs = xs[:80]
train_ys = ys[:80]
# test set
test_xs = xs[80:]
test_ys = ys[80:]

# number of epochs
epochs = 10
# learning rate
lr = 0.01

# initial value for weights w1, w2, and bias b
w1 = np.random.randn(1)
w2 = np.random.randn(1)
b = np.zeros(1)

for epoch in np.arange(epochs):
  for i in np.arange(80):
    y_pred = w1 * train_xs[i] + w2 * train_xs[i] + b # Extend y prediction to include w2 weights
    
    grad_w = (y_pred - train_ys[i]) * train_xs[i]
    grad_b = (y_pred - train_ys[i])
    
    w1 -= lr * grad_w
    w2 -= lr * grad_w
    b -= lr * grad_b
    
test_loss = 0
for i in np.arange(20):
  test_loss += 0.5 * (w1 * test_xs[i] + w2 * test_xs[i] + b - test_ys[i]) ** 2 # Extend loss to include w2 weights
test_loss /= 20

test_loss

pred_ys = w1 * test_xs + w2 * test_xs + b # Extend prediction to include w2 weights

'''
Plot in 3D since there are now 2 weights + b 
'''
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(25, 25)
plt.plot(test_xs, test_ys, "b.")
plt.plot(test_xs, pred_ys, "r.") # predicted values
plt.xlabel("$x_1$", fontsize=1)
plt.ylabel("$y$", rotation=0, fontsize=1)
plt.axis([0, 1.05, 4, 9])
plt.show()
