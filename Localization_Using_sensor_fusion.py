import numpy as np
import matplotlib.pyplot as plt

# import the dataset
dataset = np.load('dataset.npz')
t = dataset['t'] # 12609 X 1 timesamples
x_true = dataset['x_true'] # 12609 X 1 true x-position
y_true = dataset['y_true'] # 12609 X 1 true y-position
th_true = dataset['th_true'] # 12609 X 1 true theta-position
l = dataset['l'] # 17 X 2 position of 17 landmarks
r = dataset['r'] # 12609 X 17 something complicated
r_var = dataset['r_var'] #
b = dataset['b'] # 12609 X 17
b_var = dataset['b_var']
v = dataset['v'] # 12609 X 1 speed
v_var = dataset['v_var'] #
om = dataset['om'] # 12609 X 1 rotational speed
om_var = dataset['om_var'] #
d = dataset['d'] # the distance between the center of the robot and the laser rangefinder

plt.plot(x_true,y_true)
plt.show()
