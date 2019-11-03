import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# import the dataset
dataset = np.load('dataset.npz')
dt = 0.1
t = dataset['t'] # 12609 X 1 timesamples
x_true = dataset['x_true'] # 12609 X 1 true x-position
y_true = dataset['y_true'] # 12609 X 1 true y-position
th_true = dataset['th_true'] # 12609 X 1 true theta-position
l = dataset['l'] # 17 X 2 position of 17 landmarks
r = dataset['r'] # 12609 X 17 something complicated
r_var = dataset['r_var'] # 1
b = dataset['b'] # 12609 X 17 angle
b_var = dataset['b_var'] # 1
v = dataset['v'] # 12609 X 1 speed
v_var = dataset['v_var'] # 1
om = dataset['om'] # 12609 X 1 rotational speed
om_var = dataset['om_var'] # 1
d = dataset['d'] # the distance between the center of the robot and the laser rangefinder

def predict_mu(mu,vk,omk):
    post_mult = np.array([[vk],[omk]])
    post_mult *= dt
    pre_mult = np.array([[np.cos(mu[2,0]),0],[np.sin(mu[2,0]),0],[0,1]])
    g = pre_mult.dot(post_mult)
    mu = np.add(mu,g)
    return mu

def corrected_th(th):
    th=th%(2*np.pi)
    if th>np.pi:
        th-=2*np.pi
    elif th<-np.pi:
         th+=2*np.pi
    return(th)

def get_F(v,th):
    return np.array([[1, 0, -v*dt*np.sin(th)],
                     [0, 1, v*dt*np.cos(th)],
                     [0, 0, 1]])

def get_G(landmark,mu,d):
    del_kx = (l[landmark, 0] - mu[0,0] - d*np.cos(mu[2,0]))
    del_ky = (l[landmark, 1] - mu[1,0] - d*np.sin(mu[2,0]))
    root_qk = np.sqrt(del_kx**2 + del_ky**2)
    theta_zk = math.atan2(del_ky,del_kx) - mu[2,0]
    theta_zk =corrected_th(theta_zk)
    z_k = np.array([[root_qk],
                    [theta_zk]])
    third_term = -((del_kx*np.cos(mu[2,0])+del_ky*np.sin(mu[2,0]))*d + del_kx**2 + del_ky**2)/(root_qk**2)
    
    G = np.array([[-del_kx/root_qk, -del_ky/root_qk, (del_kx*np.sin(mu[2,0])-del_ky*np.cos(mu[2,0]))*d/root_qk],
                    [del_ky/(root_qk**2), -del_kx/(root_qk**2), third_term]])
    return G,z_k

debug = False
number_of_states = len(x_true)
predicted_without_EKF = []
P = np.array([[1,0,0],
                [0,1,0],
                [0,0,0.1]])
R1 = np.array([[r_var[0,0],0],
              [0,b_var[0,0]]])
mu = np.array([[x_true[0,0]], [y_true[0,0]], [th_true[0,0]]])
for i in range(number_of_states):
    mu = predict_mu(mu,v[i][0],om[i][0])
    predicted_without_EKF.append([mu[0,0],mu[1,0],mu[2,0]])

#plot for predicted traj without EKF
x,y,mu_th = np.array(predicted_without_EKF).T
plt.plot(x,y)
plt.plot(x_true,y_true)
lx,ly = l.T
plt.scatter(lx,ly,color = 'r')
plt.title("plot for predicted traj without EKF")
plt.show()

debug = False
correction_applied = []
number_of_states = len(x_true)
predicted_using_EKF = []
sig_using_EKF = []
P = np.array([[1,0,0],
              [0,1,0],
              [0,0,0.1]])
R1 = np.array([[r_var[0,0],0],
              [0,b_var[0,0]]])
Q1 = np.array([[v_var[0,0],v_var[0,0],0],
             [v_var[0,0],v_var[0,0],0],
             [0,0,om_var[0,0]]])

mu = np.array([[x_true[0,0]], [y_true[0,0]], [th_true[0,0]]])

for i in range(number_of_states):
    mu = predict_mu(mu,v[i][0],om[i][0])
    F = get_F(v[i][0],mu[2,0])
    Q2 = np.array([[np.cos(mu[2,0])**2, np.sin(mu[2,0])*np.cos(mu[2,0]), np.cos(mu[2,0])],
                   [np.sin(mu[2,0])*np.cos(mu[2,0]), np.sin(mu[2,0])**2, np.sin(mu[2,0])],
                   [np.cos(mu[2,0]), np.sin(mu[2,0]), 1 ]])
    Q = np.multiply(Q1,Q2)
    P = F.dot(P).dot(F.T) + Q
    sig_using_EKF.append(P)
    for landmark in range(len(l)):
            if(r[i][landmark]!=0):
                    G,z_k = get_G(landmark,mu,d[0,0])
                    S = G.dot(P).dot(G.T) + R1
                    S = np.array(S,dtype='float')
                    Kalman_gain = P.dot(G.T).dot(np.linalg.inv(S))
                    P = (np.eye(3) - Kalman_gain.dot(G)).dot(P)
                    z = np.array([[r[i][landmark]],[b[i][landmark]]]) - z_k
                    mu = np.add(mu,Kalman_gain.dot(z))
                    correction_applied.append(Kalman_gain.dot(z)[0,0])
                    

    # end of for loop for Landmark
    predicted_using_EKF.append((mu[0,0],mu[1,0],mu[2,0]))

# plot for predicted using EKF
predictions = np.array(predicted_using_EKF)
x = predictions[:,0]
y = predictions[:,1]
plt.plot(x,y)
plt.plot(x_true,y_true)
plt.scatter(lx,ly,color = 'r')
plt.title("plot for predicted traj with EKF")
plt.show()

def plot_covariance_ellipse(xEst, PEst):
	Pxy = PEst[0:2, 0:2]
	eigval, eigvec = np.linalg.eig(Pxy)
	if eigval[0] >= eigval[1]:
		width = math.sqrt(eigval[0])
		height = math.sqrt(eigval[1])
		angle = math.atan2(eigvec[0, 1], eigvec[1, 0])
	else:
		width = math.sqrt(eigval[1])
		height = math.sqrt(eigval[0])
		angle = math.atan2(eigvec[1, 1], eigvec[1, 0])

	return Ellipse(xy=(xEst[0],xEst[1]),
         	       width=width,
         	       height=height,
         	       angle=angle,
         	       linewidth=0.2, fill=False)

fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
plt.title("Path Trajectory with Confidence Ellipse")
for i in range(number_of_states):
    ax.scatter(x[i],y[i],c='b')
    ax.add_artist(plot_covariance_ellipse(predictions[i], sig_using_EKF[i]))
    ax.scatter(x_true[i],y_true[i],c='r')
    plt.pause(0.0000005)
#    ax.cla()

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

plt.show()
