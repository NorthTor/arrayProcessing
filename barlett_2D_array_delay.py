


import numpy as np
from scipy.io import loadmat
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D




def eta_x(angle,lamb):
	x = (0.01/lamb) * np.cos(angle) # add the dx/lambda here
	return x

def eta_y(angle,lamb):
	y = (0.01/lamb) * np.sin(angle)  # add the dx/lambda here
	return y

# Import data

file = "measurements.mat"
data = loadmat(file)
X = data["X_synthetic"] # frequency measurements

r_array = data["r"]

# Shape of synthetic data x = 4686 x 101 -> translates to 101 samples of all array elements 
# Decyphering the measurement data: 
#			Each 1x4686 vector corresponds to one "snapshot" or one sample session of all array elements.
#			As the 2D array element dimension is given as 71x66, colums in the array is taken as 71 and rows as 66.  
#			The measurement array is decomposed into 66 vectors of size 1x71

# Decomposing and sorting measurements into 71 ULAs where the measurments from 1 ULA is represented in a matrix.

# only use sub array in order to lower computation time say
R_xx = X @ X.conj().T

f_carrier = 7.5e9 # 7.5 GHz
c = 300e6
lamb = c/f_carrier
K = 4686

tau_start = -2e-9
tau_stop = 4e-7
tau_nbr_steps = 4686
tau_dist = tau_stop - tau_start
tau_step_size = tau_dist/tau_nbr_steps

tau_search = np.arange(start=tau_start, stop=tau_stop, step=tau_step_size ) # Phase delay 

theta_start = -np.pi
theta_stop = np.pi 
nbr_theta = 4686
theta_dist = theta_stop - theta_start
theta_step_size = theta_dist/nbr_theta

theta_search = np.arange(start=theta_start, stop=theta_stop, step=theta_step_size) # Angle of Arival

M = len(theta_search)
Q = len(tau_search)
Nf = 101
f_tau = np.arange(start=0, stop=Nf, step=1)


Power = np.zeros((M,Q), dtype=complex)
A = np.zeros((M,4686), dtype=complex)

# Construct the angle matrix (size = M-by-4686)
for m in range(M):
	A[m,:] = np.exp(1j * 2 * (np.pi / lamb) * (np.array([np.cos(theta_search[m]), np.sin(theta_search[m])]) @ r_array))
	# A is a matrix where each row corresponds to a steering vector with specific angle
	# The amount of rows correspond to the amount of search angle
	# the amount of columns = the amount of sensors ie. antennas (4686)

# Construct the phase delay matrix (Need to be same dimension as A)
# should have different values in each row - ie. each row should contain all delays
B = np.zeros(4686, dtype=complex)
B = np.exp(-1j * 2 * np.pi * f_carrier * tau_search)

e = np.zeros((M,4686), dtype=complex)
for i in range(M):
	e[i,:] = A[i,:] * B

# print(np.shape(A))

# Multiply A and B entry wise 
print(np.shape(e))

numerator = e.conj().T @ R_xx @ e
denominator = e.conj().T @ e

P_barlett = numerator / denominator

Power = abs(P_barlett)
x = theta_search * (180/np.pi)
y = tau_search * (180/np.pi)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(x, y)  

ax.plot_surface(X.T, Y.T, Power, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none') 

ax.set_xlabel('Angles')
ax.set_ylabel('Delay tau (phase)')
ax.set_zlabel('Magnitude Power')

plt.show()


