


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
X = data["X_synthetic"] # time measurements

r_array = data["r"]

# Shape of synthetic data x = 4686 x 101 -> translates to 101 samples of all array elements 
# Decyphering the measurement data: 
#			Each 1x4686 vector corresponds to one "snapshot" or one sample session of all array elements.
#			As the 2D array element dimension is given as 71x66, colums in the array is taken as 71 and rows as 66.  
#			The measurement array is decomposed into 66 vectors of size 1x71

# Decomposing and sorting measurements into 71 ULAs where the measurments from 1 ULA is represented in a matrix.

print(np.shape(r_array))
R_xx = X @ X.conj().T

f_carrier = 7.5e9 # 7.5 GHz
c = 300e6
lamb = c/f_carrier

tau_search = np.arange(start=0, stop=2*np.pi, step=0.1) # Phase delay 
theta_search = np.arange(start=-np.pi, stop=np.pi, step=0.01) # Angle of Arival

M = len(theta_search)
Q = len(tau_search)


P_barlett = np.zeros(M, dtype=complex)
for m in range(M):
	print(m, 'of', M, 'angles')
	a = np.exp(1j * (2 * np.pi / lamb) * (np.array([np.cos(theta_search[m]), np.sin(theta_search[m])]) @ r_array))
	#print(np.shape(a))
	numerator = a.conj().T @ R_xx @ a
	denominator = a.conj().T @ a

	P_barlett[m] = numerator / denominator

P_barlett = abs(P_barlett)


plt.plot(theta_search*(180/np.pi), P_barlett, linewidth=1)
plt.ylabel('')
plt.xlabel('')
plt.grid()
plt.show()



