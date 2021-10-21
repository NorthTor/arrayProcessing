

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def ULA_data_generator(theta_vec, L, N, SNR):

	M = np.shape(theta_vec)[0]
	s_amp = np.sqrt(10)
	s_phase = np.exp(1j * np.random.rand(M, N) * 2 * np.pi)
	s = s_amp * s_phase
	print(type(s[0]))

	noise_variance = 1/(SNR/(s_amp**2))
	# for A: 1st column, a of theta[0], second column a of theta[1] etc.
	A = np.zeros((L,M), dtype=complex)

	for m in range(len(theta_vec)):
			A[:,m] = np.exp(1j * np.pi * np.arange(L) * np.cos(theta_vec[m])) # A is L-by-M response matrix

	noise = np.sqrt(noise_variance) / np.sqrt(2) * np.random.rand(L, N) + 1j * np.random.rand(L,N)

	X = (A @ s) + noise  

	return X


def barlett(X, L, theta_search):
	# X = input observations 
	# L = number of antenna elements
	# d = wave-length = 1/2

	# The Bartlett beamforming algorithm maximizes the signal output 
	# by giving a large weight to the input signal from a specific direction.
	M = len(theta_search) # number of test angles
	Rxx = X @ X.conj().T

	# construct respone vector A (L-by-M) matrix (steering matrix) for standard ULA
	# each column present sepparate angle. 
	# we define the spatial frequency as 1/2 cos(theta)
	terms = np.arange(L)
	A = np.empty(((L),M), dtype=complex)

	for m in range(M):
		A[:,m] = np.exp(1j * 2 * np.pi * terms * 0.5 * np.cos(theta_search[m]))


	P_barlett = np.empty(M, dtype=complex)
	for m in range(M):
		# estimate output power
		a = A[:,m]	
		num = a.conj().T @ Rxx @ a
		denom = a.conj().T @ a

		P_barlett[m] = num / denom

	return P_barlett





#--------- END Functions -------------------------------------------------------------------------------

N = 300       # number of samples in each signal
SNR = 1000    # SNR power
L = 8 # number of array elements (antennas)

theta_vec = np.array([90, 120, 150])
theta_vec = theta_vec * (np.pi/180) # radians

theta_search_rad = np.arange(start=0, stop=np.pi, step=0.001) # 1x3142 vector with radian angles
theta_search_ang = theta_search_rad * (180/np.pi)

X = ULA_data_generator(theta_vec, L, N, SNR)

P_barlett = barlett(X, L, theta_search_rad)

plt.plot(theta_search_ang, P_barlett, linewidth=1)
plt.ylabel('')
plt.xlabel('')
plt.grid()
plt.show()


