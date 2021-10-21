

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# --------- Start functions ----------------------------------------------------------------
def spectral_MUSIC(X, M, theta_search):
	# theta_search: vector with search angle values 
	L, N = np.shape(X) 

	# compute spatial covariance matrix from observations X
	Rxx = (X @ X.conj().T) / N

	# eigenvalue decomposition
	eigVal, eigVec = linalg.eig(Rxx) # eigenvectors in columns sorted in decending order 
	print(eigVec) 

	# sort the list of eigenvalues and eigenvectors in ascending order (lowest first)
	idx = np.argsort(eigVal)
	eigVal = eigVal[idx] 
	eigVec = eigVec[:,idx] 

	# Select U_noise matrix with the L-M noise eigenvectors associated with the least eigenvalues 
	U_n = eigVec[:, 0:(L-M)] 

	# determine the frequency estimates as the M frequencies
	# corresponding to the M highest maxima of the Music spectrum
	P_music = np.zeros(len(theta_search))

	for k in range(len(theta_search)):
		a = np.exp(1j * np.pi * (np.arange(L) * np.cos(theta_search[k])))
	
		denom = a.conj().T @ U_n @ U_n.conj().T @ a
		P_music[k] = 1/np.abs(denom)

	return P_music



def ESPRIT(X, M):
	# theta_search: vector with search angle values 
	L, N = np.shape(X) 

	# compute spatial covariance matrix from observations X
	Rxx = (X @ X.conj().T) / N

	# eigenvalue decomposition
	eigVal, eigVec = linalg.eig(Rxx) # eigenvectors in columns sorted in decending order 
	print(eigVec) 

	# sort the list of eigenvalues and eigenvectors in ascending order (lowest first)
	idx = np.argsort(eigVal)
	eigVal = eigVal[idx] 
	eigVec = eigVec[:,idx] 

	# Select U_noise matrix with the L-M noise eigenvectors associated with the least eigenvalues 
	U_n = eigVec[:, 0:(L-M)] 

	# determine the frequency estimates as the M frequencies
	# corresponding to the M highest maxima of the Music spectrum
	P_music = np.zeros(len(theta_search))

	for k in range(len(theta_search)):
		a = np.exp(1j * np.pi * (np.arange(L) * np.cos(theta_search[k])))
	
		denom = a.conj().T @ U_n @ U_n.conj().T @ a
		[k] = 1/np.abs(denom)

	return 



def ULA_data_generator(theta_vec, L, N, SNR):

	M = np.shape(theta_vec)[0]
	s_amp = np.sqrt(10)
	s_phase = np.exp(1j * np.random.rand(M, N) * 2 * np.pi)
	s = s_amp * s_phase

	noise_variance = 1/(SNR/(s_amp**2))
	# for A: 1st column, a of theta[0], second column a of theta[1] etc.
	A = np.zeros((L,M), dtype=complex)

	for m in range(len(theta_vec)):
			A[:,m] = np.exp(1j * np.pi * np.arange(L) * np.cos(theta_vec[m])) # A is L-by-M response matrix

	noise = np.sqrt(noise_variance) / np.sqrt(2) * np.random.rand(L, N) + 1j * np.random.rand(L,N)

	X = (A @ s) + noise  

	return X




def generateData(rho, M, N, SNR):
	# rho: covariance 
	# M:   number of waves
	# N:   number of samples in each signal

	# generate signals with random phase
	s_amp = np.sqrt(10) # fix the amplitude of signals
	s_phase = 	np.exp(1j*np.random.rand(M,N)*2*np.pi) # set signals random phase ( 0 ->2pi)
	complex_signal_matrix = s_amp * s_phase 


	# generate noise signals
	noise_variance = (s_amp**2)/SNR  # see short description on how this came to be
	complex_noise_matrix = np.sqrt(noise_variance) * np.exp(np.random.rand(M,N)*2*np.pi)

	# Generate an arbitrary covariance matrix A for the signals
	C = np.full((N,N), rho) # fill array with rho values
	np.fill_diagonal(C,1)   # add 1 along the diagonal
	
	B = np.linalg.cholesky(A) # cholesky decomposition of A

	# create partially correlated signals
	X = B @ (complex_signal_matrix.T + complex_noise_matrix)
	
	return X


# ---- End functions -------------------------------------------------

# ----- Start main ---------------------------------------------------

rho = 00.1    # for generating correlated signals

M = 4 		  # number of impeeding waves  
N = 200       # number of samples in each signal
SNR = 1000    # power

L = 8 # number of antennas

theta_vec_angles = np.array([25, 45, 65, 125])
theta_vec_rad = theta_vec_angles * (np.pi/180)

signals = ULA_data_generator(theta_vec_rad, L, N, SNR)

print(np.shape(signals))

theta_search_rad = np.arange(start=0, stop=np.pi, step=0.001) # 1x3142 vector with radian angles
theta_search_ang = theta_search_rad * (180/np.pi)

P = spectral_MUSIC(signals, M, theta_search_rad)


plt.plot(theta_search_ang, P, linewidth=1)
plt.ylabel('')
plt.xlabel('')
plt.grid()
plt.show()









