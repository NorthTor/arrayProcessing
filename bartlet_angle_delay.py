import numpy as np
from scipy.io import loadmat
from numpy import linalg
import matplotlib.pyplot as plt

from numpy import linalg as LA


def noise_synthetic(X, SNR):
    signal_power = 0
    for i in range(len(X)):
        signal_power += (np.transpose(X[i]).conj() @ X[i])
    signal_power /= X.size
    noise_power = 10 ** (-SNR / 10) * signal_power
    Noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, size=(len(X), len(X[0]))) + 1j*np.random.normal(0, 1, size=(len(X), len(X[0]))))
    X_noise = X + Noise
    return X_noise

def get_sub_array_position(N_1, N_2, r_original):
# get new sub array from master position array 
# L = N_1 x N_2 = the ammount of xy coordinates
# picn N_1 number of samples from N_2 number of sample blocks

	idx = 0
	r_sub_array = np.zeros((2,(N_1 * N_2)))
	for i in range(N_2):
		for j in range(N_1):
			r_sub_array[:,idx] = np.array(r_original[:, (j + 71 * i)])
			idx = idx + 1

	return r_sub_array


def get_sub_array_data(N_1, N_2, N_3, data_original):
	# N_1 = maximum 71 -> First dimension of new sub sensor array
	# N_2 = maximum 66 -> Second dimension of new sub sensor array
	# N_3 = amout of total array samples used -> maximum 101

	# Output dimensions = (N_1 x N_2) x N_3  -> in a 2D matrix 
    # Dimension of data_original = (71 x 66) x 101 -> in a 2D matrix (2D dimension = 4686 x 101)

    # Remember that the original measurement data comes in a matrix where each column is a 1 x 4686 vector.
    # The 1 x 4686 vector is arranged such that for every multiple of 71 entry, we have a new "sample block".
    # Each "sample block" corresponds to measurements of one row of the real (71 x 66) sensor array.

    # Example: We want to extract a (15 x 10) x 101 sub array 
    # I then want the 15 first samples from each sample block of size 71. 
    # The amount of blocks to extract 15 samples from is then 10.

	data_sub_array = np.zeros(((N_1 * N_2), N_3), dtype=complex)  # L = N_1 x N_2 and L_s = N_3
	
	# Pick N_1 number of samples from N_2 sample blocks 
	
	for n in range(N_3):
		idx = 0
		for i in range(N_2): 
			for j in range(N_1):
				data_sub_array[idx, n] = data_original[(j + 71*i), n]
				idx = idx + 1

	return data_sub_array
	
#------ END functions --------------------

# Import data
file = "measurements.mat"
data = loadmat(file)
X = data["X"] # frequency domain measurements
r = data["r"] # Get the array sensor possition vector 

# Set parameters for sub array dimensions
N1 = 7 # first dimension
N2 = 7  # second dimension
N3 = 101 # amount of samples used (max 101)


# Get the sub arrays, data and position
r_array = get_sub_array_position(N1, N2, r)
X_data = get_sub_array_data(N1, N2, N3, X)

SNR = -10
X = noise_synthetic(X_data, SNR)

print(np.shape(X))
# flatten data matrix column wise
X = np.array([X.flatten('F')])

# transpose to get a row vector
X = X.T
print('Shape X:', np.shape(X))

# Barlett implementation without delay
R_xx = (X @ X.conj().T)

print('Shape Rxx:', np.shape(R_xx))

nbr_steps_tau = 100
# Set up search angles and delays for barlett implementation
theta_search = np.arange(start=0, stop=2*np.pi, step=0.01)
tau_search = np.linspace(start=1.6667e-7, stop=(1.6667e-7 + 35e-9), num=nbr_steps_tau) 
tau_seconds = (np.arange(nbr_steps_tau) / nbr_steps_tau) * 35e-9 

Q = len(tau_search)
M = len(theta_search)

f_carrier = 7.5e9 # 7.5 GHz
c = 300e6 
lamb = c/f_carrier # should be the same unit as r
delta_f = 2e6   # frequency spacing in measurements = 2MHz
f_0 = 7.4e9 # lower frequency of signal 

array = np.arange(N3)


a = np.zeros((M,N1*N2), dtype=complex)

P_bartlett = np.zeros((M,Q), dtype=complex)

for m in range(M):
	print(m, 'of', M, 'angles')

	# compute a-vector
	#print('shape e:', np.shape(e))
	#print(np.shape(r_array))
	
	a = np.exp(1j * ((2 * np.pi) / lamb) * (np.array([np.cos(theta_search[m]), np.sin(theta_search[m])]) @ r_array))
	
	#print("Shape a:", np.shape(a))

	for q in range(Q):
		# compute the b matrix
		b = np.exp(-1j * 2 * np.pi * array * tau_search[q] * delta_f)
		#print("Shape b:", np.shape(b))
		u = np.kron(b, a).T
		#u = np.array([u]).T
		#print('shape u:', np.shape(u))

		numerator = u.conj().T @ R_xx @ u
		denominator = u.conj().T @ u

		P_bartlett[m,q] = numerator / denominator
		# print(np.shape(data))

Power = 20 * np.log10(abs(P_bartlett))
Power = np.rot90(Power) # needed for heat map
#limits = 20 * np.log10(np.amax(abs(P_bartlett))) + np.array([-240,0])


rows, cols = np.shape(Power)

Power_reduced = np.zeros((rows,cols))
# replace values lower than 40 db in the spectrum
for i in range(rows):
	for j in range(cols):
		if Power[i,j] < -200:
			Power_reduced[i,j] = -200
		else:
			Power_reduced[i,j] = Power[i,j]

"""
x = theta_search * (180/np.pi)
y = tau_seconds * c # to go to meters
fig = plt.figure()
fig.tight_layout()
ax = fig.add_subplot(projection='3d')
X,Y = np.meshgrid(x, y)  
ax.plot_surface(X.T, Y.T, Power, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none') 
ax.set_xlabel('Angles')
ax.set_ylabel('Delay (seconds)')
ax.set_zlabel('Power (db)')
#ax.set_zlim(limits[0], limits[1])
ax.set_title('Bartlett spectrum')
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""

plt.imshow(Power_reduced, extent=[0, 360, 0, 350e-10], cmap='viridis', interpolation='none', aspect=1e10)
plt.scatter(50.9, 1.0e-8, s=60, c='red', marker='x', linewidths=1) 
plt.scatter(34.8, 1.46e-8, s=60, c='red', marker='x', linewidths=1) 
plt.scatter(63.8, 1.9027e-8, s=60, c='red', marker='x', linewidths=1) 
plt.scatter(164.8, 3.1118e-8, s=60, c='red', marker='x', linewidths=1) 
plt.scatter(273.7, 3.1175e-8, s=60, c='red', marker='x', linewidths=1)
plt.xlabel(r'$\theta_{m}$ (degrees)') 
plt.ylabel(r'$\tau_{q}$ (seconds)') 

plt.clim(np.amin(Power_reduced), np.amax(Power_reduced)) 
cbar = plt.colorbar(fraction=0.046, pad=0.02)
cbar.set_label('Power (dB)')
plt.show()


