
import numpy as np
from scipy.io import loadmat
from numpy import linalg
import matplotlib.pyplot as plt


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
X = data["X_synthetic"] # frequency domain measurements
x = data["x_synthetic"] # time domain measurements
r = data["r"] # Get the array sensor possition vector 

# Set parameters for sub array dimensions
N1 = 9  # first dimension
N2 = 9  # second dimension
N3 = 101 # amount of samples used (max 101)
SNR = 10

X = noise_synthetic(X, SNR) # add complex noise

# Get the sub arrays, data and position
r_array = get_sub_array_position(N1, N2, r)
X = get_sub_array_data(N1, N2, N3, X)

X = np.array([X.flatten('F')])
X = X.T

print('Shape X:', np.shape(X))
# Barlett implementation without delay
R_xx = (X @ X.conj().T) / N3 

R_xx_inv = np.linalg.inv(R_xx)

print('Shape Rxx:', np.shape(R_xx_inv))


f_carrier = 7.5e9 # 7.5 GHz
c = 300e6 
lamb = c/f_carrier # should be the same unit as r

# Set up search angles for barlett implementation
theta_search = np.arange(start=-np.pi, stop=np.pi, step=0.1)
tau_search = np.arange(start=1e-8, stop=4e-8, step=0.1e-8) # need better search field # OBS

Q = len(tau_search)
M = len(theta_search)

Lf = N3 
Lf_array = np.arange(Lf)

delta_f = 2e6 # frequency spacing in measurements = 2MHz


a = np.zeros((M,N1*N2), dtype=complex)

P_capon= np.zeros((M,Q), dtype=complex)

for m in range(M):
	print(m, 'of', M, 'angles')

	# compute a vector
	a = np.exp(1j * (2 * np.pi / lamb) * (np.array([np.cos(theta_search[m]), np.sin(theta_search[m])]) @ r_array))
	#print("Shape a:", np.shape(a))

	for q in range(Q):
		# compute the b matrix
		b = np.exp(-1j * 2 * np.pi * Lf_array * tau_search[q] * delta_f)
		# print("Shape b:", np.shape(b))
		# print("Shape u:", np.shape(u))
		u = np.kron(a, b)

		#print('shape u:', np.shape(u))
		#print(np.shape(data))
		
		P_capon[m,q] = (u.conj().T @ R_xx_inv @ u)


Power = abs(P_capon)

x = theta_search * (180/np.pi)
y = tau_search

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(x, y)  

ax.plot_surface(X.T, Y.T, Power, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none') 

ax.set_xlabel('Angles')
ax.set_ylabel('Delay tau (phase)')
ax.set_zlabel('Magnitude Power')

plt.show()



