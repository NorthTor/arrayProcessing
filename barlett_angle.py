
import numpy as np
from scipy.io import loadmat
from numpy import linalg
import matplotlib.pyplot as plt


# get new sub array from master position array 
# L = N_1 x N_2

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

N1 = 30
N2 = 30
N3 = 70
Lf = N3

r_array = get_sub_array_position(N1, N2, r)
X = get_sub_array_data(N1, N2, N3, X)

# Barlett implementation without delay

R_xx = X @ X.conj().T

f_carrier = 7.5e9 # 7.5 GHz
c = 300e6 
lamb = c/f_carrier

# set up search angles and delays for barlett implementation
tau_search = np.arange(start=0, stop=2*np.pi, step=0.01) # Phase delay 
theta_search = np.arange(start=-np.pi, stop=np.pi, step=0.01) # Angle of Arival

M = len(theta_search)
Q = len(tau_search)

P_barlett = np.zeros((M,Q), dtype=complex)

lf_vector = np.arange(Lf)
delta_f = 2e6   # Frequency spacing of samples 2 MHz

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




