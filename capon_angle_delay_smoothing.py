
import numpy as np
from scipy.io import loadmat
from numpy import linalg
import matplotlib.pyplot as plt

from numpy.linalg import matrix_rank

def noise_synthetic(X, SNR):
    signal_power = 0
    for i in range(len(X)):
        signal_power += (np.transpose(X[i]).conj() @ X[i])
    signal_power /= X.size
    noise_power = 10 ** (-SNR / 10) * signal_power
    Noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, size=(len(X), len(X[0]))) + 1j*np.random.normal(0, 1, size=(len(X), len(X[0]))))
    X_noise = X + Noise
    return X_noise


def re_arange_data(data_original):
	# re arange measurement data from 2D matrix to 3D matrix
	# create 66-by-71 matrix where each entry has dimension 101
	freq_points = 101
	array_x = 71 # 71
	array_y = 66 # 66
	# Each block of 71 becomes a new row in the new matrix
	out_matrix = np.zeros((101,71,66), dtype=complex)  #(Z, X, Y) or (Frequency_point, Rows, Columns) or (101, 66, 71)

	# Run through all 1x4686 vectors (101 in total) and insert every 71 element into a new row of the new 3D matrix named "array_matrix"

	for l in range(freq_points):
		idx = 0
		for y in range(array_y): # Remember we have 66 rows in the real antenna array and 71 columns. y goes from 0 -> 66
			for x in range(array_x):
				out_matrix[l, x, y] = np.array(data_original[idx,l])
				idx = idx + 1 # max value = 66x71 = 4686
				#print(idx)

	return out_matrix


def get_sub_array_position(N_1, N_2, r_original):
# get new sub array from master position array
# Always starts with the sensor element located at  coordinate (1,1)
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



def spatial_smoothing(data_matrix_3D):
	
	freq_points = 101 # number of frequency sample points to use in smoothing 

	Lx = 71 # 71 # Lx: number of antenna elements in x dir. of original 2D array
	Ly = 66 # 66 # Ly: number of antenna elements in y dir. of original 2D array

	M = 4 # M: number of antenna elements in sub array along x dir. 
	N = 4 # N: number of antenna elements in sub array along y dir.

	# returns smoothed covariance matrix with size = (M x N x Lf) x (M x N x Lf) 

	nbr_sub_arrays_x  = Lx - M + 1  # for Lx = 71 and M = 10 -> the number of sub arrays in x dimension = 62
	nbr_sub_arrays_y  = Ly - N + 1 #  for Ly = 66 and N = 10 -> Ns, the number of sub arrays in y dimension = 57

	total_nbr_sub_arrays = nbr_sub_arrays_x * nbr_sub_arrays_y # also from lectures (p)

	R_smooth = np.zeros((M * N * freq_points, M * N * freq_points), dtype=complex)

	sub_a = 0

	for y in range(nbr_sub_arrays_y): # increment along y axis
		for x in range(nbr_sub_arrays_x): # increment along x axis

			X = [] # initialize empty vector for holding stacked rows of sub arrays
			for l in range(freq_points): 
				# get (N-by-M) sub-array for the l'th frequency point
				sub_array = data_matrix_3D[l, x:(x + M), y:(y + N)]
				# stack the rows of sub_array into 1D vector 
				X_sub_array = sub_array.flatten()
				# now for the rest of the l frequency points keep stacking the newly generated 1D vector 
				X.extend(X_sub_array)

			X = np.array([X]).T # convert back to numpy array and transpose

			Rxx = X @ X.conj().T  # construct covariance matrix
			R_smooth = np.add(R_smooth, Rxx)  
			sub_a = sub_a + 1
			print("nbr of sub array cov matrices calculated:", sub_a, "of:", total_nbr_sub_arrays)

	return R_smooth/(total_nbr_sub_arrays)		
		

#------ END functions --------------------

# Import data
file = "measurements.mat"
file2 = "R_f_column.mat"
data = loadmat(file)
data2 = loadmat(file2)
X = data["X_synthetic"] # frequency domain measurements
x = data["x_synthetic"] # time domain measurements
r = data["r"] # Get the array sensor possition vector 

R_smooth = data2["R_f_column"]
R_smooth = R_smooth
print(np.shape(R_smooth))
print('Rank of smoothed R:', matrix_rank(R_smooth))
R_inv = np.linalg.inv(R_smooth) 


# -------------------------------------------------------
# Set parameters for sub  data array dimensions
N1 = 71  # first dimension
N2 = 66  # second dimension
N3 = 101 # amount of samples used (max 101)

# -------------------------------------------------------
# Extract data
r_array = get_sub_array_position(4, 4, r)

f_carrier = 7.5e9 # 7.5 GHz
c = 300e6 
lamb = c/f_carrier # should be the same unit as r

nbr_steps_tau = 100
theta_search = np.arange(start=0, stop=2*np.pi, step=0.1)
tau_search = np.linspace(start=1.6667e-7, stop=(1.6667e-7 + 35e-9), num=nbr_steps_tau) 

tau_seconds = (np.arange(nbr_steps_tau) / nbr_steps_tau) * 35e-9 # axis for plot

Q = len(tau_search)
M = len(theta_search)

Lf_array = np.arange(N3)
delta_f = 2e6 # frequency spacing in measurements = 2MHz


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
		u = np.kron(b, a)
		#print('shape u:', np.shape(u))
		#print(np.shape(data))
		
		P_capon[m,q] = 1 / (u.conj().T @ R_inv @ u)

Power = 20 * np.log10(abs(P_capon))
Power = np.rot90(Power) # needed for heat map

"""
x = theta_search * (180/np.pi)
y = tau_seconds  

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(x, y)  

ax.plot_surface(X.T, Y.T, Power, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none') 

ax.set_xlabel(r'$\theta_{m}$ (degrees)')
ax.set_ylabel(r'$\tau_{n}$ (seconds)' )
ax.set_zlabel('Power (dB)')

plt.show()
"""
plt.imshow(Power, extent=[0, 360, 0, 350e-10], cmap='viridis', interpolation='none', aspect=1e10)
plt.scatter(50.9, 1.0e-8, s=60, c='red', marker='x', linewidths=1) 
plt.scatter(34.8, 1.46e-8, s=60, c='red', marker='x', linewidths=1) 
plt.scatter(63.8, 1.9027e-8, s=60, c='red', marker='x', linewidths=1) 
plt.scatter(164.8, 3.1118e-8, s=60, c='red', marker='x', linewidths=1) 
plt.scatter(273.7, 3.1175e-8, s=60, c='red', marker='x', linewidths=1)
plt.xlabel(r'$\theta_{m}$ (degrees)') 
plt.ylabel(r'$\tau_{n}$ (seconds)') 

plt.clim(np.amin(Power), np.amax(Power)) 
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label('Power (dB)')
plt.show()


