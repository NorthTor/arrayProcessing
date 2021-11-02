##### Implementation of the 2D Bartlett algorithm.
##################################################

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##### Definition of variables
L = 4686
Lx = 71
Ly = 66
SNR = -25
N = 101
f0 = 7.5e9
delta_f = 200e6
Tau = 5e-9          #Sampling time = 5ns
wave_length = 3e8 / f0

data = scipy.io.loadmat("../MeasurementforMiniproject.mat")
x_synthetic = data["x_synthetic"]
X_synthetic = data["X_synthetic"]
x = data["x"]
X = data["X"]
r = data["r"]
range_step = 100
step = np.pi / range_step

def getSubarray(N_row, N_column, L1, L2, spacing=1):
    idx_column = []
    idx_row = []
    for i in range(1, N_row, spacing):
        idx_row.append(i)
    for i in range(1, N_column, spacing):
        idx_column.append(i)
    if (len(idx_column) < L1) or (len(idx_row) < L2):
        print("Problem in finding the subarray")
    else:
        idx_column = idx_column[:L1]
        idx_row = idx_row[:L2]
    idx_array = []
    for il2 in range(L2):
        for i in range(len(idx_column)):
            idx_array.append(idx_column[i] + N_row*il2*spacing)
    return idx_array

def noise_synthetic(X, SNR):
    signal_power = 0
    for i in range(len(X)):
        signal_power += (np.transpose(X[i]).conj() @ X[i])
    signal_power /= X.size
    noise_power = 10 ** (-SNR / 10) * signal_power
    Noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, size=(len(X), len(X[0]))) + 1j*np.random.normal(0, 1, size=(len(X), len(X[0]))))
    X_noise = X + Noise
    return X_noise

def Bartlett_2D(X, step, range_step):
    R_hat = X @ X.conj().T
    theta_search = [step * i for i in range(-range_step, range_step)]
    theta_degree = [(theta * 180) / np.pi for theta in theta_search]
    P_bartlett = []
    verif = 0
    for i in range(len(theta_search)):
        print(verif)
        e = np.array([np.cos(theta_search[i]), np.sin(theta_search[i])])
        e = np.transpose(e)
        a = []
        for j in range(len(r[0])):
            x_r = r[0][j]
            y_r = r[1][j]
            r_position = np.array([x_r, y_r])
            scalar_product = e @ r_position
            a_coef = np.exp(1j * 2*np.pi / wave_length * scalar_product)
            a.append(a_coef)
        a = np.array(a)
        a_T = np.transpose(a)
        numerator = (a_T.conj() @ R_hat @ a )
        denominator = a_T.conj() @ a
        P_bartlett.append(abs(numerator/denominator))
        verif += 1
    return P_bartlett, theta_degree


def spatial_smoothing(X, subarray_x, subarray_y):
    y_size = len(X)
    x_size = len(X[0])
    R_f = np.zeros((subarray_x, subarray_y), dtype=complex)
    for y in range(y_size - subarray_y + 1):
        for x in range(x_size - subarray_x + 1):

            # Creation of the 2D matrix Xp for a subarray of size (subarray_x, subarray_y)
            Xp = []
            for y2 in range(subarray_y):
                new_row = []
                for x2 in range(subarray_x):
                    # print(x+x2)
                    new_row.append(X[y+y2][x+x2])
                Xp.append(new_row)
            Xp = np.array(Xp)


            R_hat = Xp @ np.transpose(Xp).conj()
            R_f += R_hat
        print(y)

    #Division of R_f by "P" which is here in this case the product of the number of subarrays on each axis.
    R_f /= ( (y_size - subarray_y + 1) * (x_size - subarray_x + 1) )

    # subarray_y can also be used instead of subarray_x as Js is a square Matrix.
    Js = np.eye(subarray_x)[::-1]

    R_fb = 1 / 2 * (R_f + Js @ R_f.conj() @ Js)
    return R_fb


def Bartlett_delay(X, step, range_step):
    R_hat = X @ X.conj().T
    theta_search = [step * i for i in range(-range_step, range_step)]
    theta_degree = [(theta * 180) / np.pi for theta in theta_search]
    Azimuth = []
    Delay = []
    verif = 0
    b = []
    U =[]
    for i in range(N):
        b.append(np.exp(-1j*2*np.pi*i*delta_f*Tau))
    for i in range(len(theta_search)):
        print(verif)
        e = np.array([np.cos(theta_search[i]), np.sin(theta_search[i])])
        e = np.transpose(e)
        a = []
        print(len(r[0]))
        for j in range(len(r[0])):
            x_r = r[0][j]
            y_r = r[1][j]
            r_position = np.array([x_r, y_r])
            scalar_product = e @ r_position
            a_coef = np.exp(1j * 2*np.pi / wave_length * scalar_product)
            a.append(a_coef)
        a = np.array(a)
        a_T = np.transpose(a)
        numerator = (a_T.conj() @ R_hat @ a )
        denominator = a_T.conj() @ a
        Azimuth.append(abs(numerator/denominator))
        U_column = np.kron(a,b)
        U.append(U_column)
        # print(u.shape)
        # print(R_hat.shape)
        columnTranspose = np.transpose(U_column)
        delay_num = columnTranspose.conj() @ R_hat @ U_column
        delay_denom = columnTranspose.conj() @ U_column
        # print(delay_num.shape)
        # print(delay_denom.shape)
        Delay.append(abs(delay_num/delay_denom))
        verif += 1
    return Azimuth, Delay, theta_degree




X_noise = noise_synthetic(X_synthetic, SNR)
# print(X_synthetic)
# R_fb = spatial_smoothing(X_noise, 8, 8)
# print(R_fb)
idx_array = getSubarray(71, 66, 4, 4)
print(idx_array)
# P, delay, x_axis = Bartlett_delay(X_noise, step, range_step)
# plt.plot(x_axis, delay)
# plt.show()
