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
SNR = 0
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
range_theta = 100
step_theta = np.pi / range_theta
range_tau = 10
step_tau = 4e-8 / range_tau

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


def get_sub_array_position(N_1, N_2, r_original):
    # get new sub array from master position array
    # L = N_1 x N_2 = the ammount of xy coordinates
    # picn N_1 number of samples from N_2 number of sample blocks

    idx = 0
    r_sub_array = np.zeros((2, (N_1 * N_2)))
    for i in range(N_2):
        for j in range(N_1):
            r_sub_array[:, idx] = np.array(r_original[:, (j + 71 * i)])
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
                data_sub_array[idx, n] = data_original[(j + 71 * i), n]
                idx = idx + 1

    return data_sub_array


def noise_synthetic(X, SNR):
    signal_power = 0
    for i in range(len(X)):
        signal_power += (np.transpose(X[i]).conj() @ X[i])
    signal_power /= X.size
    noise_power = 10 ** (-SNR / 10) * signal_power
    Noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, size=(len(X), len(X[0]))) + 1j*np.random.normal(0, 1, size=(len(X), len(X[0]))))
    X_noise = X + Noise
    return X_noise

def Bartlett_2D(X, theta_search, tau_search, step, range_step):
    R_hat = X @ X.conj().T
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


def spatial_smoothing(X, SL1, SL2, SL3, L1, L2):
    P1 = L1 - SL1 + 1
    P2 = L2 - SL2 + 1
    SM = np.zeros((SL1*SL2, 101))
    R_p = np.zeros((SL1*SL2, SL1*SL2, P1*P2))
    R_f = np.zeros((SL1, SL2))
    for j in range(P2):
        for i in range(P1):
            for m in range(SL2):
                SM[m*SL1:SL1*(m+1), :] = X[m*L1+j*L2+i : m*L1+SL1+i+j*L2, :]

                R_p[:,:,j*P2+i] = SM @ SM.conj().T
    R_f = np.sum(R_p, 2) / P1*P2
    print(R_f.shape[0])
    Js = np.eye(R_f.shape[0])[::-1]

    R_fb = 0.5 * (R_f + Js * R_f.conj() * Js)

    return R_fb



def Bartlett_delay(X_bartlett, r_bartlett, theta_search, tau_search, Lf):
    X_LLf = []
    Lf_array = np.arange(Lf)
    for i in range(len(X_bartlett[0])):
        for j in range(len(X_bartlett)):
            X_LLf.append(X_bartlett[j][i])
    X_LLf = np.array(X_LLf)[np.newaxis]
    X_LLf = X_LLf.T

    # X = np.array([X_bartlett.flatten('F')])
    # X = X.T

    R_hat = X_LLf @ X_LLf.conj().T / N
    verif = 0
    P_bartlett = np.zeros((len(theta_search), len(tau_search)), dtype=complex)
    for i in range(len(theta_search)):
        print(verif)
        a = np.exp(1j * (2 * np.pi / wave_length) * (np.array([np.cos(theta_search[i]), np.sin(theta_search[i])]) @ r_bartlett))
        verif2 = 0

        for j in range(len(tau_search)):
            # print("verif2 vaut :" + str(verif2))
            b = np.exp(-1j * 2 * np.pi * Lf_array * tau_search[j] * delta_f)
            u_bartlett = np.kron(a, b)
            num = u_bartlett.conj().T @ R_hat @ u_bartlett
            denom = u_bartlett.conj().T @ u_bartlett
            coef = num / denom
            # print(coef)
            P_bartlett[i, j] = coef
            verif2 += 1
        verif += 1
    return P_bartlett



X_noise = noise_synthetic(X_synthetic, SNR)
Xn_sub = get_sub_array_data(10, 10, 101, X_noise)
r_sub = get_sub_array_position(10, 10, r)
theta_search = [step_theta * i for i in range(-range_theta, range_theta)]
# theta_search = np.arange(start=-np.pi, stop=np.pi, step=0.01)
# tau_search = np.arange(start=1e-8, stop=4e-8, step=0.1e-8)  # need better search field # OBS
theta_degree = [(theta * 180) / np.pi for theta in theta_search]
tau_search = [step_tau * i for i in range(-range_tau, range_tau)]

print(Xn_sub.shape)
R = spatial_smoothing(Xn_sub, SL1=6, SL2=6, SL3=6, L1=10, L2=10)
print(R.shape)
print(R)

# print(X_synthetic)
# R_fb = spatial_smoothing(X_noise, 8, 8)
# print(R_fb)
# idx_array = getSubarray(71, 66, 4, 4)
# print(idx_array)
# P = Bartlett_delay(Xn_sub, r_sub, theta_search, tau_search, N)
# print(P)
# print(P[0])
# print(P.shape)
# plt.plot(x_axis, P)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# Power = abs(P)

# x = theta_degree
# y = tau_search

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# X, Y = np.meshgrid(x, y)

# ax.plot_surface(X.T, Y.T, Power, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')

# ax.set_xlabel('Angles')
# ax.set_ylabel('Delay tau (phase)')
# ax.set_zlabel('Magnitude Power')
#
# plt.show()