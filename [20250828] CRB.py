# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:08:54 2025

@author: junai
"""
import numpy as np
from DWC_F import eig_e_corr
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt

#N = [8,9,10,11, 12, 13] # number of channel uses = 3*2**N
N = np.linspace(1, 8, 20)
M = 1000 # number of estimates for averaging
gamm = 0.8 # fixed channel

# V2 of data_files. These are fractions of channel capacity
x_targ = 0.6
y_targ = 0.45
z_targ = 0.4

# V1 of data files were run on these parameters. These parameters are absolute rates
# x_targ = 0.3
# y_targ = 0.2
# z_targ = 0.1



p_X_hat = np.zeros([len(N), M])
p_Y_hat = np.zeros([len(N), M])
p_Z_hat = np.zeros([len(N), M])
e_X_hat = np.zeros([len(N), M])
e_Y_hat = np.zeros([len(N), M])
e_Z_hat = np.zeros([len(N), M])

pauli_P = eig_e_corr(2, gamm)

X_channel = [pauli_P[0] + pauli_P[1], pauli_P[2] + pauli_P[3]]
Y_channel = [pauli_P[0] + pauli_P[2], pauli_P[1] + pauli_P[3]]
Z_channel = [pauli_P[0] + pauli_P[3], pauli_P[2] + pauli_P[1]]

A = np.linalg.inv(np.matrix([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1]]))

for idN, n in enumerate(N):
    for jj in range(M):
        N_samples = int((10**n)/3)
        err_X = np.random.multinomial(N_samples, X_channel)
        e_X = err_X[0]/N_samples
        err_Y = np.random.multinomial(N_samples, Y_channel)
        e_Y = err_Y[0]/N_samples
        err_Z = np.random.multinomial(N_samples, Z_channel)
        e_Z = err_Z[0]/N_samples
        
        e_X_hat[idN, jj] = e_X
        e_Y_hat[idN, jj] = e_Y
        e_Z_hat[idN, jj] = e_Z
        
        P = A*np.array([[e_X], [e_Y], [e_Z], [1]])
        
        p_X_hat[idN, jj] = np.array(P[0])[0][0]
        p_Y_hat[idN, jj] = np.array(P[1])[0][0]
        p_Z_hat[idN, jj] = np.array(P[2])[0][0]

var_X = np.var(p_X_hat, axis = 1)
var_Y = np.var(p_Y_hat, axis = 1)
var_Z = np.var(p_Z_hat, axis = 1)

var_e_X = np.var(e_X_hat, axis = 1)
var_e_Y = np.var(e_Y_hat, axis = 1)
var_e_Z = np.var(e_Z_hat, axis = 1)

#%% Analytical total error

total_error = []
total_error_e = []
for n in N:
    p_I = pauli_P[0]
    p_X = pauli_P[1]
    p_Y = pauli_P[2]
    p_Z = pauli_P[3]
    total_error.append((9 / (2 * (10**n))) * ( 1 - p_I - p_X**2 - p_Y**2 - p_Z**2 - p_X*p_Y - p_X*p_Z - p_Y*p_Z) )
    total_error_e.append( (3/(10**n)) *(X_channel[0]*(1 - X_channel[0]) + \
                                        Y_channel[0]*(1 - Y_channel[0]) + Z_channel[0]*(1 - Z_channel[0])) )




#%%
#plt.semilogy(N, var_X, 'b')
#plt.semilogy(N, var_Y, 'r')
#plt.semilogy(N, var_Z, 'g')
plt.semilogy(N, var_X + var_Y + var_Z, 'b*', label = "p Simul")
plt.semilogy(N, total_error, 'b', label = "p Theor")

plt.semilogy(N, var_e_X + var_e_Y + var_e_Z, 'r*', label = "e Simul")
plt.semilogy(N, total_error_e, 'r', label = "e Theor")
plt.legend()
plt.show()

#%%
for n, v, e in zip(N, np.log10(var_X + var_Y + var_Z), np.log10(total_error)):
    print(f"{n}\t{v}\t{e}")
