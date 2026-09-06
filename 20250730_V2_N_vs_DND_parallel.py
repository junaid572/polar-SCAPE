# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:10:22 2025

@author: junai
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 13:06:03 2025

@author: junaid

channel uses vs. diamond norm distance
"""
import numpy as np
from DWC_F import eig_e_corr
import matplotlib.pyplot as plt
from polar_codes.polar_code import PolarCode
from polar_codes.channels.bsc_channel import BscChannel
from DWC_F import bsc_cap

from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from tqdm import tqdm

N = [8,9,10,11, 12, 13] # number of channel uses = 3*2**N
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



DN_distance = np.zeros([len(N), M])
DN_distance_polar = np.zeros([len(N), M])

pauli_P = eig_e_corr(2, gamm)

X_channel = [pauli_P[0] + pauli_P[1], pauli_P[2] + pauli_P[3]]
Y_channel = [pauli_P[0] + pauli_P[2], pauli_P[1] + pauli_P[3]]
Z_channel = [pauli_P[0] + pauli_P[3], pauli_P[2] + pauli_P[1]]
cap_x = bsc_cap(X_channel[1])
cap_y = bsc_cap(Y_channel[1])
cap_z = bsc_cap(Z_channel[1])

A = np.linalg.inv(np.matrix([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1]]))


def polar_code_fix(kk, n, p):
    '''
        kk = information bits in 2**n code. 1<= kk <= 2**n - 16.
        16 bits reserved for CRC
        p = error probability
        n => 2**n code. 2**10 = 1024
    '''
    # code construction with prior estimated error probability
    fix_channel = BscChannel(p)
    #### 
    mess = np.random.choice(2, kk) # binary message of len kk
    
    code = PolarCode(n=n, K=kk, construction_method='PW', channel=fix_channel, CRC_len=16)
    
    enc = code.encode(mess)
    err = np.random.choice([0,1], len(enc), p=[1 - p, p])
    #err = np.reshape(np.random.choice([0,1], len(p), p=p), [k, N]).T
    rx = np.mod(enc + err, 2)
    
    # print(err)
    fsc_u_est_message = code.decode(rx, decoding_method='SCL', list_size=32)
    # fsc_u_est_message = code.decode(rx, decoding_method='FSC')
    # print(fsc_u_est_message, len(u_message))
    if fsc_u_est_message is None:
        return np.nan, [np.nan, np.nan], np.nan
    dec_message = code.get_message_info_bits(fsc_u_est_message)
    bit_err_rate = np.sum(mess != dec_message)/kk
    ber_un = sum(enc != rx)/(2**n)
    #print(mess,"\n", enc, "\n\n", err, "\n\n", rx, "\n\n", maj_vote, "\n\n", dec, "\n\n", symb_err_rate)
    #print(symb_err_rate)
    ### Now estimating the probabilities
    
    ##### Re encoding ######
    recoded_msg = code.encode(dec_message)
    ber_re = sum(recoded_msg != rx)/(2**n)
    est_prob = [1 - ber_re, ber_re]
    #print(p, "\n\n", est_prob)
    return(bit_err_rate, est_prob, ber_un)
    


def run_simulation_for_sample(args):
    x_targ, y_targ, z_targ, A, pauli_P, X_channel, Y_channel, Z_channel, N_list, idM = args
    DN_row = np.zeros(len(N_list))
    DN_polar_row = np.zeros(len(N_list))

    for i, n in enumerate(N_list):
        try:
            X_est = np.random.multinomial(2**n, X_channel) / 2**n
            Y_est = np.random.multinomial(2**n, Y_channel) / 2**n
            Z_est = np.random.multinomial(2**n, Z_channel) / 2**n

            b_direct = np.array([X_est[0], Y_est[0], Z_est[0], 1])
            x_direct = np.linalg.matmul(A, b_direct)
            DN_row[i] = np.sum(np.abs(pauli_P - x_direct))

            if n == 8:  # lower targets for n=8
                x_t, y_t, z_t = 0.535, 0.36, 0.34
            else:
                x_t, y_t, z_t = x_targ, y_targ, z_targ
            kk_X = int(np.round(x_t * cap_x * (2**n)))
            ber_X, X_est_polar, _ = polar_code_fix(kk_X, n, X_channel[1])
            kk_Y = int(np.round(y_t * cap_y * (2**n)))
            ber_Y, Y_est_polar, _ = polar_code_fix(kk_Y, n, Y_channel[1])
            kk_Z = int(np.round(z_t * cap_z * (2**n)))
            ber_Z, Z_est_polar, _ = polar_code_fix(kk_Z, n, Z_channel[1])

            b_polar = np.array([X_est_polar[0], Y_est_polar[0], Z_est_polar[0], 1])
            x_polar = np.linalg.matmul(A, b_polar)
            DN_polar_row[i] = np.sum(np.abs(pauli_P - x_polar))
        except:
            DN_row[i] = np.nan
            DN_polar_row[i] = np.nan
        fname = 'Data_20250730/fix_channel2_' + str(idM) + '_.csv'
        data = np.array([np.log2(3 * 2**np.array(N)), DN_row, DN_polar_row]).T
        np.savetxt(fname, data, delimiter = ',')
    return DN_row, DN_polar_row


def run_parallel_samples(M, N_list, x_targ, y_targ, z_targ, A, pauli_P, X_channel, Y_channel, Z_channel):
    args = [(x_targ, y_targ, z_targ, A, pauli_P, X_channel, Y_channel, Z_channel, N_list, idM) for idM in range(M)]

    with mp.Pool(mp.cpu_count()-4) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(run_simulation_for_sample, args),
                           total=M, desc="Running M Samples"):
            results.append(result)



    DN_direct = np.array([r[0] for r in results])   # shape: (M, len(N))
    DN_polar  = np.array([r[1] for r in results])   # shape: (M, len(N))

    return DN_direct.T, DN_polar.T  # reshape to (len(N), M)


        
        
        
if __name__ == '__main__':    
    DN_distance, DN_distance_polar = run_parallel_samples(M, N, x_targ, y_targ, z_targ, A, pauli_P, X_channel, Y_channel, Z_channel)


    
    DN_distance_avg = np.log2(np.mean(DN_distance, axis=1))
    DN_distance_polar_avg = np.log2(np.mean(DN_distance_polar, axis=1))
    ch_uses = np.log2(3 * 2**np.array(N))


    
    plt.plot(ch_uses, DN_distance_avg, 'b', label="EFPE")
    plt.plot(ch_uses, DN_distance_polar_avg, 'r--', label="Polar")
    plt.legend()
    plt.show()






