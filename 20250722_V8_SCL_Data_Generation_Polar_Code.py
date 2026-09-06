# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:25:32 2025

@author: junai
"""

import numpy as np
from DWC_F import eig_e_corr
from DWC_F import bsc_cap
import matplotlib.pyplot as plt
from collections import Counter
import time

from polar_codes.polar_code import PolarCode
from polar_codes.channels.bsc_channel import BscChannel

from multiprocessing import Pool, cpu_count
import tqdm
import warnings





def TVQC_prob(bias, amplitude, phase, sample_rate, N_samples, freq = 1, d = 2):
    '''
    Sampling the probability distribution of time-varying Pauli channels.
    
    Parameters
    ----------
    bias : Double.
        mean of coefficient of exponential correlatoon.
    amplitude : Double
        amplitude of variation in coefficient of exponential correlation.
        0 <= bias - amplitude, bias + amplitude <= 1
    sample_rate : Double
        sampling rate from quantum channel.
    N_samples : integer
        number of samples to gather from quantum channel.
    freq : double
        frequency of time variation.
    d : integer, optional
        Dimension of Pauli channel. The default is 2.

    Returns
    -------
    N_sample time-varying pmfs of d^2 alphabet.

    '''
    gamma = [bias + amplitude*np.sin(phase + 2*np.pi*freq*time/sample_rate)\
             for time in range(N_samples)]
    time_v = [tt/sample_rate for tt in range(N_samples)]
    probs = [np.sort(eig_e_corr(2, gg))[::-1] for gg in gamma]
    return probs, time_v

def rep_code_adaptive(k, p):
    '''k-repetition code for time-varying BSC with cross-over probability
    1 - p. p is time-varying vector.'''
    N = len(p) # total channel uses

    mess_bits = int(np.floor(N/k)) # message length
    if mess_bits == 0:
        print(k, mess_bits, N)
        print("XXXXXX k too large for available channel uses XXXXXX")
    #### Using channel only for k x mess_bits uses for first
    p = p[0:k*mess_bits]
    mess = np.random.choice(2, mess_bits) # binary message of len N/k
    
    
    enc = np.matlib.repmat(mess, k, 1).T
    err = np.reshape([np.random.choice([0,1], 1, p=prob) \
                      for prob in p], [k, mess_bits]).T
    #err = np.reshape(np.random.choice([0,1], len(p), p=p), [k, N]).T
    rx = np.mod(enc + err, 2)
    maj_vote = [Counter(m).most_common() for m in rx]
    dec = np.array([c[0][0] for c in maj_vote])
    symb_err_rate = (mess_bits - np.sum(mess == dec))/mess_bits
    #print(mess,"\n", enc, "\n\n", err, "\n\n", rx, "\n\n", maj_vote, "\n\n", dec, "\n\n", symb_err_rate)
    #print(symb_err_rate)
    ### Now estimating the probabilities
    dec_rep = np.matlib.repmat(dec, k, 1).T
    err_est = np.mod(rx - dec_rep, 2)
    rx_hamm_weight = np.sum(rx, axis = 1) # since binary
    free_bits = np.abs(k/2 - rx_hamm_weight)-0.5
    # print(np.mean(free_bits), Counter(free_bits).most_common()[0:2])
    #print(Counter(err_est.reshape([1, k*N])[0]).most_common())
    err_counts = Counter(err_est.reshape([1, k*mess_bits])[0]).most_common()
    est_prob = np.zeros(2)
    for kk in err_counts:
        est_prob[kk[0]] = kk[1]/(k*mess_bits)
    #print(p, "\n\n", est_prob)
    return(symb_err_rate, est_prob, Counter(free_bits).most_common())

def polar_code_adaptive_V3(kk, p, err_est, n = 10):
    '''
        kk = information bits in 2**n code. 1<= kk <= 2**n - 16.
        16 bits reserved for CRC
        p = vector of error probabilities (time-varying)
        n => 2**n code. 2**10 = 1024
        NEW in V2: fix channel based on error estimate of previous channel
        New in V3 (V8 of file): using SCL decoder, which is better than FSC
    '''
    if len(p) != 2**n:
        raise ValueError("The list of probabilities should match number of channel uses.", len(p), 2**n)
    
    
    # code construction with prior estimated error probability
    fix_channel = BscChannel(err_est)
    #### 
    mess = np.random.choice(2, kk) # binary message of len kk
    
    code = PolarCode(n=n, K=kk, construction_method='PW', channel=fix_channel, CRC_len=16)
    
    enc = code.encode(mess)
    err = np.reshape([np.random.choice([0,1], 1, p=prob) for prob in p], [1, len(p)])[0]
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

# %% 
# 

def adaptive_polar_tomo(idx):
    n = 10
    N_tomo = 3*(2**n) # number of channel uses in each estimate
    tot_exp = 64 # total number of times we want estimation
    N_samples = N_tomo*tot_exp
    freq = 1
    inc_perc = 0.07
    dec_perc = 0.05
    inc_after = 10
    inc_wait = 8
    # 8-5
    target_perc_x = 0.6
    target_perc_y = 0.45
    target_perc_z = 0.4
    
    # 8-4 below
    # target_perc_x = 0.6
    # target_perc_y = 0.3
    # target_perc_z = 0.3
    k_curr_x, k_curr_y, k_curr_z = 200, 20, 2
    perc_x, perc_y, perc_z = 0.25*target_perc_x, 0.25*target_perc_y, 0.25*target_perc_z
    k_x, k_y, k_z = [], [], []
    perc_x_v, perc_y_v, perc_z_v = [], [], []
    ch_estimates = []
    ch_estimates_nan = [] # hold nan when unreliable, regardless of history
    DN_distance = []
    DN_distance_nan = []
    bias, amplitude, phase, sample_rate\
        = 0.8, 0.1, np.pi/2, 50_000
    probs, time_v = TVQC_prob(bias, amplitude,phase, sample_rate, N_samples, \
                                  freq = freq)
    
    time_snaps = np.round(np.arange(0, N_samples+1, \
                                    int(np.round(N_samples/tot_exp))))
    reliability = []
    cap_x = []
    cap_act_x = []
    rat_x = []
    cap_y = []
    cap_act_y = []
    rat_y = []
    cap_z = []
    cap_act_z = []
    rat_z = []
    prob_x = []
    prob_y = []
    prob_z = []
    # keep last valid estimate to reuse after an unreliable block
    last_valid_prob_x = [0.9, 0.1]
    last_valid_prob_y = [0.9, 0.1]
    last_valid_prob_z = [0.9, 0.1]
    
    reliability_idx_x = 0
    reliability_idx_y = 0
    reliability_idx_z = 0
    for ii in range(len(time_snaps)-1):
        # picking channel probabilities for current time period
        ch_varying = probs[int(time_snaps[ii]):int(time_snaps[ii+1])]
        
        N_basis = int(len(ch_varying)/3) # number of samples for each basis
        ch_varying_for_x = ch_varying[:N_basis]
        ch_varying_for_y = ch_varying[N_basis:2*N_basis]
        ch_varying_for_z = ch_varying[2*N_basis:]
        ch_varying_x = [[prob[0] + prob[1], 1 - prob[0] - prob[1]] \
                        for prob in ch_varying_for_x]
        ch_varying_y = [[prob[0] + prob[2], 1 - prob[0] - prob[2]] \
                        for prob in ch_varying_for_y]
        ch_varying_z = [[prob[0] + prob[3], 1 - prob[0] - prob[3]] \
                        for prob in ch_varying_for_z]
        # simulating the corresponding BSCs
        if len(prob_x) == 0:
            prob_x = [0.9, 0.1]
        if len(prob_y) == 0:
            prob_y = [0.9, 0.1]
        if len(prob_z) == 0:
            prob_z = [0.9, 0.1]
        err_x, prob_x, ber_un = polar_code_adaptive_V3(k_curr_x, ch_varying_x, prob_x[1], n = n)
        #print(err_x, prob_x)
        #
        err_y, prob_y, ber_un = polar_code_adaptive_V3(k_curr_y, ch_varying_y, prob_y[1], n = n)
        #print(err_y, prob_y)
        #
        err_z, prob_z, ber_un = polar_code_adaptive_V3(k_curr_z, ch_varying_z, prob_z[1], n = n)
        #print(err_x, prob_x)
        rat_x.append(k_curr_x/(2**n))
        rat_y.append(k_curr_y/(2**n))
        rat_z.append(k_curr_z/(2**n))
        # checking reliability of all three codes
        
        
        # capacity x, reliability x
        capacity_x = bsc_cap(prob_x[0])        
        if np.isnan(prob_x[0]): #unreliable
            reliability_x = 0
            reliability_idx_x = 0
            cap_x.append(np.nan)
            perc_x -= dec_perc
            perc_x = max(0.001, perc_x)
            if len(cap_x) > 4:
                slice = cap_x[-1:-4:-1]
                if np.all(np.isnan(slice)):
                    cap_ref = 0.5
                else:
                    cap_ref = np.nanmin(slice)
            else:
                cap_ref = 0.5
        else: # reliable
            reliability_x = 1
            cap_x.append(capacity_x)
            reliability_idx_x += 1
            if len(cap_x) > 4:
                slice = cap_x[-1:-4:-1]
                if np.all(np.isnan(slice)):
                    cap_ref = 0.5
                else:
                    cap_ref = np.nanmin(slice)
            else:
                cap_ref = 0.5
            if reliability_idx_x >= inc_after:
                # k_curr_x = min(2**n - 16, int(np.ceil(1.1*k_curr_x)))
                perc_x += inc_perc
                perc_x = min(target_perc_x, perc_x)
                reliability_idx_x = inc_wait
        cap_act_x.append(bsc_cap(ch_varying_x[2**(n - 1)][0]))
        perc_x_v.append((k_curr_x/((2**n)*cap_act_x[-1])))
        k_curr_x = max(1, int(np.floor(perc_x*2**n*cap_ref))) # setting at percent
        if np.isnan(prob_x[0]): # unreliable, use last valid estimate
            prob_x = last_valid_prob_x
        else:
            last_valid_prob_x = prob_x
        
        
        
        
        
        # capacity y, reliability y
        capacity_y = bsc_cap(prob_y[0])        
        if np.isnan(prob_y[0]): #unreliable
            reliability_y = 0
            reliability_idx_y = 0
            cap_y.append(np.nan)
            perc_y -= dec_perc
            perc_y = max(0.001, perc_y)
            if len(cap_y) > 4:
                slice = cap_y[-1:-4:-1]
                if np.all(np.isnan(slice)):
                    cap_ref = 0.5
                else:
                    cap_ref = np.nanmin(slice)
            else:
                cap_ref = 0.5
        else: # reliable
            reliability_y = 1
            cap_y.append(capacity_y)
            reliability_idx_y += 1
            if len(cap_y) > 4:
                slice = cap_y[-1:-4:-1]
                if np.all(np.isnan(slice)):
                    cap_ref = 0.5
                else:
                    cap_ref = np.nanmin(slice)
            else:
                cap_ref = 0.5
            if reliability_idx_y >= inc_after:
                # k_curr_x = min(2**n - 16, int(np.ceil(1.1*k_curr_x)))
                perc_y += inc_perc
                perc_y = min(target_perc_y, perc_y)
                reliability_idx_y = inc_wait
        cap_act_y.append(bsc_cap(ch_varying_y[2**(n - 1)][0]))
        perc_y_v.append((k_curr_y/((2**n)*cap_act_y[-1])))
        k_curr_y = max(1, int(np.floor(perc_y*2**n*cap_ref))) # setting at percent
        if np.isnan(prob_y[0]): # unreliable, use last valid estimate
            prob_y = last_valid_prob_y
        else:
            last_valid_prob_y = prob_y
        
        
        # capacity z, reliability z
        capacity_z = bsc_cap(prob_z[0])        
        if np.isnan(prob_z[0]): #unreliable
            reliability_z = 0
            reliability_idx_z = 0
            cap_z.append(np.nan)
            perc_z -= dec_perc
            perc_z = max(0.001, perc_z)
            if len(cap_z) > 4:
                slice = cap_z[-1:-4:-1]
                if np.all(np.isnan(slice)):
                    cap_ref = 0.5
                else:
                    cap_ref = np.nanmin(slice)
            else:
                cap_ref = 0.5
        else: # reliable
            reliability_z = 1
            cap_z.append(capacity_z)
            reliability_idx_z += 1
            if len(cap_z) > 4:
                slice = cap_z[-1:-4:-1]
                if np.all(np.isnan(slice)):
                    cap_ref = 0.5
                else:
                    cap_ref = np.nanmin(slice)
            else:
                cap_ref = 0.5
            if reliability_idx_z >= inc_after:
                # k_curr_x = min(2**n - 16, int(np.ceil(1.1*k_curr_x)))
                perc_z += inc_perc
                perc_z = min(target_perc_z, perc_z)
                reliability_idx_z = inc_wait
        cap_act_z.append(bsc_cap(ch_varying_z[2**(n - 1)][0]))
        perc_z_v.append((k_curr_z/((2**n)*cap_act_z[-1])))
        k_curr_z = max(1, int(np.floor(perc_z*2**n*cap_ref))) # setting at percent
        if np.isnan(prob_z[0]): # unreliable, use last valid estimate
            prob_z = last_valid_prob_z
        else:
            last_valid_prob_z = prob_z
                
                
                
        rel = reliability_x == 1 and reliability_y == 1 and reliability_z == 1
        if rel: # estimate is reliable
        # Estimating the full Pauli channel
            est_Pauli = [0.5*(prob_x[0] + prob_y[0] + prob_z[0] - 1), \
                         0.5*(prob_x[0] - prob_y[0] - prob_z[0] + 1), \
                         0.5*(-prob_x[0] + prob_y[0] - prob_z[0] + 1), \
                         0.5*(-prob_x[0] - prob_y[0] + prob_z[0] + 1)]
            ch_estimates.append(est_Pauli)
            ch_estimates_nan.append(est_Pauli)
        elif rel == False and len(ch_estimates)>1: # estimate is unreliable but we have history
            est_Pauli = ch_estimates[-1]
            ch_estimates.append(ch_estimates[-1]) # current estimate unreliable, using old estimate
            ch_Pauli = [np.nan, np.nan, np.nan, np.nan]
            ch_estimates_nan.append(ch_Pauli)
        else: # no old estimate available, current one unreliable
            ch_Pauli = [np.nan, np.nan, np.nan, np.nan]
            est_Pauli = ch_Pauli
            ch_estimates.append(ch_Pauli)
            ch_estimates_nan.append(ch_Pauli)
        # Calculating the diamond norm distance
        
        DN_distance.append(np.sum(np.abs(est_Pauli - ch_varying[-1])))
        DN_distance_nan.append(np.sum(np.abs(ch_estimates_nan[-1] - ch_varying[-1])))
        
        # for record keeping
        # fb_m_x_v.append(free_b_m_x)
        # fb_m_y_v.append(free_b_m_y)
        # fb_m_z_v.append(free_b_m_z)
        k_x.append(k_curr_x)
        k_y.append(k_curr_y)
        k_z.append(k_curr_z)
        
        
    
    #plt.plot(time_v, probs, '--')
    #plt.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, ch_estimates)
    #plt.show()
    
    # fig, (ax1, ax3, ax4) = plt.subplots(3, figsize=(8, 8))
    # ax2 = ax1.twinx()
        
    # ax1.plot(time_v, probs, '--')
    # ax1.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, ch_estimates, '-')
    # #
    # ax2.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, DN_distance, 'r')
    # ax2.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, DN_distance_nan, 'b')
    # ax2.set_ylim(0,0.4)
    
    # #ax4 = ax3.twinx()
    # ax3.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, k_x, 'r')
    # ax3.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, k_y, 'g')
    # ax3.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, k_z, 'b')
    
    # ax5 = ax4.twinx()
    
    # # ax4.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, cap_x, 'r')
    # # ax4.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, cap_act_x, 'r', lw = 2)
    # # ax4.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, cap_y, 'g')
    # # ax4.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, cap_act_y, 'g', lw = 2)
    # # ax4.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, cap_z, 'b')
    # # ax4.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, cap_act_z, 'b', lw = 2)
    
    # ax4.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, rat_x, 'r--')
    # ax5.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, perc_x_v, 'r', lw = 2)
    # ax4.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, rat_y, 'g--')
    # ax5.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, perc_y_v, 'g', lw = 2)
    # ax4.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, rat_z, 'b--')
    # ax5.plot(time_v[-1]*np.array(time_snaps[1:])/N_samples, perc_z_v, 'b', lw = 2)
    # ax5.set_ylim([0, 1.5*target_perc_x])
    #fig.show()
    
    # data1: complete time, actual channel data (ground truth)
    data1 = [[time_v[ii], probs[ii][0], probs[ii][1], probs[ii][2], probs[ii][3]] \
             for ii in range(len(time_v))]
    if idx == 0:
        np.savetxt('Data_20250722/polar8-5_SCL_A_' + str(idx) + '_.csv', data1, delimiter = ',')
    
    time_estimates = time_v[-1]*np.array(time_snaps[1:])/N_samples
    data2 = [[time_estimates[ii], ch_estimates[ii][0], ch_estimates[ii][1], \
              ch_estimates[ii][2], ch_estimates[ii][3], rat_x[ii], \
              rat_y[ii], rat_z[ii], \
              cap_x[ii], cap_y[ii], cap_z[ii], \
              cap_act_x[ii], cap_act_y[ii], cap_act_z[ii],\
              perc_x_v[ii], perc_y_v[ii], perc_z_v[ii],\
              DN_distance[ii], DN_distance_nan[ii]] \
             for ii in range(len(time_estimates))]
    np.savetxt('Data_20250722/polar8-5_SCL_B_' + str(idx) + '_.csv', data2, delimiter = ',')
    
if __name__ == '__main__':
    N = 1000
    num_workers = min(cpu_count()-2, N)
    warnings.filterwarnings("ignore")
    with Pool(processes=num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(adaptive_polar_tomo, range(N)), total = N):
            pass