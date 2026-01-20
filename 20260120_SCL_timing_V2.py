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



import time
import numpy as np

def polar_code_adaptive_timing(kk, p, err_est, n=10):
    '''
        returns timings of code generation, encoding, decoding operations
        
        kk = information bits in 2**n code. 1<= kk <= 2**n - 16.
        16 bits reserved for CRC
        p = vector of error probabilities (time-varying)
        n => 2**n code. 2**10 = 1024
        NEW in V2: fix channel based on error estimate of previous channel
        New in V3: using SCL decoder
    '''

    if len(p) != 2**n:
        raise ValueError("The list of probabilities should match number of channel uses.",
                         len(p), 2**n)

    N = 2**n
    timings = {}

    # ----------------------------------------------------
    # Randomness (NOT TIMED)
    # ----------------------------------------------------
    mess = np.random.choice(2, kk)
    #print(p)
    err = np.array([np.random.choice([0,1], p=[prob[0]+prob[1], prob[2]+prob[3]]) for prob in p])

    # ----------------------------------------------------
    # Code construction timing
    # ----------------------------------------------------
    t0 = time.perf_counter()
    fix_channel = BscChannel(err_est)
    code = PolarCode(n=n, K=kk, construction_method='PW',
                     channel=fix_channel, CRC_len=16)
    timings["construct"] = time.perf_counter() - t0

    # ----------------------------------------------------
    # Encoding timing
    # ----------------------------------------------------
    t1 = time.perf_counter()
    enc = code.encode(mess)
    timings["encode"] = time.perf_counter() - t1

    # Apply noise (not timed)
    rx = (enc + err) % 2

    # ----------------------------------------------------
    # Decoding timing
    # ----------------------------------------------------
    t2 = time.perf_counter()
    fsc_u_est_message = code.decode(rx, decoding_method='SCL', list_size=32)
    timings["decode"] = time.perf_counter() - t2


    return timings
# %% 
# 
if __name__ == '__main__':

    n = 10
    N = 2**n
    runs = 100
    err_est = 0.1
    freq = 1
    bias, amplitude, phase, sample_rate = 0.8, 0.1, np.pi/2, 50_000

    # 10 values of k between small and large
    k_values = np.linspace(N//512, N - N//512 -16, 10, dtype=int)

    # Generate time-varying Pauli channel probabilities once
    probs, _ = TVQC_prob(bias, amplitude, phase, sample_rate, N, freq=freq)

    construct_mean = []
    encode_mean = []
    decode_mean = []

    construct_std = []
    encode_std = []
    decode_std = []

    for kk in k_values:
        print(f"Running k = {kk} (N = {N}) ...")

        c_times, e_times, d_times = [], [], []

        for _ in range(runs):
            timings = polar_code_adaptive_timing(kk, probs, err_est, n=n)
            c_times.append(timings["construct"])
            e_times.append(timings["encode"])
            d_times.append(timings["decode"])

        # Store log10 means and stds
        construct_mean.append(np.mean(np.log10(c_times)))
        encode_mean.append(np.mean(np.log10(e_times)))
        decode_mean.append(np.mean(np.log10(d_times)))

        construct_std.append(np.std(np.log10(c_times)))
        encode_std.append(np.std(np.log10(e_times)))
        decode_std.append(np.std(np.log10(d_times)))

    # ----------------------------------------------------
    # Plot results
    # ----------------------------------------------------
    plt.figure(figsize=(10,6))

    plt.errorbar(k_values, construct_mean, yerr=construct_std,
                 fmt='-o', label='Construct')
    plt.errorbar(k_values, encode_mean, yerr=encode_std,
                 fmt='-o', label='Encode')
    plt.errorbar(k_values, decode_mean, yerr=decode_std,
                 fmt='-o', label='Decode')

    plt.xlabel("Information bits k")
    plt.ylabel("log10(Time [seconds])")
    plt.title("Polar Code Timing vs k (mean ± std over runs)")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------
    # Save results as a tab-delimited file (log-safe)
    # ----------------------------------------------------
    filename = "polar_timing_vs_k.txt"
    
    with open(filename, "w") as f:
        # Header row
        f.write(
            "k\t"
            "construct_mean\tconstruct_std\tconstruct_lower\tconstruct_upper\t"
            "encode_mean\tencode_std\tencode_lower\tencode_upper\t"
            "decode_mean\tdecode_std\tdecode_lower\tdecode_upper\n"
        )
    
        # Data rows
        for i in range(len(k_values)):
            cm, cs = construct_mean[i], construct_std[i]
            em, es = encode_mean[i], encode_std[i]
            dm, ds = decode_mean[i], decode_std[i]
    
            # log-safe lower bounds
            cl = max(cm - cs, 1e-12)
            el = max(em - es, 1e-12)
            dl = max(dm - ds, 1e-12)
    
            # upper bounds
            cu = cm + cs
            eu = em + es
            du = dm + ds
    
            f.write(
                f"{k_values[i]}\t"
                f"{cm:.8e}\t{cs:.8e}\t{cl:.8e}\t{cu:.8e}\t"
                f"{em:.8e}\t{es:.8e}\t{el:.8e}\t{eu:.8e}\t"
                f"{dm:.8e}\t{ds:.8e}\t{dl:.8e}\t{du:.8e}\n"
            )
    
    print(f"\nTiming data written to: {filename}")


