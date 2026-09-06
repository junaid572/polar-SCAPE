# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:17:15 2025

@author: junai
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:08:46 2025

@author: junai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 21:09:43 2025

@author: junaidrehman
"""

# https://github.com/RQC-QApp/polar-codes
import numpy as np
#import matplotlib
#matplotlib.use('Qt5Agg')  # Or 'TkAgg' or other suitable interactive backends

#import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from bsc_polar import bsc_polar16
#import time
#import math
# import matplotlib.pyplot as plt
# plt.ion()  # Enable interactive mode




def run_trial(args):
    """Wrapper function for parallel execution"""
    kk, n, p = args
    ber, ber_un, ber_re = bsc_polar16(kk, n, p, verbose=False)
    return ber

def main():
    # Parameters
    n = 13
    p = 0.084
    # num_trials = 10
    inc_perc = 0.05
    dec_perc = 0.05
    num_proc = cpu_count() - 2
#    fig, ax = plt.subplots()

    # Calculate channel capacity
    bfp = p
    if p == 0:
        capacity = 1
    elif p == 0.5:
        capacity = 0
    else:
        capacity = 1 + bfp*np.log2(bfp) + (1-bfp)*np.log2(1-bfp)
    print(capacity)
    # starting with a generic descent implementation
    start_k = 3840# int(np.ceil(capacity*2**n)) # starting large, will bring it down to appropriate value
    history = 1000 # time steps of histor
    reliability_idx = 0 # keeping a history of f*k ups.
    k_history = np.zeros(history)
    c_history = np.zeros(history)
    kk = start_k
    percentage = 0.6
    ii = 0
    highest_reliability = 0 
    while 1:
        with Pool(num_proc) as pool:
            results = []
            args = [(kk, n, p) for _ in range(num_proc)]
            for result in tqdm(pool.imap_unordered(run_trial, args),
                               total=num_proc, desc="Running M Samples"):
                results.append(result)
        #ber, ber_un, ber_re = bsc_polar16(kk, n, p, verbose=False)
        #print(results)
        if np.isnan(results).sum() > 0: # unreliable, decrease code rate
            reliability_idx = 0 # forcing to wait before trying to increase again
            kk -= 1
            print("*************************")
            print("Reliability reset. New kk")
            print("Current k:\t", kk)
            print("*************************")
        else: # reliable
            reliability_idx += num_proc
            print("*************************")
            print("Reliability:\t", reliability_idx)
            print("Current k:\t", kk,\
                  "\nCR:\t", kk/(2**n),\
                  "\nPercentage:\t", kk/((2**n)*capacity))
            print("*************************")
        
        if reliability_idx >= 1000:
            print("Success!")
            break

        
if __name__ == '__main__':
    main()