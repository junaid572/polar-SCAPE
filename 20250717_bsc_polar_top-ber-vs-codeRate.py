# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 13:12:05 2025

@author: junai
"""
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from bsc_polar import bsc_polar0

def run_trial(args):
    """Wrapper function for parallel execution"""
    kk, n, p = args
    ber, _, _ = bsc_polar0(kk, n, p, verbose=False)
    return ber

def main():
    # Parameters
    n = 9
    p = 0.01
    num_trials = 1000
    
    
    # Calculate channel capacity
    bfp = p
    if p == 0:
        capacity = 1
    elif p == 0.5:
        capacity = 0
    else:
        capacity = 1 + bfp*np.log2(bfp) + (1-bfp)*np.log2(1-bfp)
    
    k_list = range(2, int(np.ceil(capacity*2**n)), 10)
    # Prepare arguments for parallel processing
    trial_args = [(kk, n, p) for kk in k_list for _ in range(num_trials)]
    
    # Create a pool of workers
    num_workers = min(cpu_count()-2, len(k_list))  # Leave 2 cores free
    with Pool(processes=num_workers) as pool:
        # Run trials in parallel
        results = pool.map(run_trial, trial_args)
        
        # Reshape results and compute averages
        results = np.array(results).reshape(len(k_list), num_trials)
        ber_list_avg = np.nanmean(results, axis=1)
    
    # Calculate rates normalized to capacity
    rate_list = np.array(k_list) / (2**n)  # Raw code rate
    normalized_rate_list = rate_list / capacity  # Normalized to capacity
    
    # Save data to file
    filename = f"polar0_n{n}_p{p:.3f}.dat"
    data = np.column_stack((normalized_rate_list, ber_list_avg))
    np.savetxt(filename, data, delimiter='\t', 
              header='X\t Y', comments='')
    
    print(f"Data saved to {filename}")
    
    print(ber_list_avg)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_rate_list[::-1], ber_list_avg[::-1], 'b-', linewidth=2)
    plt.xlim(0, 1.1)  # Now shows 1.0 = capacity
    plt.yscale('log')
    plt.xlabel('Code Rate (Normalized to Channel Capacity)')
    plt.ylabel('Average BER (log scale)')
    plt.title(f'Polar Code Performance on BSC (p={p:.3f}, Capacity={capacity:.3f} bits, N={2**n})')
    plt.grid(True, which="both", ls="--")
    
    # Add vertical line at capacity
    plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)
    plt.text(1.02, 0.5*plt.ylim()[1], 'Channel Capacity', rotation=90, color='r')
    plt.ylim([0.0001, 1])
    plt.show()

if __name__ == '__main__':
    main()