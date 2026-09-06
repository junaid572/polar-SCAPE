# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:46:45 2025

@author: junai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 22:34:53 2023

@author: junaidrehman
"""

import os
import numpy as np
import matplotlib.pyplot as plt

targ_perc_x = 0.6
targ_perc_y = 0.3
targ_perc_z = 0.2

write_file = False


data1 = np.genfromtxt('Data_20250722/polar8-5_SCL_A_' + str(0) + '_.csv', delimiter = ',')
time_v = [ii[0] for ii in data1]
TVQC = [[ii[1], ii[2], ii[3], ii[4]] for ii in data1]

data2 = np.genfromtxt('Data_20250722/polar8-5_SCL_B_' + str(0) + '_.csv', delimiter = ',')
time_estimates = [ii[0] for ii in data2]
cap_act = data2[:, 11:14]

# B: time stamps of estimates, estimated channel, errors, k, free_bits diamond norm distance
pI, pX, pY, pZ = [], [], [], []
e_X, e_Y, e_Z = [], [], []
k_X, k_Y, k_Z = [], [], []
fb_X, fb_Y, fb_Z = [], [], []
DN_distance = []
DN_distance_nan = []
ch_estimates = []
cap_estimates = []
rates = []
perc_ach = []


for idx in range(1000):
    data2 = np.genfromtxt('Data_20250722/polar8-5_SCL_B_' + str(idx) + '_.csv', delimiter = ',')
    DN_distance_nan.append(data2[:, -1])
    DN_distance.append(data2[:, -2])
    ch_estimates.append(data2[:, 1:5])
    rates.append(data2[:, 5:8])
    cap_estimates.append(data2[:, 8:11])
    perc_ach.append(data2[:, 14:17])
    
    #print(idx)

# setting rate = 0 for unreliable (NaN) blocks
rates = np.array(rates)
rates[np.isnan(np.array(cap_estimates))] = 0.0

mean_DN_distance = np.mean(DN_distance, axis = 0)
std_DN = 0.5*np.std(DN_distance, axis = 0)

mean_ch_estimates = np.mean(ch_estimates, axis = 0)
std_ch_estimates = 0.5*np.std(ch_estimates, axis = 0)

mean_rates = np.mean(rates, axis = 0)
min_rates = np.min(rates, axis = 0)
max_rates = np.max(rates, axis = 0)
std_rates = 0.5*np.std(rates, axis = 0)

mean_cap_estimates = np.nanmean(cap_estimates, axis = 0)
std_cap_estimates = 0.5*np.nanstd(cap_estimates, axis = 0)

mean_perc_estimates = np.mean(perc_ach, axis = 0)
std_perc_estimates = 0.5*np.std(perc_ach, axis = 0)

#%% plotting starts here

fig, (ax1, ax3, ax4, ax5) = plt.subplots(4, figsize=(8, 8))
ax2 = ax1.twinx()

ax1.fill_between(time_estimates, mean_ch_estimates[:,0]-std_ch_estimates[:,0], \
                 mean_ch_estimates[:,0]+std_ch_estimates[:,0], \
                 alpha = 0.4, color = 'blue')
ax1.fill_between(time_estimates, mean_ch_estimates[:,1]-std_ch_estimates[:,1], \
                 mean_ch_estimates[:,1]+std_ch_estimates[:,1], \
                 alpha = 0.4, color = 'blue')
ax1.fill_between(time_estimates, mean_ch_estimates[:,2]-std_ch_estimates[:,2], \
                 mean_ch_estimates[:,2]+std_ch_estimates[:,2], \
                 alpha = 0.4, color = 'blue')
ax1.fill_between(time_estimates, mean_ch_estimates[:,3]-std_ch_estimates[:,3], \
                 mean_ch_estimates[:,3]+std_ch_estimates[:,3], \
                 alpha = 0.4, color = 'blue')
ax1.plot(time_v, TVQC, '--')
ax1.plot(time_estimates, mean_ch_estimates)
TVQCa = np.array(TVQC)
#ax1.plot(time_v[0:-1:1000], 1- (TVQCa[0:-1:1000,0] + TVQCa[0:-1:1000,1]), '*-')
#ax1.plot(time_v[0:-1:1000], 1- (TVQCa[0:-1:1000,0] + TVQCa[0:-1:1000,2]), '*-')
#ax1.plot(time_v[0:-1:1000], 1- (TVQCa[0:-1:1000,0] + TVQCa[0:-1:1000,3]), '*-')
# ax1.plot(time_estimates, mean_ch_estimates)
#ax1.plot(time_estimates, mean_ch_estimates)
# ax1.plot(time_v[0:-1:1000], TVQCa[0:-1:1000,0], 'r--', label = "pI")
# ax1.plot(time_v[0:-1:1000], TVQCa[0:-1:1000,1], 'g--', label = "pX")
# ax1.plot(time_v[0:-1:1000], TVQCa[0:-1:1000,2], 'b--', label = "pY")
# ax1.plot(time_v[0:-1:1000], TVQCa[0:-1:1000,3], 'k--', label = "pZ")


ax2.fill_between(time_estimates, mean_DN_distance-std_DN, mean_DN_distance+std_DN, \
                 alpha = 0.4, color = 'red')
ax2.plot(time_estimates, mean_DN_distance)
ax1.set_title("Pauli Parameters")

# ax2.set_ylim(0,0.05)
################################
ax3.fill_between(time_estimates, mean_rates[:,0]-std_rates[:,0], \
                  mean_rates[:,0]+std_rates[:,0], \
                  alpha = 0.4, color = 'blue')
ax3.fill_between(time_estimates, mean_rates[:,1]-std_rates[:,1], \
                  mean_rates[:,1]+std_rates[:,1], \
                  alpha = 0.4, color = 'blue')
ax3.fill_between(time_estimates, mean_rates[:,2]-std_rates[:,2], \
                  mean_rates[:,2]+std_rates[:,2], \
                  alpha = 0.4, color = 'blue')
#ax3.plot(time_v, TVQC, '--')
ax3.plot(time_estimates, mean_rates)
########
ax3.plot(time_estimates, targ_perc_x*mean_cap_estimates[:,0])
ax3.plot(time_estimates, targ_perc_y*mean_cap_estimates[:,1])
ax3.plot(time_estimates, targ_perc_z*mean_cap_estimates[:,2])
######
ax3.set_ylim([0, 0.6])
ax3.set_title("Achieved Rates")
################################
ax4.fill_between(time_estimates, mean_cap_estimates[:,0]-std_cap_estimates[:,0], \
                 mean_cap_estimates[:,0]+std_cap_estimates[:,0], \
                 alpha = 0.4, color = 'blue')
ax4.fill_between(time_estimates, mean_cap_estimates[:,1]-std_cap_estimates[:,1], \
                 mean_cap_estimates[:,1]+std_cap_estimates[:,1], \
                 alpha = 0.4, color = 'blue')
ax4.fill_between(time_estimates, mean_cap_estimates[:,2]-std_cap_estimates[:,2], \
                 mean_cap_estimates[:,2]+std_cap_estimates[:,2], \
                 alpha = 0.4, color = 'blue')
#ax3.plot(time_v, TVQC, '--')
ax4.plot(time_estimates, mean_cap_estimates)
ax4.plot(time_estimates, cap_act, '--')
ax4.set_title("Estimated and actual capacities")
################################
ax5.fill_between(time_estimates, mean_perc_estimates[:,0]-std_perc_estimates[:,0], \
                 mean_perc_estimates[:,0]+std_perc_estimates[:,0], \
                 alpha = 0.4, color = 'blue')
ax5.fill_between(time_estimates, mean_perc_estimates[:,1]-std_perc_estimates[:,1], \
                 mean_perc_estimates[:,1]+std_perc_estimates[:,1], \
                 alpha = 0.4, color = 'blue')
ax5.fill_between(time_estimates, mean_perc_estimates[:,2]-std_perc_estimates[:,2], \
                 mean_perc_estimates[:,2]+std_perc_estimates[:,2], \
                 alpha = 0.4, color = 'blue')
#ax3.plot(time_v, TVQC, '--')
ax5.plot(time_estimates, mean_perc_estimates)
ax5.set_ylim([0, 0.8])
ax5.set_title("percentage achievement")

plt.show()

if write_file:
    # data 1: samplling time. TVQC
    # data 2: estimation time, estimated channel. ch_std, DN distance. std_DN
    data1_export = [[time_v[ii]] + TVQC[ii] for ii in range(0, len(TVQC), 200)] 
    data2_export = [[time_estimates[ii]] + mean_ch_estimates[ii].tolist() \
                    + std_ch_estimates[ii].tolist() + [mean_DN_distance[ii]] \
                        + [std_DN[ii]] for ii in range(len(mean_ch_estimates))]
    
    
    # data3: estimation time, error rates, code index k, CRI
    data3_export = [[time_estimates[ii]] + 
                    [mean_rates[ii,0]] + [std_rates[ii,0]] +\
                    [mean_rates[ii,1]] + [std_rates[ii,1]] +\
                    [mean_rates[ii,2]] + [std_rates[ii,2]] +\
                    [mean_cap_estimates[ii,0]] + [std_cap_estimates[ii,0]] +\
                    [mean_cap_estimates[ii,1]] + [std_cap_estimates[ii,1]] +\
                    [mean_cap_estimates[ii,2]] + [std_cap_estimates[ii,2]] +\
                    [cap_act[ii,0]] + [cap_act[ii,1]] + [cap_act[ii,2]] +\
                    [mean_perc_estimates[ii,0]] + [std_perc_estimates[ii,0]] +\
                    [mean_perc_estimates[ii,1]] + [std_perc_estimates[ii,1]] +\
                    [mean_perc_estimates[ii,2]] + [std_perc_estimates[ii,2]] \
                        for ii in range(len(mean_ch_estimates))]
    
    num_columns = len(data1_export[0])
    header1 = "X" + "".join([f"\tY{i}" for i in range(1, num_columns)])
    num_columns = len(data2_export[0])
    header2 = "X" + "".join([f"\tY{i}" for i in range(1, num_columns)])
    num_columns = len(data3_export[0])
    header3 = "X" + "".join([f"\tY{i}" for i in range(1, num_columns)])
    
        
    os.makedirs('Data_Processed_20250722', exist_ok = True)
    np.savetxt('Data_Processed_20250722/polar8-5_SCL_A_.dat', data1_export, delimiter = '\t', header = header1, comments = "")
    np.savetxt('Data_Processed_20250722/polar8-5_SCL_B_.dat', data2_export, delimiter = '\t', header = header2, comments = "")
    np.savetxt('Data_Processed_20250722/polar8-5_SCL_C_.dat', data3_export, delimiter = '\t', header = header3, comments = "")
    
    