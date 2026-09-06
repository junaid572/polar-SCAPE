# -*- coding: utf-8 -*-
"""
Fig. 3 (total error) and Fig. 4 (diamond norm distance) data for polar SCAPE
and EFPE, at a fixed channel (gamma = 0.8). Run from the repository root.
Outputs fig3_fig4_estimates.npz and Data_Processed_20250730/fix_channel2_R2.dat,
and prints the Fig. 3 values.
"""
import os
import numpy as np
from DWC_F import eig_e_corr, bsc_cap
from polar_codes.polar_code import PolarCode
from polar_codes.channels.bsc_channel import BscChannel
import multiprocessing as mp
from tqdm import tqdm

N_LIST = [8, 9, 10, 11, 12, 13]
M = 1000
GAMM = 0.8
# capacity fractions (X, Y, Z), lower at n=8
TARGETS = {8: (0.535, 0.36, 0.34)}
DEFAULT_TARGETS = (0.6, 0.45, 0.4)
LIST_SIZE = 32

pauli_P = eig_e_corr(2, GAMM)
X_channel = [pauli_P[0] + pauli_P[1], pauli_P[2] + pauli_P[3]]
Y_channel = [pauli_P[0] + pauli_P[2], pauli_P[1] + pauli_P[3]]
Z_channel = [pauli_P[0] + pauli_P[3], pauli_P[2] + pauli_P[1]]
cap_x, cap_y, cap_z = (bsc_cap(c[1]) for c in (X_channel, Y_channel, Z_channel))
A = np.linalg.inv(np.matrix([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1]]))


def polar_code_fix(kk, n, p):
    """one polar block, re-encoding estimate"""
    code = PolarCode(n=n, K=kk, construction_method='PW',
                     channel=BscChannel(p), CRC_len=16)
    mess = np.random.choice(2, kk)
    enc = code.encode(mess)
    err = np.random.choice([0, 1], len(enc), p=[1 - p, p])
    rx = np.mod(enc + err, 2)
    dec = code.decode(rx, decoding_method='SCL', list_size=LIST_SIZE)
    if dec is None:                     # CRC fail
        return [np.nan, np.nan], False
    recoded = code.encode(code.get_message_info_bits(dec))
    ber_re = np.sum(recoded != rx) / (2**n)
    return [1 - ber_re, ber_re], True


def run_sample(idM):
    nN = len(N_LIST)
    est_e = np.full((nN, 4), np.nan)
    est_p = np.full((nN, 4), np.nan)
    fails = np.zeros((nN, 3), dtype=np.int8)
    for i, n in enumerate(N_LIST):
        xt, yt, zt = TARGETS.get(n, DEFAULT_TARGETS)
        X_est = np.random.multinomial(2**n, X_channel) / 2**n
        Y_est = np.random.multinomial(2**n, Y_channel) / 2**n
        Z_est = np.random.multinomial(2**n, Z_channel) / 2**n
        b = np.array([X_est[0], Y_est[0], Z_est[0], 1])
        est_e[i, :] = np.asarray(np.linalg.matmul(A, b)).ravel()

        pX, okX = polar_code_fix(int(np.round(xt * cap_x * 2**n)), n, X_channel[1])
        pY, okY = polar_code_fix(int(np.round(yt * cap_y * 2**n)), n, Y_channel[1])
        pZ, okZ = polar_code_fix(int(np.round(zt * cap_z * 2**n)), n, Z_channel[1])
        fails[i, :] = [not okX, not okY, not okZ]
        b = np.array([pX[0], pY[0], pZ[0], 1])
        est_p[i, :] = np.asarray(np.linalg.matmul(A, b)).ravel()
    return idM, est_e, est_p, fails


def main():
    nN = len(N_LIST)
    EST_E = np.full((M, nN, 4), np.nan)
    EST_P = np.full((M, nN, 4), np.nan)
    FAILS = np.zeros((M, nN, 3), dtype=np.int8)
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        for idM, est_e, est_p, fails in tqdm(pool.imap_unordered(run_sample, range(M)),
                                             total=M, desc="runs"):
            EST_E[idM], EST_P[idM], FAILS[idM] = est_e, est_p, fails
    np.savez("fig3_fig4_estimates.npz", EST_E=EST_E, EST_P=EST_P, FAILS=FAILS,
             pauli_P=pauli_P, N_LIST=np.array(N_LIST))

    print("\ndecoding failures (out of M = %d):" % M)
    print("  n   fails X/Y/Z")
    for i, n in enumerate(N_LIST):
        fx, fy, fz = FAILS[:, i, :].sum(axis=0)
        print(f"{n:>3}   {fx}/{fy}/{fz}")

    # Fig. 3 total error tr(Cov(p_hat))
    print("\nFig. 3 total error:")
    print("  n   channel uses   EFPE          polar SCAPE")
    for i, n in enumerate(N_LIST):
        ve = np.sum(np.var(EST_E[:, i, 1:4], axis=0))
        vp = np.sum(np.var(EST_P[:, i, 1:4], axis=0))
        print(f"{n:>3}   3*2^{n:<10} {ve:.6e}  {vp:.6e}")

    print("\nFig. 3 polar SCAPE points "
          "(log10 channel uses, log10 total error):")
    for i, n in enumerate(N_LIST):
        vp = np.sum(np.var(EST_P[:, i, 1:4], axis=0))
        print(f"{np.log10(3.0 * 2.0**n):.6f}\t{np.log10(vp):.6f}")

    # Fig. 4 data (diamond norm distance)
    DN_E = np.sum(np.abs(EST_E - pauli_P), axis=2)
    DN_P = np.sum(np.abs(EST_P - pauli_P), axis=2)
    X = np.log2(3.0 * 2.0**np.array(N_LIST, dtype=float))
    data = np.column_stack([
        X,
        np.mean(np.log2(DN_E), axis=0), 0.5 * np.std(np.log2(DN_E), axis=0),
        np.mean(np.log2(DN_P), axis=0), 0.5 * np.std(np.log2(DN_P), axis=0)])
    os.makedirs("Data_Processed_20250730", exist_ok=True)
    np.savetxt("Data_Processed_20250730/fix_channel2_R2.dat", data, delimiter="\t",
               header="X\tY1\tY2\tY3\tY4", comments="")
    print("\nwrote fig3_fig4_estimates.npz and "
          "Data_Processed_20250730/fix_channel2_R2.dat")


if __name__ == "__main__":
    main()
