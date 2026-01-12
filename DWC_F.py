import numpy as np
import cvxpy as cp
from math import pi, log, gcd
from cmath import exp
import numpy.matlib
import qutip as qt
from scipy.stats import entropy
import random
import numpy.matlib
from collections import Counter

# %% Functions

def bsc_cap(p):
    """Returns the capacity of binary symmetric channel having crossover
    probability p.
    """
    if p == 0 or p==1:
        capacity = 1
    elif p == 0.5:
        capacity = 0
    else:
        capacity = 1 + p*np.log2(p) + (1-p)*np.log2(1-p)
    return capacity

def gen_weyl(d):
    """Returns a list of discrete Weyl operators in dimesion d"""
    id_d = np.matrix(np.eye(d))
    w = exp(2*pi*1j/d)
    zero_mat = np.zeros((d,d))
    wO = [[zero_mat for n in range(0,d)] for m in range(0,d)]
    for n in range(d):
        for m in range(d):
            for k in range(d):
                wO[n][m] = wO[n][m] + w**(k*n)*id_d[:,k]*id_d[(k + m)%d,:]
            wO[n][m] = qt.Qobj(wO[n][m]).tidyup()
    return(wO)


def eig_e_corr(d,rho):
    """Returns normalized eigenvalues of d^2 x d^2 dimensional exponential correlation matrix with correlation coefficient rho"""
    ec_Mat = np.zeros((d**2,d**2))
    for x in range(0,d**2):
        for y in range(0,d**2):
            ec_Mat[x,y] = rho**(abs(x - y))
    eig_e = np.linalg.eig(ec_Mat)
    eig_e = eig_e[0]/np.sum(eig_e[0])
    return(eig_e)

def apply_map(rho_in, J):
    """Returns the output state when the rho_in is input to CPTP specified by the Choi Matrix J.
    Both rho_in and J are Qobj of qutip"""
    d = rho_in.dims[0]
    A = qt.tensor(qt.Qobj(np.matrix(rho_in).T), qt.qeye(d))
    #B = qt.tensor(qt.qeye(d), qt.Qobj(np.matrix(rho_in).T))
    rho_out = (A*J).ptrace(1)
    return(rho_out)

def apply_kraus(rho_in, Kr, Pxy):
    """Returns the output state when the input is rho_in. CPTP is defined by Kraus operators Kr"""
    d = rho_in.dims[0][0]
    rho_out = np.zeros((d,d))
    for nn in range(d):
        for mm in range(d):
            rho_out += Pxy[nn][mm]*Kr[nn][mm]*rho_in*Kr[nn][mm].dag()
    return(rho_out)

def get_choi_DWC(Pxy, weyl_oper):
    """Returns the Choi matrix of discrete Weyl channel with Weyl operators W and corresponding probabilities Pxy"""
    d = len(Pxy)
    S = sum([Pxy[n][m]*qt.sprepost(weyl_oper[n][m], weyl_oper[n][m].dag()) for n in range(d) for m in range(d)])
    J = qt.to_choi(S) # Choi Matrix
    J.dims = [[d,d],[d,d]]
    return(J)

def get_eig_relations(d):
    """Returns the input output relations of DWC excited by the eigenstates of Weyl operators.
    Currently only for prime d. Will extend soon"""
    eigRel = np.zeros((d+1,d**2))
    Sniter = [1] + [ii for ii in range(d)]
    Smiter = [0] + [1 for ii in range(d)]
    for kk in range(len(Sniter)):
        appOp = 0;
        for nn in range(d):
            for mm in range(d):
                eigRel[kk,appOp] = (Smiter[kk]*nn - Sniter[kk]*mm)%d#(mm*g - nn*t)%d
                appOp +=1
    eigRel = {(Sniter[ii], Smiter[ii]): list(eigRel[ii]) for ii in range(len(Sniter))}
    return(eigRel)

from functools import lru_cache

@lru_cache(maxsize=2)
def get_eig_relations_comp(d):
    """Returns the input output relations of DWC excited by the eigenstates of Weyl operators in composite d"""
    Sniter = np.array(range(d))
    Sniter = np.matlib.repmat(Sniter,d,1)
    Sniter = Sniter.reshape((1,d**2))
    Smiter = np.array(range(d))
    Smiter = np.matlib.repmat(Smiter,d,1)
    Smiter = Smiter.T.reshape((1,d**2))

    # starting with all operators except (0,0)
    Sniter = Sniter[0][1:]
    Smiter = Smiter[0][1:]
    #print("Start Sniter, Smiter:", Sniter, Smiter)
    # Removing operators with gcd(n,m,d)>1, otherwise not full rank
    Sngcd = [gcd(Sniter[ii], d)%d for ii in range(len(Sniter))]
    Smgcd = [gcd(Smiter[ii], d)%d for ii in range(len(Smiter))]
    #print("Start Sngcd, Smgcd:", Sngcd, Smgcd)
    Sngcd[:] = [d if x==0 else x for x in Sngcd]
    Smgcd[:] = [d if x==0 else x for x in Smgcd]
    gcdComm = (np.array(Sngcd) >1) & (np.array(Smgcd) >1) # d+1 should be 2, just checking
#    print("gcdComm: ", gcdComm,"\nSngcd: ", Sngcd,"\nSmgcd: ", Smgcd)
    Sniter = Sniter[~gcdComm]
    Smiter = Smiter[~gcdComm]
    #print(type(Sniter))
#     print("Last Sniter, Smiter:", Sniter, Smiter)
    # removing operators which commte with any other in the set
    pop_oper = [] # operators to be removed
    for ii in range(len(Sniter)):
        p = Sniter[ii]
        q = Smiter[ii]
        kk = (q*np.array(Sniter) - p*np.array(Smiter))%d
        z_vals = [i for i, e in enumerate(kk) if e == 0]
        pop_oper.append(z_vals)

#    print(pop_oper)
    pop_oper = np.unique(pop_oper, axis = 0)
#    print(pop_oper)
#    print(pop_oper[:,1:])
    pop_oper = pop_oper[:,1:]
    Sniter = np.delete(Sniter, pop_oper, axis = None)
    Smiter = np.delete(Smiter, pop_oper, axis = None)
    #print("Final Sniter, Smiter:", Sniter, Smiter)
    eigRel = np.zeros((len(Sniter),d**2))
    for kk in range(len(Sniter)):
        appOp = 0;
        for nn in range(d):
            for mm in range(d):
                eigRel[kk,appOp] = (Smiter[kk]*nn - Sniter[kk]*mm)%d#(mm*g - nn*t)%d
                appOp +=1
    eigRel = {(Sniter[ii], Smiter[ii]): list(eigRel[ii]) for ii in range(len(Sniter))}
#    print(eigRel)
    return(eigRel)

def meas_tom(J, v, N, pp):
    """meas_tom(J, v, N) returns the ideal and actual measurement frequencis. J is the channel Choi state,
    v is the measurement basis (first element will be channel input) repeated over N trials."""
    rho_in = qt.ket2dm(qt.Qobj(v[:,0]))
    d = len(v)
    rho_out = apply_map((1 - pp)*rho_in + pp*np.eye(d)/d, J)
    ideal = np.diag(v.getH()*np.matrix(rho_out)*v)
    ideal = ideal.real
    ideal.setflags(write=1)
    ideal[ideal<0] = 0
    ideal[ideal>1] = 1
    #print(r_ideal)
    #ideal = ideal/np.sum(ideal)
    freq = np.random.multinomial(N, ideal)
    #print(freq/N)

    return ideal, freq/N

def is_prime(n):
    '''check if integer n is a prime'''
    # make sure n is a positive integer
    n = abs(int(n))
    # 0 and 1 are not primes
    if n < 2:
        return False
    # 2 is the only even prime number
    if n == 2:
        return True
    # all other even numbers are not primes
    if not n & 1:
        return False
    # range starts with 3 and only needs to go up the squareroot of n
    # for all odd numbers
    for x in range(3, int(n**0.5)+1, 2):
        if n % x == 0:
            return False
    return True



def get_choi_DWC_SDP(cv_P, weyl_oper):
    """Choi operator of DWC for semidefinite programming"""
    d = len(weyl_oper)
    mes = qt.qeye(d)
    mesP = qt.Qobj(np.zeros((d**2,1)))
    for ii in range(d):
        mesP += np.kron(qt.Qobj(mes[:,ii]), qt.Qobj(mes[:,ii]))
    mes_dm = qt.ket2dm(mesP)
    mes_dm.dims = [[d,d],[d,d]]
    #print(mes_dm)
    id_ch = qt.qeye(d) # ideal channel
    out_st = qt.Qobj(np.zeros((d**2,d**2)))
    for nn in range(d):
        for mm in range(d):
            out_st += cv_P[nn][mm]*np.kron(id_ch, np.matrix(weyl_oper[nn][mm]))*mes_dm*np.kron(id_ch, np.matrix(weyl_oper[nn][mm].dag()))
    return out_st


def Hol_cap_DWC(Pxy):
    """Returns upper and lower bounds on the Holevo capacity of a DWC. Currently only for prime d."""
    # Lower bound
    d = len(Pxy)
    Pxy = Pxy.real
    Pxy[Pxy < 1e-16] = 1e-16
    eig_Rel = get_eig_relations(d)
    out_rels = list(eig_Rel.values()) # corresponding output relations
    rel_matrix = np.matrix([np.array(row)==kk for row in out_rels for kk in range(d)], dtype = int)
    Pxy_v = Pxy.reshape((1,d**2))
    min_ent = min(entropy((rel_matrix*Pxy_v.T).reshape((d+1,d)), base = 2, axis = 1))
    # Upper bound
    Pxy_so = Pxy.reshape((1,d**2))
    Pxy_so.sort()
    min_min_ent = entropy(np.add.reduceat(Pxy_so[0], np.arange(0, len(Pxy_so[0]), d)), base = 2)
    return log(d,2)-min_ent, log(d,2)-min_min_ent
######################################
######################################
######################################
######################################
######################################
######################################
def column(matrix, i):
    return [row[i] for row in matrix]

#####################################
def rep_code(k, p, N):
    '''k-repetition code for transmitting N symbols
    over symmetric with error-symbol probabilities
    given by vector p.'''
    d = len(p)
    # since channel is symmetric, we can fix input symbol to be 0 w.l.o.g., but we will not do
    # so that finite sample effects do not disappear
    mess = np.random.choice(np.arange(d), N)
    enc = np.matlib.repmat(mess, k, 1).T
    err = np.reshape(np.random.choice(np.arange(d), k*N, p=p), [k, N]).T
    rx = np.mod(enc + err, d)
    maj_vote = [Counter(m).most_common() for m in rx]
    dec = np.array([c[0][0] for c in maj_vote])
    symb_err_rate = (N - np.sum(mess == dec))/N
    #print(mess,"\n", enc, "\n\n", err, "\n\n", rx, "\n\n", maj_vote, "\n\n", dec, "\n\n", symb_err_rate)
    #print(symb_err_rate)
    ### Now estimating the probabilities
    dec_rep = np.matlib.repmat(dec, k, 1).T
    err_est = np.mod(rx - dec_rep, d)
    #print(Counter(err_est.reshape([1, k*N])[0]).most_common())
    err_counts = Counter(err_est.reshape([1, k*N])[0]).most_common()
    est_prob = np.zeros(d)
    for kk in err_counts:
        est_prob[kk[0]] = kk[1]/(k*N)
    #print(p, "\n\n", est_prob)
    return(symb_err_rate, est_prob)
