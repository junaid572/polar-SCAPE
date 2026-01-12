# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 17:38:44 2026

@author: junaid
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize_scalar

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


# ==================== Utility Functions ====================
def shannon_entropy(p):
    """Compute Shannon entropy H(p) in bits."""
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))

def H2(q):
    """Binary entropy function H₂(q)."""
    return shannon_entropy([q, 1 - q])

# ==================== Estimation Error Bounds ====================
def I_gamma(gamma):
    """Fisher Information for Pauli-twirled AD channel. (From Mathematica)"""
    return (4 - 3*gamma)/(4*gamma - 6*gamma**2 + 2*gamma**3)

def I_ad_channel(gamma):
    """Fisher Information for Bernoulli (AD channel)."""
    return 1/(gamma * (1 - gamma))

# ==================== Communication Rate/Capacity ====================
def ad_holevo_info(gamma):
    """Holevo information for AD channel.
    Equation after (28) of 
    https://journals-aps-org.proxy.bnl.lu/pra/pdf/10.1103/PhysRevA.97.012332
    proved in 
    PhysRevA.71.032314
    """
    # Convert to array for consistent handling
    gamma_array = np.atleast_1d(gamma)
    results = []
    
    for g in gamma_array:
        def capacity_bound(p):
            p_val = float(p)
            term1 = H2((1 - g) * p_val)
            inside_sqrt = 1.0 - 4.0 * g * (1 - g) * (p_val ** 2)
            inside_sqrt = max(inside_sqrt, 0.0)  # OK for scalar
            q = (1.0 + np.sqrt(inside_sqrt)) / 2.0
            term2 = H2(q)
            return term1 - term2
        
        # Optimize
        res = minimize_scalar(lambda p: -capacity_bound(p), bounds=(0, 1), method="bounded")
        results.append(capacity_bound(res.x))
    
    return results[0] if np.isscalar(gamma) else np.array(results)

def pauli_twirled_holevo_info(gamma):
    """Holevo information for Pauli-twirled AD channel."""
    # Convert to array for consistent handling
    gamma_array = np.atleast_1d(gamma)
    results = []
    
    for g in gamma_array:
        # Calculate probabilities for this specific gamma value
        # Reference: https://link.aps.org/doi/10.1103/PhysRevA.88.012314
        # eq. (20), lambda = 0
        pX = pY = g / 4.0
        pZ = (1.0 - 0.5 * g - np.sqrt(1.0 - g)) / 2.0
        pI = 1 - pX - pY - pZ
        
        # Holevo information = 1 - min(H2(pI + pX), H2(pI + pY), H2(pI + pZ))
        pauli_capacity = min(H2(pI + pX), H2(pI + pY), H2(pI + pZ))
        
        results.append(1 - pauli_capacity)
    
    return results[0] if np.isscalar(gamma) else np.array(results)

# ==================== Main Plot ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Generate gamma values
gamma_vals = np.linspace(0.001, 0.999, 200)

# --- Subplot 1: Cramér–Rao Bound ---
ax1.plot(
    gamma_vals, 1 / I_gamma(gamma_vals),
    'b-', linewidth=2.5,
    label=r'Pauli-twirled AD: $1/I(\gamma)$'
)
ax1.plot(
    gamma_vals, 1 / I_ad_channel(gamma_vals),
    'r--', linewidth=2,
    label=r'AD channel: $\gamma(1-\gamma)$'
)

ax1.set_xlabel(r'$\gamma$', fontsize=12)
ax1.set_ylabel(r'Cramér--Rao Bound', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.4)

# --- Subplot 2: Communication Rate (Holevo Information) ---
ax2.plot(
    gamma_vals, ad_holevo_info(gamma_vals),
    'r--', linewidth=2.5,
    label=r'AD channel'
)
ax2.plot(
    gamma_vals, pauli_twirled_holevo_info(gamma_vals),
    'b-', linewidth=2,
    label=r'Pauli-twirled AD'
)

ax2.set_xlabel(r'$\gamma$', fontsize=12)
ax2.set_ylabel(r'Holevo Information (bits)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# --- Common vertical marker ---
for ax in [ax1, ax2]:
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ymax = ax.get_ylim()[1]
    ax.text(
        0.52, 0.9 * ymax,
        r'$\gamma = 0.5$',
        fontsize=9, color='gray'
    )

plt.tight_layout()
plt.show()


print(30*"="+ " MORAL " + 30*"=", "\nIf there exists a SCAPE protocol that simultaneously achieves the Holevo inforamtion and CRB for amplitude damping channel, it will outpuerform the proposed polar SCAPE applied on the Pauli twirling-approximated ampltide damping channel.\n", 66*"=")
