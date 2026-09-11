'''
This file will generate both the data and the plots for the demonstration of the correlation and 
autocorrelation time calculations for the magnetization of the 2D Ising model.
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils import autocorrelation, get_autocorrelation_time
import ising_class

# Path to save plot
path = os.path.join(os.getcwd(), 'Ising', 'Figures', 'magnetization_autocorrelation.png')

# Parameters
N = 128
a = 0.5
b = 0
steps = 10000
burn_in = 2000
t_max = 100

# Create and evolve the Ising model
ising = ising_class.Ising(N, a, b, alignment='random')
ising.evolve(steps=steps, algorithm='metropolis_hastings', magnetization_history=True)

# Calculate the autocorrelation and autocorrelation time
autocorr = autocorrelation(np.abs(ising.rel_mag_history), t_max=t_max, burn_in=burn_in)
autocorr_time = get_autocorrelation_time(autocorr)

# Plot the magnetization and autocorrelation
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes[0].plot(np.abs(ising.rel_mag_history))
axes[0].axvline(burn_in, color='r', linestyle='--', label='Burn-in')
axes[0].set_xlim(0, steps)
axes[0].set_ylim(0, 1)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Absolute magnetization')
axes[0].set_title('Absolute magnetization vs. step')
axes[0].legend()
axes[1].plot(autocorr)
axes[1].set_xlim(0, t_max)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel('Lag/steps')
axes[1].set_ylabel('Correlation of absolute magnetization')
axes[1].set_title('Correlation of absolute magnetization vs. lag')
axes[1].axvline(autocorr_time, color='r', linestyle='--', label=f'Autocorrelation time = {autocorr_time}')
axes[1].plot(np.arange(len(autocorr)), np.exp(-np.arange(len(autocorr))/autocorr_time), label='Exponential decay', color='k', linestyle='--')
axes[1].legend()
plt.tight_layout()
plt.savefig(path, dpi=300, bbox_inches='tight')
plt.show()
