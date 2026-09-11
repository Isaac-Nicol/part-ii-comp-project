'''
This file will compute the correlation and autocorrelation times for the 2D Ising model using the data generated in
critical_slowing_down.py.
'''
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.append(os.getcwd())
from utils import autocorrelation, get_autocorrelation_time


# Parameters
N = 128
a_values = [0.1, 0.3, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.5, 0.6, 0.8, 1]
a_values.reverse() # This helps the Wolff algorithm to converge for small a; it doesn't work so well when clusters can't grow very large
b = 0
algorithms = ['metropolis_hastings', 'wolff']
steps = 2000
t_max = 750
burn_in = 500

# Load data
path = os.path.join(os.getcwd(), 'Ising', 'Data', 'critical_slowing_down.hdf5')
f = h5py.File(path, 'r')

# Compute autocorrelation times
autocorr_times = {}
for algorithm in algorithms:
    for a in a_values:
        data = f[f'rel_mag_{algorithm}_{a}'][:]
        autocorr = autocorrelation(data, t_max=t_max, burn_in=burn_in)
        autocorr_times[f'{algorithm}_{a}'] = get_autocorrelation_time(autocorr)

# Save path
save_path = os.path.join(os.getcwd(), 'Ising', 'Figures')

# Plot autocorrelation times vs a
for algorithm in algorithms:
    plt.plot(a_values, [autocorr_times[f'{algorithm}_{a}'] for a in a_values], label=algorithm)
plt.axvline(x=0.44, color='black', linestyle='--', label='Onsager critical point')
plt.xlabel('a')
plt.ylabel('Autocorrelation time/iterations')
plt.title('Autocorrelation time vs a')
plt.legend()
plt.xlim(0.1, 1)
plt.savefig(os.path.join(save_path, f'critical_slowing_down.png'), dpi=300, bbox_inches='tight')
plt.show()
