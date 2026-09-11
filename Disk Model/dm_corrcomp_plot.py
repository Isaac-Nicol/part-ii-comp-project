'''
This file calculates the correlation function and correlation times for the disk model using the data generated in
dm_correlation_comparison.py. It then plots the correlation times for the two algorithms against the number of disks N
and saves the figures.
'''
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.append(os.getcwd())
from utils import autocorrelation, get_autocorrelation_time


# Parameters
L = 25
N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
algorithms = ['generalized_mh', 'generalized_cluster']
steps = 2000
t_max = 1500
burn_in = 100

# Load data
path = os.path.join(os.getcwd(), 'Disk Model', 'Data', 'correlation_time_series.hdf5')
f = h5py.File(path, 'r')

# Compute autocorrelation times
autocorr_times = {}
for algorithm in algorithms:
    for N in N_values:
        data = f[f'sys_energy_{algorithm}_{N}'][:]
        autocorr = autocorrelation(data, t_max=t_max, burn_in=burn_in)
        autocorr_times[f'{algorithm}_{N}'] = get_autocorrelation_time(autocorr)

# Save path
save_path = os.path.join(os.getcwd(), 'Disk Model', 'Figures')

# Plot autocorrelation times vs a
for algorithm in algorithms:
    plt.plot(N_values, [autocorr_times[f'{algorithm}_{N}'] for N in N_values], label=algorithm)
plt.xlabel('Number of disks N')
plt.ylabel('Autocorrelation time/iterations')
plt.title('Autocorrelation time vs N')
plt.legend()
plt.savefig(os.path.join(save_path, f'dm_correlation_comparision.png'), dpi=300, bbox_inches='tight')
plt.show()
