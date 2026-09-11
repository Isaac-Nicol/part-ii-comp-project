'''
This file contains code to generate data used to calculate the autocorrelation time of energy with a range of 
values of the number N of disks at the same L for each of the algorithms.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import dm_class


# Parameters
L = 25
N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
algorithms = ['generalized_mh', 'generalized_cluster']
steps = 2000

# Create file to store data
path = os.path.join(os.getcwd(), 'Disk Model', 'Data', 'correlation_time_series.hdf5')
f = h5py.File(path, 'w')

# Run the simulation for both algorithms and different values of a
for algorithm in algorithms:
    for i,N in enumerate(N_values):
        dm = dm_class.DiskModel(N, L)
        dm.evolve(algorithm=algorithm, steps=steps, energy_history=True)
        f.create_dataset(f'sys_energy_{algorithm}_{N}', data=dm.energy_history)
