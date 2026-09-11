'''
This file performs preliminary tests of the Metropolis-Hastings and cluster algorithms for the hard disk model
to ensure that they are functioning as expected. 
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import dm_class


# Create file to store data
path = os.path.join(os.getcwd(), 'Disk Model', 'Data', 'disk_test.hdf5')
f = h5py.File(path, 'w')

# Set parameters
N = 100
L = 100
algorithms = ['metropolis_hastings', 'cluster']
config_samples = np.zeros((6, N, 2))

# Run the simulation for both algorithms
for algorithm in algorithms:
    dm = dm_class.DiskModel(N, L)
    config_samples[0, :] = dm.configuration
    dm.evolve(algorithm=algorithm, steps=1)
    config_samples[1, :] = dm.configuration
    dm.evolve(algorithm=algorithm, steps=9)
    config_samples[2, :] = dm.configuration
    dm.evolve(algorithm=algorithm, steps=90)
    config_samples[3, :] = dm.configuration
    dm.evolve(algorithm=algorithm, steps=900)
    config_samples[4, :] = dm.configuration
    dm.evolve(algorithm=algorithm, steps=9000)
    config_samples[5, :] = dm.configuration
    f.create_dataset(f'config_{algorithm}', data=config_samples)
