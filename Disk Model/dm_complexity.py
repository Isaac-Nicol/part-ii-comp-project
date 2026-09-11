'''
This file generates data for a complexity analysis of the generalized Metroplis-Hastings and 
generalized cluster algorithms for the hard disk model.
'''
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import dm_class

# Parameters
N_values = [20, 40, 60, 80, 100, 120, 140]
L = 25
algorithms = ['generalized_mh', 'generalized_cluster']
steps = 500

# Create file to store data
path = os.path.join(os.getcwd(), 'Disk Model', 'Data', 'dm_complexity.hdf5')
f = h5py.File(path, 'w')

# Run simulations and record runtimes
for algorithm in algorithms:
    runtimes = []
    for N in N_values:
        # Run simulation
        start_time = time.time()
        for i in range(3):
            dm = dm_class.DiskModel(N, L)
            dm.evolve(algorithm=algorithm, steps=steps)
        end_time = time.time()

        # Calculate runtime
        runtime = end_time - start_time
        runtimes.append(runtime/3)

    # Store runtimes in the HDF5 file
    f.create_dataset(f'runtimes_{algorithm}', data=runtimes)
