'''
This file generates data for a complexity analysis of the Metroplis-Hastings and Wolff algorithms for the 2D Ising model.
'''
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import ising_class

# Parameters
N_values = [64, 128, 256, 512, 1024, 2048]
a = 0.44
b = 0
algorithms = ['metropolis_hastings', 'wolff']
steps = 500

# Create file to store data
path = os.path.join(os.getcwd(), 'Ising', 'Data', 'ising_complexity.hdf5')
f = h5py.File(path, 'w')

# Run simulations and record runtimes
for algorithm in algorithms:
    runtimes = []
    for N in N_values:
        # Run simulation
        start_time = time.time()
        for i in range(3):
            ising = ising_class.Ising(N, a, b, alignment='random')
            ising.evolve(algorithm=algorithm, steps=steps)
        end_time = time.time()
            
        # Calculate runtime
        runtime = end_time - start_time
        print(runtime/3)
        runtimes.append(runtime/3)

    # Store runtimes in the HDF5 file
    f.create_dataset(f'runtimes_{algorithm}', data=runtimes)
