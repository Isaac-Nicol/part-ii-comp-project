'''
This file will generate data used to calculate the autocorrelation time of with a range of values of the
parameter a for each of the algorithms.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import ising_class


# Parameters
N = 128
a_values = [0.1, 0.3, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.5, 0.6, 0.8, 1]
a_values.reverse() # This helps the Wolff algorithm to converge for small a; it doesn't work so well when clusters can't grow very large
b = 0
algorithms = ['metropolis_hastings', 'wolff']
steps = 2000

# Create file to store data
path = os.path.join(os.getcwd(), 'Ising', 'Data', 'critical_slowing_down.hdf5')
f = h5py.File(path, 'w')

# Run the simulation for both algorithms and different values of a
for algorithm in algorithms:
    ising = ising_class.Ising(N, 0, b, alignment='up')
    for i, a in enumerate(a_values):
        ising.modify_params(a, b)
        ising.evolve(algorithm=algorithm, steps=steps, magnetization_history=True)
        f.create_dataset(f'rel_mag_{algorithm}_{a}', data=np.abs(ising.rel_mag_history[steps*i:steps*(i+1)]))
