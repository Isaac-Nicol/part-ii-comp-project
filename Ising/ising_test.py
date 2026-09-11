'''
This file performs preliminary tests of both the Metropolis-Hastings and Wolff algorithms for the 2D Ising model
to ensure that they reproduce the expected physics.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from ising_class import Ising


# Create file to store data
path = os.path.join(os.getcwd(), 'Ising', 'Data', 'ising_test.hdf5')
f = h5py.File(path, 'w')

# Set parameter values
N = 128
a_values = [0.4, 0.6]
b = 0
algorithms = ['metropolis_hastings', 'wolff']
lattice_samples = np.zeros((5, N, N))

# Run the simulation for both algorithms and different values of a
for algorithm in algorithms:
    for i, a in enumerate(a_values):
        ising = Ising(N, a, b, alignment='up')
        lattice_samples[0, :, :] = ising.lattice
        ising.evolve(algorithm=algorithm, steps=1)
        lattice_samples[1, :, :] = ising.lattice
        ising.evolve(algorithm=algorithm, steps=9)
        lattice_samples[2, :, :] = ising.lattice
        ising.evolve(algorithm=algorithm, steps=90)
        lattice_samples[3, :, :] = ising.lattice
        ising.evolve(algorithm=algorithm, steps=900)
        lattice_samples[4, :, :] = ising.lattice
        f.create_dataset(f'lattice_{algorithm}_{a}', data=lattice_samples)

# Parameters
a_values = [0.1, 0.3, 0.4, 0.43, 0.45, 0.5, 0.6, 0.8, 1]
a_values.reverse() # This helps the Wolff algorithm to converge for small a; it doesn't work so well when clusters can't grow very large

ising = Ising(N, 0, b, alignment='up')
for algorithm in algorithms:
    rel_mag_array = np.array([])
    for a in a_values:
        ising.modify_params(a, b)
        ising.evolve(algorithm=algorithm, steps=200)
        ising.update_magnetization()
        rel_mag_array = np.append(rel_mag_array, np.abs(ising.rel_mag))
    f.create_dataset(f'rel_mag_{algorithm}', data=rel_mag_array)
