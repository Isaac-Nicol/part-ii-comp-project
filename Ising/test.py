import numpy as np
import matplotlib.pyplot as plt
from ising_class import Ising


# N = 128
# a_values = [-0.2, 0.6]
# b = 0
# lattice_samples = np.zeros((len(a_values), 6, N, N))

# # Run the simulation for different values of b
# for i, a in enumerate(a_values):
#     ising = Ising(N, a, b, alignment='up')
#     lattice_samples[i, 0, :, :] = ising.lattice
#     ising.evolve(algorithm='metropolis_hastings', steps=1)
#     lattice_samples[i, 1, :, :] = ising.lattice
#     ising.evolve(algorithm='metropolis_hastings', steps=9)
#     lattice_samples[i, 2, :, :] = ising.lattice
#     ising.evolve(algorithm='metropolis_hastings', steps=90)
#     lattice_samples[i, 3, :, :] = ising.lattice
#     ising.evolve(algorithm='metropolis_hastings', steps=900)
#     lattice_samples[i, 4, :, :] = ising.lattice
#     ising.evolve(algorithm='metropolis_hastings', steps=9000)
#     lattice_samples[i, 5, :, :] = ising.lattice

# # Plot the lattice for each value of b at each of the sampled times
# fig, axes = plt.subplots(len(a_values), 6, figsize=(15, 6))
# for i in range(len(a_values)):
#     for j, t in enumerate([0, 1, 10, 100, 1000, 10000]):
#         axes[i, j].imshow(lattice_samples[i, j, :, :], cmap='gray')
#         axes[i, j].set_xticks([])
#         axes[i, j].set_yticks([])
#         axes[i, j].set_title(f'a = {a_values[i]}, t = {t}')
# plt.suptitle('Evolution of the lattice for different values of a and upward initial configuration', y=0.92)
# plt.show()

# Set initial parameters
N = 128
a_values = [0.1, 0.3, 0.4, 0.43, 0.45, 0.5, 0.6, 0.8, 1]
b = 0
rel_mag_array = np.array([])

# Run the simulation for different values of a
ising = Ising(N, 0, b, alignment='up')
for a in a_values:
    ising.modify_params(a, b)
    ising.evolve(algorithm='metropolis_hastings', steps=2000)
    ising.update_magnetization()
    rel_mag_array = np.append(rel_mag_array, ising.rel_mag)

# Theoretical prediction
x_0 = np.linspace(0, 0.441, 1000)
x_1 = np.linspace(0.441, 1, 1000)
supercritical = np.zeros(1000)
subcritical_pos = (1 - (np.sinh(2*x_1))**(-4)) ** (1/8)
x = np.append(x_0, x_1)
predicted_rel_mag_pos = np.append(supercritical, subcritical_pos)
predicted_rel_mag_neg = -predicted_rel_mag_pos

plt.plot(x, predicted_rel_mag_pos, '-k', label='Theoretical prediction')
plt.plot(x, predicted_rel_mag_neg, '-k')
plt.plot(a_values, rel_mag_array, 'or', label='Simulated values')
plt.legend()
plt.xlabel('a')
plt.ylabel('Relative magnetisation')
plt.title('Relative magnetisation vs a')
plt.show()