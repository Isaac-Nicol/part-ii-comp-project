'''
This file will contain definitions for the Ising class, with methods to initialize, perform iterations
of the Metropolis-Hastings & Wolff algorithms, and to calculate the magnetization, energy etc.
Plus whatever else I, the God of all Atoms, decide to add.
'''


from collections import deque
import numpy as np
import matplotlib.pyplot as plt


# Define the Ising class, which contains the lattice as well as methods to compute physical quantities and perform iterations of each algorithm
class Ising:
    '''
    Class to simulate the Ising model in 2D using periodic boundary conditions. The object is created with parameters N, a, b:
    N: the lattice size (N x N)
    a: the dimensionless spin-spin coupling constant
    b: the dimensionless spin-field coupling constant
    The lattice is initialized with random spins by default, but can also be initialized with all spins
    up or down.
    The evolve method is used to evolve the lattice using the specified algorithm for the specified number of steps. It takes arguments:
    algorithm: 'metropolis_hastings' or 'wolff' (case-sensitive) - specifies the algorithm to use
    steps: the number of iterations to perform
    (optional) history: boolean, whether to store the magnetization and energy at each step
    '''
    def __init__(self, N_, a_, b_, alignment='random'):
        # Parameters
        self.N = N_
        self.a = a_
        self.b = b_
        
        # Lattice initialization
        if alignment == 'random':    
            self.lattice = np.random.choice([-1, 1], size=(N_, N_)).astype(np.int64)
        elif alignment == 'up':
            self.lattice = np.ones((N_, N_)).astype(np.int64)
        elif alignment == 'down':
            self.lattice = -np.ones((N_, N_)).astype(np.int64)
        
        # General attributes
        self.rel_mag_history = np.array([])
        self.sys_energy_history = np.array([])
        self.update_magnetization()
        self.update_energy()

        # Metropolis-Hastings attributes
        self.mh_lookup = np.exp(np.array([0, 0, -4, 0, -8, 8, 0, 4, 0]) * self.a) # lookup table for Metropolis-Hastings algorithm
        # Boolean masks to allow Metroplis-Hastings algorithm to be vectorized
        self.checkerboard = np.indices((N_, N_)).sum(axis=0) % 2
        self.opposite_checkerboard = np.logical_not(self.checkerboard)

        # Wolff attributes
        self.wolff_padd = 1 - np.exp(-2 * self.a)

    # General method definitions ########################################################################
    def update_magnetization(self):
        '''
        Method to compute the magnetization and relative magnetization of the lattice, and append the
        relative magnetization to the rel_mag_history array.
        '''
        self.magnetization = np.sum(self.lattice)
        self.rel_mag = self.magnetization / (self.N**2)
        self.rel_mag_history = np.append(self.rel_mag_history, self.rel_mag)

    def sum_neighbours(self):
        '''
        Method to compute the sum of the spins of the nearest neighbours of each site in the lattice.
        '''
        return np.multiply(self.lattice, (np.roll(self.lattice, 1, axis=0) + np.roll(self.lattice, -1, axis=0) 
                               + np.roll(self.lattice, 1, axis=1) + np.roll(self.lattice, -1, axis=1)))
    
    def update_energy(self):
        '''
        Method to compute the energy of the lattice, and append the energy to the sys_energy_history array.
        '''
        self.energy = -self.a * 0.5 * np.sum(self.sum_neighbours()) - self.b * np.sum(self.lattice)
        self.sys_energy_history = np.append(self.sys_energy_history, self.energy)

    def modify_params(self, a_, b_):
        '''
        Method to modify the parameters of the Ising model. Doing this is useful for sweeping through a
        range of parameters, because the lattice will come to equilibrium faster if it starts from the
        previous equilibrium state.
        '''
        self.a = a_
        self.b = b_
        self.mh_lookup = np.exp(np.array([0, 0, -4, 0, -8, 8, 0, 4, 0]) * self.a)
        self.wolff_padd = 1 - np.exp(-2 * self.a)

    def get_partial_lattices(self):
        '''
        Method to return the two partial lattices used in the Metropolis-Hastings algorithm.
        '''
        return [self.lattice * self.checkerboard, self.lattice * self.opposite_checkerboard]
    ####################################################################################################

    # Metropolis-Hastings algorithm ####################################################################
    def metropolis_hastings_no_field(self):
        '''
        Method to perform the Metropolis-Hastings algorithm on the lattice. This method is optimized for
        the case where the spin-field coupling constant b = 0, and is only appropriate for that case.
        '''
        p = np.random.rand(self.N, self.N)

        partial_lattices = self.get_partial_lattices()
        partial_lattices[0][np.where(self.mh_lookup[self.sum_neighbours()] > p)] *= -1
        self.lattice = partial_lattices[0] + partial_lattices[1]
        partial_lattices = self.get_partial_lattices()
        partial_lattices[1][np.where(self.mh_lookup[self.sum_neighbours()] > p)] *= -1
        self.lattice = partial_lattices[0] + partial_lattices[1]
    ####################################################################################################

    # Wolff algorithm ##################################################################################
    def wolff(self):
        '''
        Method to perform the Wolff algorithm on the lattice. This method is optimized for the case where
        the spin-field coupling constant b = 0, and is only appropriate for that case. Note also that it
        doesn't work well for the antiferromagnetic case (a < 0).
        '''
        # Generate seed spin
        seed = np.random.randint(self.N, size=2)
        spin = self.lattice[seed[0], seed[1]]
        self.lattice[seed[0], seed[1]] = -spin
        unvisited = deque([seed])
        while unvisited:   # while unvisited sites remain
            site = unvisited.pop()  # take one and remove from the unvisited list
            neighbouring_sites = [((site[0]+1)%self.N,site[1]),((site[0]-1)%self.N,site[1]),
                                  (site[0],(site[1]+1)%self.N),(site[0],(site[1]-1)%self.N)]
            for nbr in neighbouring_sites:
                if self.lattice[nbr[0], nbr[1]] == spin and np.random.random() < self.wolff_padd:
                    self.lattice[nbr[0], nbr[1]] = -spin
                    unvisited.appendleft(nbr)
    ####################################################################################################

    def evolve(self, algorithm, steps, history=False, spacing=1):
        '''
        Method to evolve the lattice using the specified algorithm for the specified number of steps.
        Parameters:
        algorithm: 'metropolis_hastings' or 'wolff' (case-sensitive) - specifies the algorithm to use
        steps: the number of iterations to perform
        (optional) history: boolean, whether to store the magnetization and energy at each step
        (optional) spacing: the number of steps between each recorded magnetization and energy
        '''
        if algorithm == 'metropolis_hastings' and self.b == 0:
            for _ in range(steps):
                self.metropolis_hastings_no_field()
                self.update_magnetization() if (history and _ % spacing == 0) else None
        if algorithm == 'wolff':
            for _ in range(steps):
                self.wolff()
                self.update_magnetization() if (history and _ % spacing == 0) else None
