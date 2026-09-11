'''
This file will contain definitions for the DiskModel class, with methods to initialize, perform iterations
of the Metropolis-Hastings & cluster algorithms, and to calculate the potential energy (where applicable) 
of the system.
'''

from collections import deque
import numpy as np
import matplotlib.pyplot as plt


def plot_disk_configuration(positions,L):
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_ylim(0,L)
    ax.set_xlim(0,L)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    for x,y in positions:
        # consider all horizontal and vertical copies that may be visible
        for x_shift in [z for z in x + [-L,0,L] if -1<z<L+1]:
            for y_shift in [z for z in y + [-L,0,L] if -1<z<L+1]:
                ax.add_patch(plt.Circle((x_shift,y_shift),1))
    plt.show()


# Disk model class
class DiskModel:
    '''
    Class to simulate the hard-disk model of a fluid. The object is created with parameters N, L:
    N: the number of disks
    L: the system size (L x L)
    The configuration is initialized with a square grid pattern which is filled up to N disks, assuming
    that N disks can fit in the system. Otherwise, the number of disks is reduced to the maximum possible
    value: ceil(L/2 - 1)^2.
    The evolve method is used to evolve the system using the specified algorithm for the specified number of steps. It takes arguments:
    algorithm: 'metropolis_hastings', 'cluster', 'generalized_mh' or 'generalized_cluster' - specifies the algorithm to use
    steps: the number of iterations to perform
    (optional) energy_history: boolean, whether to store a record of the system energy (default False)
    (optional) spacing: the number of steps between each recorded energy (default 1)
    '''
    def __init__(self,N,L):
        self.N = N
        self.L = L
        
        self.configuration = self.generate_configuration()
        self.occupation = self.initialize_occupation_array()
        self.energy_history = np.array([])
        
    def generate_configuration(self):
        '''
        This generates an evenly spaced grid pattern and then fills it with disks up to self.N
        '''
        # The configuration is only possible if the number of disks is less than a certain threshold
        if self.N > int(self.L/2 - 1)**2:
            self.N = int(self.L/2 - 1)**2 

        # Number of rows and columns
        rows = np.ceil(np.sqrt(self.N)).astype(np.int32)
        cols = np.ceil(self.N/rows).astype(np.int32)
        positions = np.empty((0,2))
        
        # Arrays of possible positions
        x = np.linspace(0,self.L,cols+1)[:-1]
        y = np.linspace(0,self.L,rows+1)[:-1]
        
        filled = 0
        while filled < self.N:
            for i in range(rows):
                for j in range(cols):
                    if filled < self.N:
                        positions = np.append(positions,[[x[j],y[i]]], axis=0)
                        filled += 1
        return positions

    # General method definitions ########################################################################
    def get_distance(self, point1, point2):
        '''
        Compute the minimal distance between two points on the torus
        '''
        x1, y1 = point1
        x2, y2 = point2
        dx = np.abs(x1 - x2)
        dy = np.abs(y1 - y2)
        dx = min(dx, self.L - dx)
        dy = min(dy, self.L - dy)
        return np.sqrt(dx**2 + dy**2)
    
    def get_neighbours(self, point, extent=2, i=0):
        '''
        Return a list of indices of disks that are neighbours of point within a stated range. 
        
        Parameters:
        point: the point to check for neighbours
        extent: the range to check for neighbours - specifically, the maximum separation between points.
        i: the index of the occupation array to begin considering neighbours from. This is a slightly strange
            functionality, but is useful for calculating the total energy of the system.
        '''
        neighbours = []
        for dx in range(-2*extent,2*extent+1):
            x = (int(point[0]) + dx) % self.L
            for dy in range(-2*extent,2*extent+1):
                y = (int(point[1]) + dy) % self.L
                index = self.occupation[x,y]
                if index > i and self.get_distance(point, self.configuration[index]) < extent:
                    neighbours.append(index)
        return neighbours
    
    def valid_move(self, new_disk):
        '''
        Check if a move is valid by checking for overlaps with neighbours
        '''
        overlaps = self.get_neighbours(new_disk)
        if overlaps:
            return False
        return True
    
    def initialize_occupation_array(self):
        '''
        Create an LxL array with indices containing the disk number at that position, -1 if empty
        '''
        occupation = -np.ones((self.L, self.L), dtype='int')
        for i, (x, y) in enumerate(self.configuration):
            occupation[int(x),int(y)] = i
        return occupation
    
    def clear_disk(self, index):
        '''Remove disk from occupation array.'''
        self.occupation[int(self.configuration[index, 0]), int(self.configuration[index, 1])] = -1

    def add_disk(self, index):
        '''Add disk to occupation array.'''
        self.occupation[int(self.configuration[index, 0]),int(self.configuration[index, 1])] = index

    def calculate_potential(self, disk1, disk2):
        '''
        Calculate the potential energy between two disks according to a Yukawa potential
        '''
        r = self.get_distance(disk1, disk2)
        return 1/r * np.exp(-r) 

    def calculate_total_potential(self):
        '''
        Calculate the total potential energy of the system
        '''
        total_potential = 0
        for i in range(self.N):
            neighbours = self.get_neighbours(self.configuration[i], extent=3, i=i)
            total_potential += sum([self.calculate_potential(self.configuration[i], self.configuration[j]) for j in neighbours])
        return total_potential
    
    def disk_potential(self, disk):
        '''
        Calculate the potential energy of a disk with its neighbours
        '''
        neighbours = self.get_neighbours(self.configuration[disk], extent=3)
        return sum([self.calculate_potential(self.configuration[disk], self.configuration[i]) for i in neighbours])
    #####################################################################################################
    
    # Metropolis-Hastings method definitions ############################################################
    def metropolis_hastings(self, delta=1):
        '''
        Perform an iteration of the Metropolis-Hastings algorithm.
        
        The parameter delta is the width of the normal distribution from which the size of the move is drawn.
        '''
        # Choose a random disk
        disk = np.random.randint(self.N)
        x, y = self.configuration[disk]
                    
        # Move the disk by a random amount
        x_new = delta * (x + np.random.normal()) % self.L
        y_new = delta * (y + np.random.normal()) % self.L
        new_disk = np.array([x_new, y_new])
                    
        if self.valid_move(new_disk):
            self.clear_disk(disk)
            self.configuration[disk] = new_disk
            self.add_disk(disk)
            return
        return
    
    def metropolis_hastings_with_potential(self, delta=1):
        '''
        Perform an iteration of the Metropolis-Hastings algorithm to move a disk with short range potential.

        The parameter delta is the width of the normal distribution from which the size of the move is drawn.
        '''
        # Choose a random disk
        disk = np.random.randint(self.N)
        x, y = self.configuration[disk]
                    
        # Move the disk by a random amount
        x_new = delta * (x + np.random.normal()) % self.L
        y_new = delta * (y + np.random.normal()) % self.L
        new_disk = np.array([x_new, y_new])
                    
        if self.valid_move(new_disk):
            self.clear_disk(disk)
            
            # Check energy change of the move and accept where appropriate
            old_potential = self.disk_potential(disk)
            self.configuration[disk] = new_disk
            new_potential = self.disk_potential(disk)
            delta_potential = new_potential - old_potential
            if delta_potential <= 0 or np.random.uniform() < np.exp(-delta_potential):
                self.add_disk(disk)
                return
            else:
                self.configuration[disk] = [x, y]
                self.add_disk(disk)
                return
        return
    #####################################################################################################
    
    # Cluster method definitions ########################################################################
    def disk_cluster_move(self):
        '''Iteratively reflect disks in pivot, starting with index, 
        until no more overlaps occur.'''
        # Choose a random disk and pivot point
        index = np.random.randint(self.N)
        pivot = np.random.uniform(0, self.L, 2)
            
        movers = deque() # deque is an efficient way to manage the list of disks to move
        movers.appendleft(index)
        self.clear_disk(index)
        while movers:
            mover = movers.pop()
            self.configuration[mover] = (2*pivot - self.configuration[mover]) % self.L # reflect disk
                
            # Find overlaps and add them to the cluster
            overlap = self.get_neighbours(self.configuration[mover])
            for i in overlap:
                movers.appendleft(i)
                self.clear_disk(i)
                
            self.add_disk(mover)
        return
    
    def disk_cluster_move_with_potential(self):
        '''
        Perform one iteration of the cluster algorithm with a short range potential.
        '''
        # Choose a random disk and pivot point
        index = np.random.randint(self.N)
        pivot = np.random.uniform(0, self.L, 2)
        
        movers = deque() # deque is an efficient way to manage the list of disks to move
        cluster = [index]
        movers.appendleft(index)
        self.clear_disk(index)
        while movers:
            mover = movers.pop()
            new_disk = (2*pivot - self.configuration[mover]) % self.L # reflect disk
            
            # Find overlaps and add them to the cluster
            overlap = self.get_neighbours(new_disk)
            for i in overlap:
                movers.appendleft(i)
                cluster.append(i)
                self.clear_disk(i)
            
            # Check the energy interactions with nearby disks and add to cluster where appropriate
            old_neighbours = self.get_neighbours(self.configuration[mover], extent=3)
            new_neighbours = self.get_neighbours(new_disk, extent=3)
            interactions = set(old_neighbours + new_neighbours)
            for interaction in interactions:
                old_potential = self.calculate_potential(self.configuration[mover], self.configuration[interaction]) if interaction in old_neighbours else 0
                new_potential = self.calculate_potential(new_disk, self.configuration[interaction]) if interaction in new_neighbours else 0
                delta_potential = new_potential - old_potential
                if delta_potential >= 0 or np.random.uniform() < 1 - np.exp(-delta_potential):
                    movers.appendleft(interaction)
                    cluster.append(interaction)
                    self.clear_disk(interaction)
            
            self.configuration[mover] = new_disk
        for item in cluster:
            self.add_disk(item)
        return
    #####################################################################################################

    def evolve(self, algorithm='metropolis_hastings', steps=1000, energy_history=False, spacing=1):
        '''
        Perform the specified algorithm for the specified number of steps.

        Parameters:
        algorithm: 'metropolis_hastings', 'cluster', 'generalized_mh' or 'generalized_cluster' - the algorithm to use
        (optional) steps: the number of iterations to perform; default is 1000
        '''
        if algorithm == 'metropolis_hastings':
            for _ in range(steps):
                self.metropolis_hastings()
        elif algorithm == 'cluster':
            for _ in range(steps):
                self.disk_cluster_move()
        elif algorithm == 'generalized_mh':
            for _ in range(steps):
                self.metropolis_hastings_with_potential()
                self.energy_history = np.append(self.energy_history, self.calculate_total_potential()) if (energy_history and _ % spacing == 0) else None
        elif algorithm == 'generalized_cluster':
            for _ in range(steps):
                self.disk_cluster_move_with_potential()
                self.energy_history = np.append(self.energy_history, self.calculate_total_potential()) if (energy_history and _ % spacing == 0) else None
