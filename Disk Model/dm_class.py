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
    def __init__(self,N,L):
        self.N = N
        self.L = L
        self.configuration = self.generate_configuration()
        self.occupation = self.initialize_occupation_array()
        
    def generate_configuration(self):
        '''
        This generates an evenly spaced grid pattern and then fills it with disks up to self.N
        '''
        # The configuration is only possible if the number of disks is less than a certain threshold
        if self.N > np.ceil(self.L/2 - 1)**2:
            self.N = np.ceil(self.L/2 - 1)**2  

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

    def check_overlap(self, disk1, disk2):
        '''
        Compute the minimal distance between the two disks on the torus and check if they overlap
        '''
        x1, y1 = disk1
        x2, y2 = disk2
        dx = np.abs(x1 - x2)
        dy = np.abs(y1 - y2)
        dx = min(dx, self.L - dx)
        dy = min(dy, self.L - dy)
        return np.sqrt(dx**2 + dy**2) < 2
    
    def get_overlap(self, point):
        '''Return a list of indices of disks that have overlap with point.'''
        overlaps = []
        for dx in range(-2,3):
            x = (int(point[0]) + dx) % self.L
            for dy in range(-2,3):
                y = (int(point[1]) + dy) % self.L
                index = self.occupation[x,y]
                if index >= 0 and (((point[0]%1)-(self.configuration[index,0]%1)-dx)**2 +
                                ((point[1]%1)-(self.configuration[index,1]%1)-dy)**2 < 4):
                    overlaps.append(index)
        return overlaps
    
    def valid_move(self, new_disk):
        '''
        Check if a move is valid
        '''
        overlaps = self.get_overlap(new_disk)
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

    def metropolis_hastings(self, delta=1):
        '''
        Perform the Metropolis-Hastings algorithm to move a disk
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
            overlap = self.get_overlap(self.configuration[mover])
            for i in overlap:
                movers.appendleft(i)
                self.clear_disk(i)
            self.add_disk(mover)

    def evolve(self, algorithm='metropolis_hastings', steps=100):
        '''
        Perform the Metropolis-Hastings algorithm for a number of steps
        '''
        if algorithm == 'metropolis_hastings':
            for _ in range(steps):
                self.metropolis_hastings()
        elif algorithm == 'cluster':
            for _ in range(steps):
                self.disk_cluster_move()


instance = DiskModel(500, 100)
plot_disk_configuration(instance.configuration, instance.L)
instance.evolve(algorithm='cluster', steps=1000)
plot_disk_configuration(instance.configuration, instance.L)
