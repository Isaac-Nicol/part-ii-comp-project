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
        
    def generate_configuration(self):
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
    
    def valid_move(self, disk, new_disk):
        '''
        Check if a move is valid
        '''
        for i in range(self.N):
            if i != disk and self.check_overlap(new_disk, self.configuration[i]):
                return False
        return True
    
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
        
        if self.valid_move(disk, new_disk):
            self.configuration[disk] = new_disk
            return
        return
    
    def evolve(self, algorithm='metropolis_hastings', steps=100):
        '''
        Perform the Metropolis-Hastings algorithm for a number of steps
        '''
        if algorithm == 'metropolis_hastings':
            for _ in range(steps):
                self.metropolis_hastings()


instance = DiskModel(50, 100)
plot_disk_configuration(instance.configuration, instance.L)
instance.evolve(algorithm='metropolis_hastings', steps=1000)
plot_disk_configuration(instance.configuration, instance.L)
