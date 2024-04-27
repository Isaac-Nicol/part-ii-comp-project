'''
This script generates a plot of a bimodal distribution. The distribution is a mixture of two Gaussian 
distributions with different means and standard deviations. The plot shows the probability density 
function of the distribution. The plot is saved as a PNG file in the same directory as the script.
This code is largely taken from https://hef.ru.nl/~tbudd/mct/lectures/mcmc_in_practice.html
'''
import os
import numpy as np  
import matplotlib.pylab as plt


def gaussian(mu,sigma,x):
    return np.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))/(sigma*np.sqrt(2*np.pi))

def distribution_pi(x):
    return 0.8*gaussian(0,1,x) + 0.2*gaussian(4,0.5,x)


xrange = np.linspace(-5,10,300)
plt.plot(xrange,distribution_pi(xrange))
plt.title(r"Probability density $\pi(x)$")
plt.xlabel(r"$x$")
plt.ylabel(r"$\pi(x)$")
path = os.path.join(os.getcwd(), 'General Figures', 'bimodal.png')
plt.savefig(path)
plt.show()
