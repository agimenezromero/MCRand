#Tests and examples for the Monte Carlo Random Number Generator library (MCRand)

import numpy as np
import matplotlib.pyplot as plt
import time

from MCRand import RandGen as rg


def gaussian(x, sigma, mu):
	return (1/(np.sqrt(2*np.pi*sigma**2))) * np.exp(-(x-mu)**2/(2*sigma**2))

current_milli_time = lambda: time.time() * 1000

#####################################################################
#																	#
#								TESTS								#
#																	#
#####################################################################

x0 = -5
xf = 5
N = 1000
sigma = 1
mu = 0

print('---------- TEST ----------\n')

print('sigma=%.2f, mu=%.2f\n' % (sigma, mu))

t0 = current_milli_time()
numpy_rand = np.random.normal(mu, sigma, N)
tf = current_milli_time()

print('Numpy gaussian random generator mean: ', np.mean(numpy_rand))
print('Elapsed time: %.4fms' % (tf-t0, ))


t0 = current_milli_time()	
rand = rg.sample(gaussian, x0, xf, N, sigma, mu)
tf = current_milli_time()

print('\nMC gaussian random generator mean: ', np.mean(rand))
print('Elapsed time: %.4fms' % (tf-t0, ))

x = np.linspace(x0, xf, N)

plt.ylabel('Probability density function')
plt.hist(numpy_rand, density=True, color=(0,0,1,0.8), label='NumPy')
plt.hist(rand, density=True, color=(0,1,0,0.5), label='MCRand')
plt.plot(x, gaussian(x, sigma, mu), color='r', label='Gaussian distribution')

plt.legend()
plt.show()