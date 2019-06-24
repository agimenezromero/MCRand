import numpy as np
from random import choice


class RandGen(object):
	"""docstring for RandGen"""
	def __init__(self):
		pass
		
	@classmethod
	def distribution(cls, f, x0, xf, *args):

		#Get the maximum of the f probability function

		x = f(np.linspace(x0, xf, 1000), *args)

		maximum = np.amax(x)
		n = 0

		new_randoms = []

		while n < 10**4:

			#print(round(n/(10**4) * 100, 2), '%')

			random_number = np.random.uniform(x0, xf)
			dice = np.random.uniform(0, maximum)

			if f(random_number, *args) > dice:

				new_randoms.append(random_number)

				n += 1

		return new_randoms

	@classmethod
	def sample(cls, f, x0, xf, size, *args):
		
		numbers = cls.distribution(f, x0, xf, *args)

		if isinstance(size, int):
						
			current_sample = []

			for i in range(size):
				current_sample.append(choice(numbers))

			return current_sample

		elif isinstance(size, tuple):
			
			samples = []

			for i in range(np.prod(size[0: len(size)-1])):

				current_sample = []

				for j in range(size[-1]):

					current_sample.append(choice(numbers))

				samples.append(current_sample)

			samples = np.reshape(samples, size)
			
			return samples

		else:
			raise NameError('Invalid size argument!')



	
	

