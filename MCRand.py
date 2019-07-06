"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from random import choices


class RandGen(object):
	"""docstring for RandGen"""
	def __init__(self):
		pass
		
	@classmethod
	def distribution(cls, f, x0, xf, *args):
		# maximum number of samples returned
		max_sample = 10**4
		# maximum number of samples used at each iteration of the MC
		max_sample_per_iter = 10**4

		#Get the maximum of the f probability function

		x = f(np.linspace(x0, xf, 1000), *args)

		maximum = np.amax(x)

		new_randoms = np.empty(max_sample, dtype=float)

		n = 0
		while n < max_sample:

			random_numbers = np.random.uniform(x0, xf, max_sample_per_iter)
			dices = np.random.uniform(0, maximum, max_sample_per_iter)

			accept = random_numbers[f(random_numbers, *args) > dices]
			accepted = len(accept)

			if n + accepted > max_sample:
				new_randoms[n:max_sample] = accept[:max_sample - n]
				n = max_sample
			else:
				new_randoms[n:n+accepted] = accept[:]
				n += accepted

		return new_randoms

	@classmethod
	def sample(cls, f, x0, xf, size, *args):
		
		numbers = cls.distribution(f, x0, xf, *args)

		if isinstance(size, int):	
			return choices(numbers, k = size)

		elif isinstance(size, tuple):
			samples = choices(numbers, np.prod(size))
			return np.reshape(samples, size)

		else:
			raise NameError('Invalid size argument!')