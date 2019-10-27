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

def int_uniform(f, x0, xf, N, *args):

	if not isinstance(x0, (list, np.ndarray)):
		raise NameError('x0 must be a list or numpy.ndarray of the initial points!')

	if not isinstance(xf, (list, np.ndarray)):
		raise NameError('x0 must be a list or numpy.ndarray of the final points!')

	if len(x0) != len(xf):
		raise ValueError('x0 and xf must have the same length!')

	x0 = np.array(x0)
	xf = np.array(xf)

	D = len(x0)

	# generate a matrix (N, D) of random numbers 
	X = np.array([np.random.uniform(x0[i], xf[i], N) for i in range(D)]).transpose()

	# total volume
	V = np.prod(xf - x0)

	ys = np.array([f(x, *args) for x in X])

	integral = np.mean(ys) * V

	error = np.sqrt(np.var(ys) / N) * V

	return (integral, error)