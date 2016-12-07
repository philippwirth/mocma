import numpy as np

def f(x):
	# this is an example multi-objective fitness function used for testing
	res = np.zeros(2)
	for i in range(x.size):
		if x[i] <= 1:
			res[0] = -x[i]
		elif x <= 3:
			res[0] = x[i] - 2
		elif x <= 4:
			res[0] = 4 - x[i]
		else:
			res[0] = x[i] - 4

		res[1] += pow(x-5,2)
	return res
