import numpy as np
import math

MAX = pow(10,2)		# crowding distance that is assigned to min and max points

# computes the crowding distance for each individual in the list
def computeCrowdingDistances(individuals, n, dim):

	crowdingDistances = np.zeros(n)		# array to store the crowding distance of each individual
	dimi = np.zeros(n)					# temporary array to store fitness values of dimension i

	for i in range(dim):

		# for each dimension i sort individuals by fitness[i]							
		for j in range(n):
			dimi[j] = individuals[j].fitness[i]

		perm = dimi.argsort()
		dimi = dimi[perm]

		# assign MAX crowding distance to min and max points
		crowdingDistances[perm[0]] = MAX
		crowdingDistances[perm[n-1]] = MAX

		# compute crowding distance for all points in between
		for j in range(2,n-1):
			crowdingDistances[perm[j]] += (dimi[i+1] - dimi[i-1]) / (dimi[n-1] - dimi[0])

	# return the result
	return crowdingDistances
