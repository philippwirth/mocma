import numpy as np
import math

import crowdingdistance as cd

def selectBest(individuals, mu):

	n = len(individuals)
	dim = individuals[0].fitness.size

	# compute crowding distance
	#crowdingDistances = cd.computeCrowdingDistances(individuals, n, dim)
	crowdingDistances = np.random.rand(n)
	# assign and sort by nondomination ranks
	nonDominationRanks = np.zeros(n)

	for i in range(n):		# naive O(n^2) implementation
		for j in range(n):
			if individuals[i].dominates(individuals[j]):
				nonDominationRanks[j] += 1

	perm = nonDominationRanks.argsort()
	nonDominationRanks = nonDominationRanks[perm]	# sort nondomination ranks
	individuals = np.array(individuals)[perm]		# sort individuals by nondomination ranks


	# sort each level of nondomination according to the crowding distance
	nextGenCount = 0
	nextGen = []
	for level in range(2*mu):

		tempIndividuals = []
		tempCrowdingDistances = []
		while(nonDominationRanks[nextGenCount] == level and nextGenCount < mu):
			tempIndividuals.append(individuals[nextGenCount])
			tempCrowdingDistances.append(crowdingDistances[nextGenCount])
			nextGenCount += 1

		if (len(tempIndividuals) > 0):
			tempIndividuals = np.array(tempIndividuals)
			tempCrowdingDistances = np.array(tempCrowdingDistances)
			
			perm = tempCrowdingDistances.argsort()
			tempIndividuals = tempIndividuals[perm]
			tempCrowdingDistances = tempCrowdingDistances[perm]

			for indiv in tempIndividuals:
				nextGen.append(indiv)


	return nextGen
