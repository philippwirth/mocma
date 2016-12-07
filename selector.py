import numpy as np
import math

import crowdingdistance as cd

# TODO: Implement fast nondominated sorting
def selectBest(individuals, mu):

	n = len(individuals)			# get the number of individuals
	dim = individuals[0].fitness.size	# get the number of objectives

	# compute crowding distance
	crowdingDistances = cd.computeCrowdingDistances(individuals, n, dim)
	
	# assign and sort by nondomination ranks
	nonDominationRanks = np.zeros(n)

	for i in range(n):				# naive O(n^2) implementation
		for j in range(n):
			if individuals[i].dominates(individuals[j]):
				nonDominationRanks[j] += 1

	perm = nonDominationRanks.argsort()
	nonDominationRanks = nonDominationRanks[perm]	# sort nondomination ranks
	individuals = np.array(individuals)[perm]	# sort individuals by nondomination ranks


	# sort each level of nondomination according to the crowding distance
	nextGenCount = 0			# keeps count of the number of individuals in the next generation
	nextGen = []				# stores the best mu individuals and builds the next generation
	
	for level in range(2*mu):		# for each possible level of nondominance: sort all individuals of..
		tempIndividuals = []		# ..that level according to their crowding distance
		tempCrowdingDistances = []
		while(nonDominationRanks[nextGenCount] == level and nextGenCount < mu):
			tempIndividuals.append(individuals[nextGenCount])
			tempCrowdingDistances.append(crowdingDistances[nextGenCount])
			nextGenCount += 1

		if (len(tempIndividuals) > 0):	# if there are no individuals with that level of nondominance: no need to sort
			tempIndividuals = np.array(tempIndividuals)
			tempCrowdingDistances = np.array(tempCrowdingDistances)
			
			perm = tempCrowdingDistances.argsort()
			tempIndividuals = tempIndividuals[perm]
			tempCrowdingDistances = tempCrowdingDistances[perm]

			for indiv in tempIndividuals:
				nextGen.append(indiv)

	return nextGen
