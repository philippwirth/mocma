import numpy as np
import matplotlib.pyplot as plt
import math
import sys

import fitness
import selector

# options
mu = 256		# population size
inputDim = 1		# search space dimension
outputDim = 2		# number of objectives
maxGenerations = 100	# maximum number of generations

# external strategy parameters
pTarget = pow(5 + math.sqrt(0.5), -1) 	# target success probability
dDamping = 1 + mu/2 			# step size damping parameter
cSuccRateParam = pTarget/(2 + pTarget)	# success rate averaging parameter
cCumulTimeParam = 2/(2 + mu)		# cumulation time horizon parameter
cCov = 2/(pow(mu,2) + 6)		# covariance matrix learning rate
pThresh = 0.44				# pthreshold


# ---------------------------- Individual ---------------------------------#
class Individual:


	def __init__(self, _x, _pSucc, _sigma, _pEvol, _C):
		self.x = _x
		self.pSucc = _pSucc
		self.sigma = _sigma
		self.pEvol = _pEvol
		self.C = _C

		self.fitness = fitness.f(self.x)

		self.inputDim = self.x.size
		self.outputDim = self.fitness.size
		

	def dominates(self, other):
		temp = False
		for i in range(self.outputDim):
			
			# check whether x[i] <= y[i] for all
			if self.fitness[i] > other.fitness[i]:		
				return False
			
			# check whether exists x[i] < y[i]
			if self.fitness[i] < other.fitness[i]:		
				temp = True
		return temp

	def updateStepSize(self):
		if self.succ:
			# if mutation was successful: increase pSucc
			self.pSucc = (1 - cSuccRateParam)*self.pSucc + cSuccRateParam
		else:
			# if mutation was not successful: descrease pSucc
			self.pSucc = (1 - cSuccRateParam)*self.pSucc

		# increase step size if success probability pSucc is bigger than target success probability pTarget
		self.sigma = self.sigma * math.exp((self.pSucc - pTarget) / (dDamping*(1 - pTarget)))
		
	def updateCovariance(self):
		if self.pSucc < pThresh:
			# if the success rate is smaller than the threshold the mutation step is used to update the covariance matrix
			self.pEvol = (1 - cCumulTimeParam)*self.pEvol + math.sqrt(cCumulTimeParam*(2 - cCumulTimeParam))*self.step
			self.C = (1 - cCov)*self.C + cCov*(np.transpose(self.pEvol)*self.pEvol)
		else:
			# if the success rate is higher than the threshold the mutation steop is not used for the update
			self.pEvol = (1 - cCumulTimeParam)*self.pEvol
			self.C = (1 - cCov)*self.C + cCov*(np.transpose(self.pEvol)*self.pEvol + cCumulTimeParam*(2 - cCumulTimeParam)*self.C)

	def mutate(self):

		# find x of the mutation
		newx = np.random.multivariate_normal(self.x, pow(self.sigma,2)*self.C)
		
		# create mutated individual
		mutation = Individual(newx, self.pSucc, self.sigma, self.pEvol, self.C)
		
		# set mutation step and check whether the mutation dominates its parent
		mutation.step = (mutation.x - self.x)/self.sigma
		mutation.succ = mutation.dominates(self)
		self.succ = mutation.succ

		return mutation



# ----------------------------- MO-CMA-ES ----------------------------------#


# initialization
print('Initialization.')

initialSigma = 5.0				# initial sigma and initial mean determine..
initialMean = np.zeros(inputDim)		# ..the distribution of the initial population

currentPop = []					# create an initial population in initialMean +- 2*initialSigma
for i in range(mu):
	xi = np.random.rand(inputDim)*4*initialSigma - 2*initialSigma
	Ci = np.identity(inputDim)
	currentPop.append(Individual(xi, pTarget, initialSigma, 0, Ci))

# loop
for g in range(0,maxGenerations):

	print('Evolution progress: ', 100*(g+1)/maxGenerations, '%', end='\r')

	# step 1: reproduction
	Q = []
	for k in range(mu):
		Q.append(currentPop[k].mutate())

	# step 2: updates
	for k in range(mu):
		# update step size
		currentPop[k].updateStepSize()
		Q[k].updateStepSize()

		# update covariance matrix
		Q[k].updateCovariance()

		#create mixed population
		Q.append(currentPop[k])

	# step 3: selection
	currentPop = selector.selectBest(Q, mu)

# end
print('')
print('Final Population:')
for i in range(mu):
	print('Individual ', i, ': f(', currentPop[i].x, ') = ', currentPop[i].fitness)
print('Done.')
