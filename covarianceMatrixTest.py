import numpy as np
import matplotlib.pyplot as plt
import math
import sys

import fitness
import selector


# options
mu = 256				# population size
inputDim = 1			# search space dimension
outputDim = 2			# number of objectives
maxGenerations = 100	# maximum number of generations

plotPoints = True

# external strategy parameters
pTarget = pow(5 + math.sqrt(0.5), -1) 	# target success probability
dDamping = 1 + mu/2 					# step size damping
cSuccRateParam = pTarget/(2 + pTarget)	# success rate averaging parameter
cCumulTimeParam = 2/(2 + mu)			# cumulation time horizon parameter
cCov = 2/(pow(mu,2) + 6)				# covariance matrix learning rate
pThresh = 0.44							# pthreshold


# --------------------------- Individual ---------------------------------#
class Individual:


	def __init__(self, _x, _pSucc, _sigma, _pEvol, _C):
		self.x = _x
		self.pSucc = _pSucc
		self.sigma = _sigma
		self.pEvol = _pEvol
		self.C = _C

		#self.fitness = fitness.f(self.x)

		self.inputDim = self.x.size
		self.outputDim = 2
		#self.outputDim = self.fitness.size
		

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

		return mutation


step0 = np.random.rand(1,2)

print(step)
print(np.transpose(step))
print(np.transpose(step)*step)
ind0 = Individual(np.zeros(2), 0, 2.0, 0, np.identity(2)*np.transpose(step)*step)
ind1 = Individual(np.zeros(2), 0, 1.0, 0, np.identity(2)*np.transpose(step)*step)

for i in range(100):
	temp = ind0.mutate()
	plt.plot(temp.x[0], temp.x[1], 'bo')
	temp = ind1.mutate()
	plt.plot(temp.x[0], temp.x[1], 'ro')

plt.axis('equal')
plt.show()