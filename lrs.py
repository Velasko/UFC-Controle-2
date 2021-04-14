import random
import numpy as np

from grs import GRS

class LRS(GRS):
	"""Local Random Search.

	Herdado da classe "Global Random Search". """
	def __init__(self, *args, sigma=.1, **kwargs):
		super().__init__(*args, **kwargs)
		self.sigma = sigma

	def create_candidate(self, i=0):
		'''Função para gerar o candidato'''
		cand = self.best + np.random.normal(0, self.sigma, self.dimension)
		cand = np.clip(cand, self.min, self.max)

		j_cand = self.objective_function(*cand)
		return j_cand, cand