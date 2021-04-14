import random
import numpy as np
from multiprocessing.pool import ThreadPool

from errors import *

class GRS:
	"""Global Random Search"""
	def __init__(self, pos, objective_function, min=float('-inf'), max=float('inf')):
		self.best = pos
		self.dimension = len(pos)
		self.objective_function = objective_function
		self.j_best = objective_function(*pos)
		self.min, self.max = min, max

	def create_candidate(self, i=0):
		'''Função que gera um novo candidato para comparar com o melhor'''
		cand = []
		for i in range(self.dimension):
			try:
				min = self.min[i]
			except TypeError:
				min = self.min

			try:
				max = self.max[i]
			except TypeError:
				max = self.max

			cand.append(random.uniform(min, max))

		j_cand = self.objective_function(*cand)
		return j_cand, cand

	def __next__(self, pool=None, cand_ammt=1):
		'''Chamada de uma iteração, usando paralelismo'''
		if pool is None:
			pool = ThreadPool()

		with pool as p:
			# Gerando candidatos e pegando o menor.
			# como o retorno vem na ordem (valor_da_função, candidato), a função mínimo
			# vai retornar a tupla que possui menor valor na primeira posição.
			chosen =  min(p.map(self.create_candidate, range(cand_ammt)))

		return chosen

	def __call__(self, Ni=200, parallel=True, cands_per_iteration=15, pool=None):
		'''Chamada principal para executar multiplas iterações'''
		if parallel:
			pool = ThreadPool() if pool is None else pool
			next_candidate = lambda: next(self, pool, cands_per_iteration)
		else:
			next_candidate = lambda: self.create_candidate()

		for i in range(Ni):
			j_cand, cand = next_candidate()

			if j_cand < self.j_best:
				self.best = cand
				self.j_best = j_cand

		return self.best