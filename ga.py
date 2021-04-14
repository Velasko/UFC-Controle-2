import control
import random

from multiprocessing.pool import ThreadPool
from threading import Lock
import numpy as np

#RFunções de recombinação
class Wright():
	def __init__(self, function, pc=0.85, *args, **kwargs):
		self.pc = pc
		self.function = function

	def __call__(self, father, mother):
		if np.random.uniform(0,1) <= self.pc:
			child1 = father + mother
			child2 = 1.5*father - 0.5*mother
			child3 = -0.5*father + 1.5*mother

			childs = []
			if self.function(*child1) < self.function(*child2):
				childs.append(child1)
				other = child2
			else:
				childs.append(child2)
				other = child1

			if self.function(*other) < self.function(*child3):
				childs.append(other)
			else:
				childs.append(child3)

			return childs
		return father, mother

class SimulatedBinaryCrossover:
	def __init__(self, n=1, *args, **kwargs):
		self.n = n

	def __call__(self, father, mother):
		r = np.random.rand(len(father))
		small, large = (r <= .5), (r > .5)
		r[small] = (2 * r[small]) ** (1 / (self.n + 1))
		r[large] = (1 / (2 * (1 - r[large]) ) ) ** (1 / (self.n + 1))

		child1 =  0.5*(np.multiply(1+r, father) + np.multiply(1-r, mother))
		child2 =  0.5*(np.multiply(1-r, father) + np.multiply(1+r, mother))

		return child1, child2

# Classe principal
class Population():
	def __init__(self, N, p, function, pm=.05, alpha=0.5,
		recombination=Wright, upper=float('inf'), lower=float('-inf'),
		rec_func={'pc' : 0.85, 'n' : 0.5}):

		self.recombination_function = recombination(function=function, **rec_func)

		self.fun = function
		self.generation = 0

		self.pop_size = N
		self.dimension = p

		self.pm = pm

		self.alpha = alpha

		if isinstance(upper, int):
			self.upper = np.array([upper]*p)
		else:
			self.upper = upper

		if isinstance(lower, int):
			self.lower = np.array([lower]*p)
		else:
			self.lower = lower

		self.pop = np.multiply(np.random.rand(N, p), (upper - lower) + lower)

	def __next__(self, *args, **kwargs):
		'''Função para avançar nas gerações'''

		#elitismo:
		order = lambda ent: (self.fun(*ent), [*ent]) 	# Função para poder ordenar os melhores da população usando sort
		best = [e for e in map(order, self.pop)] 		# Mapeando a população com o ordenador
		best.sort()
		next_pop = [e[1] for e in best[:2]]				# Definindo o vetor da proxima população, com os dois melhores da população anterior inclusos
														# e excluindo o fator de ordenação de cada um dos itens.

		while len(next_pop) < self.pop_size:
			# X(t+1) = M(R(S(X(t))))
			selecteds = self.select()
			recombined = self.recombine(selecteds)
			mutated = self.mutate(recombined)
			clip = np.clip(recombined, self.lower, self.upper)
			next_pop += [*clip]

		self.generation += 1
		self.pop = np.array(next_pop)

	def get_best(self):
		'''Metodo simples para pegar o melhor da população'''
		order = lambda ent: (self.fun(*ent), [*ent])
		best = [e for e in map(order, self.pop)]
		best.sort()
		return best[0][1]

	def select(self):
		parents = self.pop[np.random.choice(self.pop_size, size=4, replace=False)]
		if self.fun(*parents[0]) < self.fun(*parents[1]):
			father = 0
		else:
			father = 1

		if self.fun(*parents[2]) < self.fun(*parents[3]):
			mother = 2
		else:
			mother = 3

		return parents[[father, mother]]
	
	def recombine(self, selecteds):
		recombined = self.recombination_function(*selecteds)
		recombined = np.array([*recombined])

		return recombined

	def mutate(self, children):
		mutants = []
		for ent in children:
			if random.uniform(0, 1) > self.pm:
				ent = ent + (self.alpha * (self.upper - self.lower) + self.lower) * np.random.rand(len(ent))
			else:
				mutants.append(ent)

		return np.array(mutants)

	# Funções cosmétias e de conveniência:
	def __iter__(self, *args, **kwargs):
		return self.pop.copy().__iter__(*args, **kwargs)

	def __str__(self, *args, **kwargs):
		return self.pop.__str__(*args, **kwargs) + f'\nGeneration: {self.generation}'

	def __repr__(self, *args, **kwargs):
		return self.pop.__repr__(*args, **kwargs) + f'\nGeneration: {self.generation}'

if __name__ == '__main__':
	of = lambda x: sum(x**2)
	p = Population(8, 3, of, lower=np.zeros(3), upper=np.array([1.5, 1, 1]))
	next(p)