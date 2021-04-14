import control
import random

import numpy as np
from multiprocessing.pool import ThreadPool

class Particle():
	swarm = []

	# O método seguinte é para lidar com as heranças.
	# Para evitar o reuso de código quando lidando com PSO global
	# ou com o PSO local. No caso de outras abordagens, possivelmente melhores,
	# isso pode se fazer desnecessário.
	def __new__(cls, *args, **kwargs):
		'''Used to have a class-wide list, without inheritance interference.
		With this, global and local particles can be used simutaneously
		without interfering with each other.'''

		obj = super(Particle, cls).__new__(cls)

		# Cada um das classes possui uma lista de todas as partículas,
		# que é acessível à todas as particulas à todos os momentos.
		# O trexo a segui instancia essa lista e adiciona as novas partículas à ela
		Particle.swarm.append(obj)
		try:
			cls.swarm.append(obj)
		except AttributeError:
			cls.swarm = [obj]

		#class constants
		cls.inertia = 0
		cls.acceleration_coefficients = np.identity(2) #shown as c1 and c2 on slides
		cls.r_vectors = None

		cls.pos_min, cls.pos_max = float('-inf'), float('inf')

		return obj

	def __init__(self, pos, speed, objective_function):
		self.pos = pos
		self.speed = speed
		self._objective_function = objective_function

		self.best = float('inf')
		self.best_pos = pos
		self.objective_function()

	def __lt__(self, other):
		if isinstance(other, Particle):
			return self.best < other.best
		raise TypeError

	def __repr__(self):
		decimal_lim = lambda x: f"{x:.2f}"
		return f"Particle obj @ {', '.join(map(decimal_lim, self.pos))} | @' {', '.join(map(decimal_lim, self.speed))}."

	def __str__(self):
		return self.__repr__()

	def update(self, best_particle):
		"""Updates particle's speed and position, using the best considered particle.
		The child defines what is the best particle"""
		inertia = type(self).inertia
		accel = type(self).acceleration_coefficients
		r = type(self).r_vectors

		direction = np.array([
			self.best_pos - self.pos,
			best_particle.best_pos - self.pos
		])

		product = sum(
			accel @ np.multiply(r, direction)
		)

		self.speed = inertia * self.speed + product
			
		self.pos += self.speed

		# Ajustando os valores para enquadrarem os mínimos e maximos.
		self.pos = np.clip(self.pos, type(self).pos_min, type(self).pos_max)

	def objective_function(self):
		pos = self.pos
		r = self._objective_function(*pos)
		if r < self.best:
			self.best_pos, self.best = pos.copy(), r

		return r

class GlobalParticle(Particle):
	"""Classe para o PSO global. Essa classe/algoritmo usará a melhor partícula
	entre todas do enxame para definir sua próxima posição"""
	def update(self):
		# Considerando que na função de iteração do Swarm, a lista de partículas é ordenada,
		# GlobalParticle.swarm[0] fica sendo a melhor partícula de todo o enxame.
		# Caso isso não fosse verdade, min(GlobalParticle.swarm) deveria ser usado,
		# causando uma penalidade em performance.
		return super(GlobalParticle, self).update(GlobalParticle.swarm[0])

class LocalParticle(Particle):
	"""Classe baseada no PSO local. Usará a melhor entre as duas partículas irmãs para
	definir sua próxima posição."""
	def __init__(self, *args, **kwargs):
		self.left, LocalParticle.swarm[-1].right = LocalParticle.swarm[-1], self
		self.right, LocalParticle.swarm[0].left = LocalParticle.swarm[0], self

		super(LocalParticle, self).__init__(*args, **kwargs)

	def update(self):
		best = min(self.left, self.right)
		return super(LocalParticle, self).update(best)

class AlternateLocalParticle(Particle):
	"""Class based on local PSO (made by me). This class will use
	the best particle within a certain distance in order to decide where
	to go next."""
	def update(self):
		local_range = lambda x: self.dist(x) < 100
		slice_ = [value for value in (map(local_range, LocalParticle.swarm))]
		best = np.array(LocalParticle.swarm)[slice_][0]
		return super(AlternateLocalParticle, self).update(best)

	def dist(self, other):
		if isinstance(other, AlternateLocalParticle):
			return ( sum( (self.pos - other.pos)**2 ) ) ** 0.5
		raise TypeError

class Swarm():
	def __init__(self,
		N,
		distribution_function,
		objective_function,
		dimension,
		particle_type=GlobalParticle,
		inertia=0.4,
		acceleration_coefficients=np.array([2.05, 2.05]),
		r_vectors=None,
		min=None, max=None):

		self.iteration = 0
		self.particle = particle_type

		# Gerando as partículas de acordo com a classe/o tipo de algorítmo desejado.
		# Uma função de distribuição é a maneira que define como que cada partícula é gerada, com sua
		# tanto sua posição, quanto sua velocidade.
		for i in range(N):
			self.particle(
				*distribution_function(i),
				objective_function=objective_function
			)

		#Definindo os parâmetros da função
		self.particle.inertia = inertia
		self.particle.acceleration_coefficients = np.identity(2) * acceleration_coefficients

		if r_vectors is None:
			r_vectors = np.random.rand(2, dimension)
		self.particle.r_vectors = r_vectors

		if not min is None:
			self.particle.pos_min = min
		if not max is None:
			self.particle.pos_max = max

	def __repr__(self):
		'''Representação da variável'''
		text = "Swarm obj: {\n"
		text += '\n'.join(map(str, self.particle.swarm[:5]))
		if len(self.particle.swarm) > 5:
			text += '\n...'
		text += "\n}\n" + f"best @ {self.particle.swarm[0].best_pos} ({self.particle.swarm[0].best:.2f})"
		text += f"\nIteration: {self.iteration}."
		return text

	def __next__(self, pool=None):
		'''Função para avançar o número de iterações'''
		if pool is None:
			pool = ThreadPool()

		with pool as p:
			p.map(self.particle.objective_function, iterable=self.particle.swarm)
			self.particle.swarm.sort()
			p.map(self.particle.update, iterable=self.particle.swarm)

		self.iteration += 1

		return self.iteration

	def __getitem__(self, key):
		return self.particle.swarm.__getitem__(key)

	def __iter__(self):
		#Uma maneira de percorer o vetor de partículas
		return self.particle.swarm.__iter__()

if __name__ == '__main__':
	dimensions = 2
	part_dist = lambda n: np.random.rand(2, dimensions)
	of = lambda x, y: - x * y
	s = Swarm(15, part_dist, of, 2)

	l = LocalParticle(*part_dist(0), of)
	print(l in GlobalParticle.swarm)

	for p in s:
		print(p)
	