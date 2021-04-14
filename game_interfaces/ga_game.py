import sys
import pygame
import numpy as np

from multiprocessing.pool import ThreadPool

from ga import *

size = 500, 500
dimensions = 2

def df(n):
	r = np.random.rand(2, dimensions)
	r[0] *= 500
	return r

pop_size = 15
dim = 2

of = lambda x, y: abs(x - 250)**4 + abs(y - 250)**4
population = Population(pop_size, dim, of, lower=np.zeros(dim), upper=np.array([500, 500]))

pygame.init()
grey = np.array([1, 1, 1])
play = False
screen = pygame.display.set_mode(size)

while True:
	screen.fill(255*grey)

	for ent in population:
		pygame.draw.circle(screen, (0, 0, 255),
			ent, 3, 3)

	if play:
		next(population)

	for event in pygame.event.get():
		if event.type == pygame.QUIT: sys.exit()
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				sys.exit()
			elif event.key == pygame.K_n:
				next(population)
				print(population)
			elif event.key == pygame.K_p:
				play = not play
			elif event.key == pygame.K_r:
				population = Population(pop_size, dim, of, lower=np.zeros(dim), upper=np.array([500, 500]))

	pygame.time.Clock().tick(60)

	pygame.display.flip()