import sys
import pygame
import numpy as np

from multiprocessing.pool import ThreadPool

sys.path.append('..')
from pso import *

size = 500, 500
dimensions = 2

def df(n):
	r = np.random.rand(2, dimensions)
	r[0] *= 500
	return r

of = lambda x, y: abs(x - 250)**4 + abs(y - 250)**4
swarm = Swarm(15, df, of, dimension=2, min=0, max=np.array(size))
pool = ThreadPool()

pygame.init()
grey = np.array([1, 1, 1])
play = False
screen = pygame.display.set_mode(size)

while True:
	screen.fill(255*grey)

	for particle in swarm:
		pygame.draw.circle(screen, (0, 0, 255),
			particle.pos, 3, 3)

	if play:
		next(swarm, pool)

	for event in pygame.event.get():
		if event.type == pygame.QUIT: sys.exit()
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				sys.exit()
			elif event.key == pygame.K_n:
				next(swarm, pool)
				print(swarm)
			elif event.key == pygame.K_p:
				play = not play
			elif event.key == pygame.K_r:
				swarm = Swarm(15, df, of, dimension=2, min=0, max=np.array(size))


	pygame.time.Clock().tick(60)

	pygame.display.flip()