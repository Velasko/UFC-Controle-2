import os
import control

import numpy as np
import matplotlib.pyplot as plt

import time as timelib
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import errors
import euler
import pso, grs, lrs, ga

s = control.TransferFunction.s
betas = {
		1.0 : [0.6570, 0.5389, 0.2458],
		1.5 : [0.6254, 0.4577, 0.2187]
	}

def base_system(kp=1, ki=0, kd=0):
	con = kp + ki/s + kd*s
	amp = 10 / (1 + .1*s)
	exc = 1 / (1 + .4 * s)
	gen = 1 / (1 + s)
	sen = 1 / (1 + .01*s)

	open_system = con * amp * exc * gen
	closed = open_system.feedback(sen)

	return closed

def Q1_i():
	import inspect
	print(inspect.getsource(base_system))
	print(base_system())

def Q1_ii():
	plt.plot(*control.step_response(base_system()))
	plt.grid(True, linestyle='--')
	plt.show()

def Q1_iii():
	time = np.arange(0, 2, .001)

	_leg = []
	for beta, params in betas.items():
		step = control.step_response(base_system(*params), time)
		print("Beta:", beta, "Error:", errors.ITAE(*step))

		plt.plot(*step)
		_leg.append(str(beta))

	plt.legend(_leg)
	plt.grid(True, linestyle='--')
	plt.xlabel('t, sec')
	plt.ylabel('Terminal Voltage')
	plt.show()

def Q2_ii():
	steps = [.001, .1]
	fig, axs = plt.subplots(1, 2)

	for n, step in enumerate(steps):
		time = np.arange(0, 5, step)

		axs[n].plot(*control.step_response(base_system(), time))
		axs[n].plot(*euler.discretization_simple(1, 0, 0, time=time), 'k^', markersize=5)

		axs[n].set_title(f"step of: {step}")
		axs[n].legend(['Resposta da Questão 1.ii', 'Discretização'])
		axs[n].grid(True, linestyle='--')
		axs[n].set(xlabel='t, sec', ylabel='Terminal Voltage')
	plt.show()

def Q2_iii():
	steps = [.001, .1]
	fig, axs = plt.subplots(1, 2)
	for n, step in enumerate(steps):
		time = np.arange(0, 5, step)
		_beta = []
		for beta, params in betas.items():
			axs[n].plot(*euler.discretization_simple(*params, time=time), '^', markersize=3)
			_beta.append(beta)

		axs[n].set_title(f"step of: {step}")
		axs[n].legend(_beta)
		axs[n].grid(True, linestyle='--')
		axs[n].set(xlabel='t, sec', ylabel='Terminal Voltage')
	plt.show()

def Q3():
	time = np.arange(0, 5, .001)
	tipos = { f'Beta: {key}' : value for key, value in betas.items()}
	tipos['Item ii'] = [1, 0, 0]

	_leg = []
	for key, value in tipos.items():
		plt.plot(*euler.discretization(base_system(*value), time=time))

		_leg.append(key)

	plt.legend(_leg)
	plt.grid(True, linestyle='--')
	plt.xlabel('t, sec')
	plt.ylabel('Terminal Voltage')
	plt.show()


# Questão 4:
default_time = np.arange(0, 2, .001)

def of_wrapper(time):
	objective_function = lambda kp, ki, kd: errors.ITAE(
		*[ np.array(item) for item in euler.discretization(
			base_system(kp, ki, kd), time=time
		)]
	)

	return objective_function

#funções para a execução com cada algorítmo.
	# elas seguem o seguinte padrão:
	# 1. Garantir o input 
	# 2. Instanciar itens necessários (como a função objetiva)
	# 3. Instanciar o objeto que representa o algorítmo de escolha
	# 4. Chamar função que executa as iterações ou iterar sobre o objeto

def Q4_PSO(population=15, iterations=200, time=default_time, pool=ThreadPool()):
	dimensions = 3
	def df(n):
		'''Função de distribuição para definir a posição e velocidade '''
		particle_dist = np.random.rand(2, dimensions) #2 pois pos e vel. dimensions (3), para kp, ki e kd
		particle_dist[0][0] *= 1.5 #valor maximo de kp é 1.5 e a distribuição estava entre 0 e 1.
		return particle_dist

	objective_function = of_wrapper(time)

	swarm = pso.Swarm(population, df, objective_function,
		dimension=dimensions, min=0, max=np.array([1.5, 1, 1]))
	
	for n in range(iterations):
		next(swarm, pool)

	return swarm[0].pos

def Q4_GRS(candidates=15, iterations=200, time=default_time, pool=ThreadPool()):
	initial = [1, 0, 0]

	objective_function = of_wrapper(time)

	g_search = grs.GRS(initial, objective_function,
		min=0, max=np.array([1.5, 1, 1]))

	# executando sem paralelismo Ni vezes.
	best = g_search(
			cands_per_iteration=candidates,
			Ni=iterations,
			parallel=False,
			# pool=pool
		)

	return best

def Q4_LRS(candidates=15, iterations=200, sigma=.1, time=default_time, pool=ThreadPool()):
	initial = [1, 0, 0]

	objective_function = of_wrapper(time)

	l_search = lrs.LRS(initial, objective_function, sigma=sigma,
		min=0, max=np.array([1.5, 1, 1]))

	best = l_search(
			cands_per_iteration=candidates,
			Ni=iterations,
			parallel=False,
			# pool=pool
		)

	return best

def Q4_AG(pop_size=16, iterations=200, time=default_time, pool=ThreadPool()):
	dim = 3

	if pop_size % 2 == 1:
		pop_size += 1

	objective_function = of_wrapper(time)

	pop = ga.Population(pop_size, dim, objective_function,
		lower=np.zeros(dim), upper=np.array([1.5, 1, 1])
	)

	for _ in range(iterations):
		next(pop)

	return pop.get_best()

def Q4():
	# Cronometagem sobre o tempo de execução
	execution_start = timelib.time()

	# Parametros do artigo que é usado para comparação
	artigo = [0.6570, 0.5389, 0.2458]

	# Definindo a linha do tempo para calculo do erro
	# e região do plot
	time = np.arange(0, 2, .001)
	objective_function = of_wrapper(time)

	# definindo número de processos disponíveis para
	# paralelizar a computação intensiva que virá
	pool = Pool(3)

	# Definindo argumentos que serão passados para as funções
	# definidas acima.
	# Os argumentos são posicionais.
	# No caso são o tamanho de cada iteração/geração
	# Quandas gerações/iterações devem ser executadas
	# range(10) define que as funções acima serão executadas 10 vezes 
	proc_args = [(15, 200) for e in range(10)]

	# As funções acima retornam somente o melhor item que encontraram.
	# ordenator é usado para ordenar os os itens que essas funções retornaram (por meio do sort).
	ordenator = lambda x: (objective_function(*x), [*x])

	# Criando os subplots
	fig, axs = plt.subplots(2, 2)

	# As funções que serão chamadas em paralelas:
	algorithms = [Q4_PSO, Q4_GRS, Q4_LRS, Q4_AG]
	with pool as p:
		for n, algorithm in enumerate(algorithms):
			f_name = algorithm.__name__.split('_')[1] 	#nome do algorítmo.
			coords = p.starmap(algorithm, proc_args)	#Iniciando execução paralelizada.

			# ordenando os itens:
			coords = [e for e in map(ordenator, coords)]
			coords.sort()
			coords = [e[1] for e in coords]

			# Definindo subplot e adicionando o comparativo do artigo
			subplt = axs[int(n > 1), int(n % 2 == 0)]
			subplt.plot(*control.step_response(base_system(*artigo), time))
			leg = [f'artigo: {objective_function(*artigo):.8f}']

			# Adicionando os dois melhores resultados ao subplot
			for k, coord in enumerate(coords[:2]):
				subplt.plot(*euler.discretization(base_system(*coord), time=time))
				leg.append(f"{k+1}o melhor: {objective_function(*coord):.5f}")

			# Escrevendo todos os 10 resultados das funções à um arquivo
			mode = 'w' if n == 0 else 'a'
			with open('gains_output.txt', mode) as file:
				file.write(f'{f_name}:\n')
				file.write('[' + ',\n'.join(map(str, coords)) + ']\n\n')

			# Adicionando informações extras
			subplt.set_title(f_name + " result:")	# Título do subplot para identificação do algorítmo
			subplt.legend(leg)						# legendas para identificação da curva
			subplt.grid(True, linestyle='--')
			subplt.set(xlabel='t, sec', ylabel='Terminal Voltage')

			print(f"{f_name} best: {coords[0]}")

	print(f"Execution took: {(timelib.time()-execution_start)/60:.2f} minutes")

	plt.show()

def Q4_plt():
	'''Função para plottar as informações escritas em gains_output.txt'''
	import re
	with open('gains_output.txt', 'r') as file:
		data = file.readlines()

	artigo = [0.6570, 0.5389, 0.2458]
	time = np.arange(0, 2, .001)
	objective_function = of_wrapper(time)
	raz = objective_function(*artigo)

	n = 0

	fig, axs = plt.subplots(2, 2)
	subplt = axs[int(n > 1), int(n % 2 == 0)]
	# subplt = plt

	leg = [f'artigo: {objective_function(*artigo):.8f}']
	content =[]
	for line in data:
		if line == '\n':
			subplt.plot(*control.step_response(base_system(*artigo), time))
			for k, coord in enumerate(content[:2]):
				subplt.plot(*euler.discretization(base_system(*coord), time=time))
				leg.append(f"{k+1}o melhor: {objective_function(*coord):.5f}")
				print(f'razão para o {k+1}o melhor: {objective_function(*coord)/raz}')

			subplt.legend(leg)
			subplt.grid(True, linestyle='--')

			subplt.set(xlabel='t, sec', ylabel='Terminal Voltage')
			# subplt.xlabel('t, sec')
			# subplt.ylabel('Terminal Voltage')
			# subplt.show()

		elif ':' in line:
			#next subplt

			subplt = axs[int(n > 1), int(n % 2 == 0)]
			subplt.set_title(line[:-2] + " result:")
			# subplt.title(line[:-2] + " result:")

			print(line[:-1])
			n += 1
			leg = [f'artigo: {objective_function(*artigo):.8f}']
			content = []

		else:
			content.append([e for e in map(float, re.sub(']|,|\n', '', line).strip('[').split(' '))])

	plt.show()

if __name__ == '__main__':
	import datetime, argparse, project
	n = datetime.datetime.now()
	print(f'starting execution at {n.hour}:{n.minute}:{n.second}.')

	parser = argparse.ArgumentParser()
	parser.add_argument("question", choices=[method for method in dir(project) if method.startswith("Q")])

	args = parser.parse_args()

	getattr(project, args.question)()
