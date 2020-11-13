import numpy as np
np.random.seed(7)
import sys


def func_1(x=None):
	# x \in [-1 ~ 1], min(f) = 1 when x = 0
	return 2 * x ** 2 + 1


def func_2(x=None):
	#return np.sin(10 * x) * x + np.cos(2 * x) * x
	return 3 * x ** 2 + 3


def encoding_dna(population=None, dna_size=None,
					x_low=None, x_high=None):
	# population: [pop_size, dna_size]
	normalize = np.dot(population, 2 ** np.arange(dna_size)[::-1]) / (2 ** dna_size - 1)

	# (x - x_low) / (x_high - x_low) = y
	# ==> x = y * (x_high - x_low) + x_low
	return normalize * (x_high - x_low) + x_low


def get_fitness(values=None):
	# Find non-zero fitness for selection.
	# The less of values, the better.
	# The less of values, the bigger of fitness.
	return np.max(values) - values + 1e-3


def select(population=None, fitness=None):
	# The bigger of fitness, the bigger probability to be selected by nature.
	population_size = population.shape[0]
	idx = np.random.choice(np.arange(population_size),
						   size=population_size,
						   replace=True,
						   p=fitness/fitness.sum())
	return population[idx]


def crossover(person=None, population=None, crossover_rate=None):
	# person: [dna_size, ]
	# population: [population_size, dna_size]
	population_size = population.shape[0]
	dna_size = population.shape[1]

	if np.random.rand() < crossover_rate:
		# select one to mate
		mate_id = np.random.randint(0, population_size, size=1)

		# choose points to cross
		cross_points = np.random.randint(0, 2, size=dna_size).astype(np.bool)

		# mate, produce child
		person[cross_points] = population[mate_id, cross_points]

	child = person

	return child


def mutate(child=None, mutation_rate=None):
	# child: [dna_size, ]
	dna_size = child.shape[0]
	for point in range(dna_size):
		if np.random.rand() < mutation_rate:
			#child[point] = 1 if child[point] == 0 else 0
			child[point] ^= 1
	return child


def get_optimization(x=None, fitness=None):
	min_idx = np.argmax(fitness)
	min_x = x[min_idx]
	return min_x


if __name__ == "__main__":
	n_generation = 200
	dna_size = 25
	population_size = 100

	crossover_rate = 0.8
	mutation_rate = 0.003

	x_low, x_high = -1, 1
	#x = np.linspace(start=-1, stop=1, num=population_size)

	# === Init population, random init.
	population = np.random.randint(low=0, high=2,
				size=(population_size, dna_size))

	func = func_1

	max_fitness_value = -1
	min_x = None
	min_f = None
	best_iter = None

	for iter in range(n_generation):
		inputs = encoding_dna(population=population, dna_size=dna_size,
				x_low=x_low, x_high=x_high)

		f_values = func(inputs)
		fitness = get_fitness(values=f_values)

		cur_best_idx = np.argmax(fitness)
		cur_max_fitness_value = fitness[cur_best_idx]
		cur_min_x = inputs[cur_best_idx]
		cur_min_f = func(cur_min_x)

		if max_fitness_value < cur_max_fitness_value:
			max_fitness_value = cur_max_fitness_value
			min_x = cur_min_x
			min_f = cur_min_f
			best_iter = iter

		population = select(population=population,
				fitness=fitness)

		pop_copy = population.copy()

		for person in population:
			child = crossover(person=person,
							  population=pop_copy,
							  crossover_rate=crossover_rate)
			child = mutate(child=child, mutation_rate=mutation_rate)
			person[:] = child # for iteration

	print("Best_iter=%d, min_x=%.4f, min_f=%.4f, max_fitness_value=%.4f" %
		  (best_iter, min_x, min_f, max_fitness_value))