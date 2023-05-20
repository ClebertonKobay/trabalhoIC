
"""
def attribute_selection_wrapper(feature_set, population_size, crossover_rate, mutation_rate, selection_method, elitism_rate):
    # Inicialização da população
    population = initialize_population(feature_set, population_size)

    # Avaliação inicial da população
    fitness_scores = evaluate_population(population, feature_set)

    # Main loop
    while not termination_condition_met():
        # Seleção de pais
        parents = select_parents(population, fitness_scores, selection_method)

        # Recombinação (crossover)
        offspring = crossover(parents, crossover_rate)

        # Mutação
        offspring = mutate(offspring, mutation_rate)

        # Avaliação da nova população
        offspring_fitness_scores = evaluate_population(offspring, feature_set)

        # Elitismo: preserva os melhores indivíduos da geração anterior
        population = elitism(population, fitness_scores, offspring, offspring_fitness_scores, elitism_rate)

        # Atualização dos valores de aptidão
        fitness_scores = evaluate_population(population, feature_set)

        # Hill-climbing: busca local para aprimorar soluções individuais
        population = hill_climbing(population, fitness_scores, feature_set)

    # Retornar a melhor solução encontrada
    best_solution = get_best_solution(population, fitness_scores)

    return best_solution
"""

# Importação de bibliotecas
import math
from numpy.random import randint
from numpy.random import rand
import random
import numpy as np
from perceptron import perceptron 

# typagem de python
from typing import List, Callable


# Seleção por torneio
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] > scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]


# Seleção por roleta
def roulette(pop: List[List[bool]], scores: List[float]) -> List[bool]:
    pop_fitness = sum(scores)
    chromo_prob = [scores[x]/pop_fitness for x in range(len(pop))]
    decoded = [p for p in range(len(pop))]
    choice = np.random.choice(decoded, p=chromo_prob)

    return pop[choice]

"""
Aqui tá suave, não precisa fazer modificação de código
"""
# crossover two parents to create two children
def crossover(p1: List[bool], p2: List[bool], r_cross: float) -> List[bool]:
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]


def mutation(bitstring: List[bool], r_mut: float):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# verificar o tipo de dado entrado
			bitstring[i] = ~bitstring[i]


def createPop(population_size,data_size):
	pop_list = []
	for _ in range(population_size):
		pop = [random.choice([True, False]) for _ in range(data_size)]
		pop_list.append(pop)

	return pop_list


def score():
    
    return

def genetic_algorithm(objective: Callable[[List[bool], List[List[any]], int], bool], n_parameters: int, _iter: Callable[[float], bool], n_pop, pop: List[List[bool]], r_cross, r_mut, select):
    gen = 0
    while _iter(best):
        gen += 1
        # criar o perceptron e avaliar ele (% de acerto e qtd. de parametros)
        
        # avaliar o score de cada perceptron
        scores = score(pop)
        # pegar a melhor solução (balancear entre % de acerto e qtd. de parametros)
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %f" % (gen, scores[i]))
        # select parents
        selected = [select(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children