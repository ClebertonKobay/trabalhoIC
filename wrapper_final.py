# Importação de bibliotecas
import math
from numpy.random import randint
from numpy.random import rand
import random
import numpy as np
import pandas as pd
from perceptron import perceptron

# typagem de python
from typing import List, Callable


# total de iterações
global n_iter
n_iter = 100

global hit_perc
hit_perc = 0.99
# bits per variable
global n_bits
n_bits = 10
# crossover rate
global crossover_rate
crossover_rate = 0.9
# mutation rate
global mutation_rate
mutation_rate = 1.0 / float(n_bits)
#tamanho da população
global population_size
population_size = 100
#Numero de parêmtros
# n_parameters = 29
#taxa de seleção dos melhores entre a população
global elitism_rate
elitism_rate = 0.3
#indice onde está a resposta do teste
global index_expected
index_expected = 0
#melhor resultado anterior
global old_best_eval
old_best_eval = 0
#melhores parametros até o momento
global best_params
best_params = []
best_params = np.zeros(30)


def fn_calculate(data: List[List[any]])-> None:
    for i in range(len(data)):
        for j in range(len(data[i])):
            match data[i][j]:
                case 'M':
                    data[i][j] = 0.0
                case 'F': 
                    data[i][j] = 1.0
                case 't':
                    data[i][j] = 1.0
                case 'f':
                    data[i][j] = 0.0
                case '?':
                    data[i][j] = 0.0
                case 'other':
                    data[i][j] = 1.0
                case 'SVI':
                    data[i][j] = 2.0
                case 'SVHC':
                    data[i][j] = 3.0
                case 'WEST':
                    data[i][j] = 4.0
                case 'SVHD':
                    data[i][j] = 5.0
                case 'STMW':
                    data[i][j] = 6.0
                case 'sick.':
                    data[i][j] = 1.0
                case 'negative.':
                    data[i][j] = 0.0
                case _:
                    if data[i][j].isnumeric():
                        data[i][j] = float(data[i][j])
                    else:
                        data[i][j] = 0.0

    return data

# Função objetivo
def objective(guess: List[bool], data: List[List[float]], k=29) -> float:
    params = []

    for i in range(len(guess)):
        if guess[i] == True:
            params.append(i)

    # print(params, k)

    x = perceptron(params, data, k)
    return x.treinar(data, 100, 1.0, False)
      


def evaluate_population(population: List[List[bool]], data: List[List[float]]) -> List[float]:
    scores = []
    for guess in population:
        # Avaliar o fitness do wrapper para cada guess na população
        fitness = objective(guess, data)  # Adaptar o índice conforme necessário
        scores.append(fitness)
    return scores


# Cria uma população inicial aleatória
def initialize_population(params_size: int) -> List[List[bool]]:
    return [[random.choice([True, False]) for _ in range(params_size)] for _ in range(population_size)]
    

# Dá para criar um array
# Anota o melhor usando "k" parametros
def termination_condition_met(best_eval: float, stop, pop) -> bool:
        global best_params, old_best_eval
        params = []
        for i in range(len(pop)):
            if pop[i] == True:
                params.append(i)

        if len(params) < len(best_params) and  best_eval >= old_best_eval:
            best_params = params.copy()
            old_best_eval = best_eval
            print("Params: ",best_params,"Eval: ",old_best_eval)
        if stop > 20:
            return True
        return False

# Seleção por torneio
def tournament(pop: List[List[bool]], scores: List[float], k: int = 2) -> List[List[bool]]:
    tournament_pop = []
    for _ in range(len(pop)):
        selected_ix = randint(len(pop))
        for ix in randint(0, len(pop), k):
            if scores[ix] > scores[selected_ix]:
                selected_ix = ix
        tournament_pop.append(pop[selected_ix])
    return tournament_pop


# Seleção por roleta
def roulette(pop: List[List[bool]], scores: List[float]) -> List[bool]:
    pop_fitness = sum(scores)
    chromo_prob = [scores[x]/pop_fitness for x in range(len(pop))]
    decoded = [p for p in range(len(pop))]
    choice = np.random.choice(decoded, p=chromo_prob)

    return pop[choice]


# crossover entre dois individuos para criar um filho (aqui tá 100%)
def crossover(population: List[List[bool]], r_cross: float) -> List[List[bool]]:
    pop = []

    for _ in range(int(len(population)/2)):
        i = randint(0, len(population))
        j = randint(0, len(population))
        # Se ocorrer a cruzamento
        if rand() < r_cross:
            # Seleciona o ponto de 
            pt = randint(1, len(population)-2)
            # perform crossover
            c1 = population[i][:pt] + population[j][pt:]
            c2 = population[j][:pt] + population[i][pt:]
            pop.append(c1)
            pop.append(c2)
        else:
            c1, c2 = population[i].copy(), population[j].copy()
            pop.append(c1)
            pop.append(c2)

    return pop


def mutation(bitstring: List[List[bool]], r_mut: float) -> List[List[bool]]:
    for i in range(len(bitstring)):
        for j in range(len(bitstring[i])):
            if rand() < r_mut:
                # verificar o tipo de dado entrado
                bitstring[i][j] = ~bitstring[i][j]
    return bitstring


def elitism(parents: List[List[bool]], fitness_scores: List[float], offspring: List[List[bool]],
            elitism_rate: float) -> List[List[bool]]:
    num_elites = int(elitism_rate * len(parents))
    elites = [(parent, fitness) for parent, fitness in zip(parents, fitness_scores)]
    elites.sort(key=lambda x: x[1], reverse=True)
    offspring.sort(key=lambda x:x[1])
    elites = elites[:num_elites]
    elites = [elite[0] for elite in elites]
    del offspring[:len(elites)]
    offspring.extend(elites)
    return offspring


file = open(".\\thyroidDisease\\sick.data", 'r')
row = file.read()
file.close()

data = row.split('\n')
info_data = []

for i in range(len(data)):
    info_data = data[i].split('|')
    data[i] = info_data[0].split(',')

fn_calculate(data)

params_size = len(data[0]) - 1

# População inicial
population = initialize_population(params_size)

# Avaliação inicial da população
fitness_scores = evaluate_population(population, data)
#parametros 
best_eval = max(fitness_scores)
params = population[fitness_scores.index(best_eval)]

def wrapper_AG(population,fitness_scores,params):
    # Main loop
    stop = 0
    best_eval = max(fitness_scores)
    while not termination_condition_met(best_eval, stop, params):
        # Seleciona os melhores filhos com base na sua pontuação
        parents = tournament(population, fitness_scores, 3)
        # Pega a criação dos filhos com base na seleção
        offspring = crossover(parents, crossover_rate)
        # Muta os filhos com base no mutation_rate
        offspring = mutation(offspring, mutation_rate)

        population = elitism(parents, fitness_scores, offspring, elitism_rate)

        # por elitismo
        fitness_scores = evaluate_population(population, data)

        # population = hill_climbing(population, fitness_scores, data)

        best_eval = max(fitness_scores)
        params = population[fitness_scores.index(best_eval)]
        print('Best: ',best_eval,'Iter: ',stop)
        stop += 1

    best_solution = max(fitness_scores)

    return best_solution


def generate_neighbors(current_solution):
    neighbors = []
    for i in range(len(current_solution)):
        neighbor = current_solution.copy()
        neighbor[i] = not neighbor[i]  # Altera o valor do elemento na posição i
        neighbors.append(neighbor)
    return neighbors


def wrapper_hillClimbing(initial_solution: List[bool],best_eval: float,params: List[bool]):
    current_solution = initial_solution
    best_solution =  initial_solution
    stop = 0
    while not termination_condition_met(best_eval, stop, params):
        neighbors = generate_neighbors(current_solution)
        neighbor_scores = evaluate_population(neighbors,data)
        best_neighbor_score = max(neighbor_scores)
        best_neighbor =  neighbors[neighbor_scores.index(best_neighbor_score)]

        params_current_solution = []
        params_best_neighbor = []
        params_old_best_neighbor = []
        for i in range(len(current_solution)):
            if current_solution[i] == True:
                params_current_solution.append(i)
            if best_neighbor[i] == True:
                params_best_neighbor.append(i)
            if best_solution[i] == True:
                params_old_best_neighbor.append(i)


        if len(params_best_neighbor) < len(params_current_solution) and best_neighbor_score >= objective(current_solution,data):
            current_solution = best_neighbor

        if len(params_best_neighbor) < len(params_old_best_neighbor) and best_neighbor_score >= objective(best_solution,data):
            best_solution = best_neighbor
        
        stop += 1
        best_eval = max(neighbor_scores)
        params = current_solution
    return best_solution


wrapper_AG_solution = wrapper_AG(population,fitness_scores,params)
wrapper_hillClimbing_solution = wrapper_hillClimbing(params,best_eval,params)