# Importação de bibliotecas
import math
from numpy.random import randint
from numpy.random import rand
import random
import numpy as np
import pandas as pd
from perceptron_sklearn import perceptron
import time
import read_database

# typagem de python
from typing import List, Callable


# total de iterações
global n_iter
n_iter = 100

global hit_perc
hit_perc = 0.99
# crossover rate
global crossover_rate
crossover_rate = 0.6
# mutation rate
global mutation_rate
mutation_rate = 0.1
#tamanho da população
global population_size
population_size = 100
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

# Função objetivo
def objective(guess: List[bool], data: List[List[float]]) -> float:
    params = []

    for i in range(len(guess)):
        if guess[i] == True:
            params.append(i)

    # print(params)

    accuracy = perceptron(params, data)

    return accuracy



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
def termination_condition_met(best_eval: float, stop) -> bool:
        if stop >= 20:
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
                bitstring[i][j] = not bitstring[i][j]
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

# data = read_database.thyroidDisease()
# data = read_database.heartDisease()
# data = read_database.dermatologyDisease()
# data = read_database.BreastCancerWisconsinDisease()
data = read_database.Ionosphere()


params_size = len(data[0]) - 1

# População inicial
population = initialize_population(params_size)

# Avaliação inicial da população
fitness_scores = evaluate_population(population, data)
#parametros 
best_eval = max(fitness_scores)
params = population[fitness_scores.index(best_eval)]

def wrapper_AG(population,fitness_scores,All_accuracy_AG,All_params_AG):
    # Main loop
    stop = 0
    best_eval = max(fitness_scores)
    best_solution = population[fitness_scores.index(best_eval)]
    All_accuracy_AG = []
    All_params_AG = []
    while not termination_condition_met(best_eval, stop):
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

        current_eval = max(fitness_scores)
        All_accuracy_AG.append(current_eval)

        current_params = population[fitness_scores.index(current_eval)]

        params_current_solution = []
        params_old_solution = []
        for i in range(len(population[0])):
            if current_params[i] == True:
                params_current_solution.append(i)
            if best_solution[i] == True:
                params_old_solution.append(i)

        if current_eval >= best_eval:
            best_solution = current_params
            best_eval = current_eval

        All_params_AG.append(params_current_solution)
        print('Best: ',best_eval,'Iter: ',stop,'Len_best: ', len(params_old_solution), 'len_current: ',len(params_current_solution))
        stop += 1
    
    if current_eval >= best_eval:
        best_solution = current_params
        best_eval = current_eval

    return best_solution, best_eval, All_accuracy_AG,All_params_AG


def generate_neighbors(current_solution):
    neighbors = []
    for i in range(len(current_solution)):
        neighbor = current_solution.copy()
        neighbor[i] = not neighbor[i]  # Altera o valor do elemento na posição i
        neighbors.append(neighbor)
    return neighbors


def wrapper_hillClimbing(initial_solution: List[bool],best_eval: float,All_accuracy_Hill,All_params_Hill):
    current_solution = initial_solution
    best_solution =  initial_solution
    stop = 0
    All_accuracy_Hill = []
    All_params_Hill = []
    while not termination_condition_met(best_eval, stop):
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


        if best_neighbor_score >= objective(current_solution,data):
            current_solution = best_neighbor
            params_current_solution = params_best_neighbor
            best_eval = best_neighbor_score

        if  best_neighbor_score >= objective(best_solution,data):
            best_solution = best_neighbor
            best_eval = best_neighbor_score
        
        stop += 1
        All_accuracy_Hill.append(best_eval)
        All_params_Hill.append(params_current_solution)

        print('Best: ',best_eval,'Iter: ',stop,'Len_best: ', len(params_old_best_neighbor), 'len_current: ',len(params_current_solution))
        
    return best_solution, best_eval, All_accuracy_Hill,All_params_Hill

# start_time_AG = time.time()
# wrapper_AG_solution , wrapper_AG_acurracy = wrapper_AG(population,fitness_scores,params)
# end_time_AG = time.time()
# elapsed_time_AG = end_time_AG - start_time_AG 


# start_time_hill = time.time()
# wrapper_hillClimbing_solution,wrapper_hillClimbing_acurracy = wrapper_hillClimbing(params,best_eval,params)
# end_time__hill = time.time()
# elapsed_time_hill = end_time__hill - start_time_hill

# wrapper_AG_solution_index = []
# wrapper_hillClimbing_solution_index = []
# for i in range(len(wrapper_AG_solution)):
#     if wrapper_AG_solution[i] == True:
#         wrapper_AG_solution_index.append(i)
#     if wrapper_hillClimbing_solution[i] == True:
#         wrapper_hillClimbing_solution_index.append(i)


# print('\n')
# print("Melhor solução com o algortimo genético foi: ",wrapper_AG_solution_index ,':',len(wrapper_AG_solution_index),"com: ", wrapper_AG_acurracy)
# print("\nDemorando: ",elapsed_time_AG,"segundos")

# print('\n')

# print("Melhor solução com o algortimo Subida da encosta foi: ",wrapper_hillClimbing_solution_index ,':',len(wrapper_hillClimbing_solution_index),"com: ", wrapper_hillClimbing_acurracy)
# print("\nDemorando: ",elapsed_time_hill,"segundos")


def api():
    global All_accuracy_AG, All_accuracy_Hill,All_params_AG,All_params_Hill
    All_accuracy_AG = []
    All_accuracy_Hill = []
    All_params_AG = []
    All_params_Hill = []
    
    start_time_AG = time.time()

    wrapper_AG_solution , wrapper_AG_acurracy, All_accuracy_AG, All_params_AG = wrapper_AG(population,fitness_scores,All_accuracy_AG,All_params_AG)

    end_time_AG = time.time()
    elapsed_time_AG = end_time_AG - start_time_AG 

    start_time_hill = time.time()

    wrapper_hillClimbing_solution,wrapper_hillClimbing_acurracy,All_accuracy_Hill,All_params_Hill = wrapper_hillClimbing(params,best_eval,All_accuracy_Hill,All_params_Hill)
    
    end_time__hill = time.time()
    elapsed_time_hill = end_time__hill - start_time_hill

    return wrapper_AG_solution , wrapper_AG_acurracy, elapsed_time_AG, wrapper_hillClimbing_solution,wrapper_hillClimbing_acurracy, elapsed_time_hill,All_accuracy_AG, All_accuracy_Hill,All_params_AG,All_params_Hill
