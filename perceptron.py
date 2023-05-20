import random

# Typing para facilitar leitura do código
from typing import List, Callable

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
                    data[i][j] = 0.0
                case 'SVI':
                    data[i][j] = 1.0
                case 'SVHC':
                    data[i][j] = 2.0
                case 'WEST':
                    data[i][j] = 3.0
                case 'SVHD':
                    data[i][j] = 4.0
                case 'STMW':
                    data[i][j] = 5.0
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


class perceptron:
    # Uma lista que fala quais parametros vai entrar na conta (os seus indices)
    parameters: List[int] = []
    # Pesos das entradas de cada parametro
    weight: List[float] = []
    # Constante que não depende da entrada
    bias: float

    # Anota o indice do data onde a resposta desse teste está posicionado
    answ_index: int

    # Anota os dados já calculados para poder usar no least_squares para atualizar os pesos
    old_tests: List[List[bool]]

    # Conta quantos deram a resposta certa
    hit_qtd: int

    # Inicializa o perceptron informando quais parametros serão utilizando, e inicializando eles com dados aleatórios
    def __init__(self, params: List[int], data: List[List[float]], answ_index: int):
        self.parameters = params
        self.answ_index = answ_index

        self.old_tests = []
        self.weight = []
        # bias \in [-0.5, 0.5)
        self.bias = random.random() - 0.5
        for _ in range(len(params)):
            # weight[i] \in [-0.5, 0.5)
            self.weight.append(random.random() - 0.5)
    

    # Função para interpretar os dados, no caso é uma função de Heaviside
    def activation_function(self, sum: float) -> bool:
        if sum >= 0.0:
            return True
        else:
            return False
        

    # Vai ver qual a resposta do perceptron (dada a entrada) e compara se bate com a resposta esperada
    def eval(self, input: List[float], answer: bool) -> bool:
        # Inicia com o valor constante
        sum: float = self.bias
        
        # Calcula a soma total
        for i in range(len(self.parameters)):
            sum += self.weight[i] * input[self.parameters[i]]

        # Anota o valor calculado
        predict = self.activation_function(sum)
        self.old_tests.append([predict, answer])
        
        # Retorna se acertou ou não para ver a taxa de acerto
        return predict == answer
    

    step_size: float = 0.001

    # Vai atualizar os weight e bias com objetivo de minimizar o erro / maximizar o acerto
    def update_coef(self, data: List[List[float]]) -> None:
        for w in range(len(self.weight)):
            for i in range(len(self.old_tests)):
                self.weight[w] -= self.step_size * (data[i][self.parameters[w]]) * (self.old_tests[i][0] - self.old_tests[i][1])
            
        for i in range(len(self.old_tests)):
            self.bias -= self.step_size * (self.old_tests[i][0] - self.old_tests[i][1])

    # Executa as funções para fazer as predições e atualização dos pesos
    def treinar(self, data: List[List[float]], n_iter: int, perc: float, p: bool) -> float:
        for _ in range(n_iter):
            self.old_tests = []
            self.hit_qtd = 0
            for d in data:
                self.hit_qtd += self.eval(d, d[self.answ_index])
            hit_ratio = self.hit_qtd/len(data)
            
            if hit_ratio >= perc:
                # print('Chegou a', perc * 100.0, '% pedida. Atingiu ', hit_ratio * 100.0, '%')
                break

            if p:
                print(self.weight, self.bias, ' -> ', hit_ratio)
            self.update_coef(data)

        # print(self.weight, self.bias, ' -> ', hit_ratio)
        return hit_ratio
