# Importação de bibliotecas
import math
from numpy.random import randint
from numpy.random import rand
import random
import numpy as np
import pandas as pd
from perceptron_sklearn import perceptron
import time

# typagem de python
from typing import List, Callable

def thyroidDisease()-> List[List[float]]:
    file = open(".\\thyroidDisease\\sick.data", 'r')
    row = file.read()
    file.close()

    data = row.split('\n')
    info_data = []

    for i in range(len(data)):
        info_data = data[i].split('|')
        data[i] = info_data[0].split(',')
    
    evalThyroid(data)
    return data

def evalThyroid(data: List[List[any]]) -> None:
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

def heartDisease() -> List[List[float]]:
    file = open(".\\heartDisease\\processed.va.data", 'r')
    row = file.read()
    file.close()

    data = row.split('\n')
    for i in range(len(data)):
        data[i] = data[i].split(',')
    
    evalHeart(data)

    return data

def evalHeart(data: List[List[any]])-> None:
    for i in range(len(data)):
        for j in range(len(data[i])):
            match data[i][j]:
                case '?':
                    data[i][j] = 0.0
                case _:
                    if data[i][j].isnumeric():
                        data[i][j] = float(data[i][j])
                    else:
                        data[i][j] = 0.0

def dermatologyDisease() -> List[List[float]]:
    file = open(".\\dermatologyDisease\\dermatology.data", 'r')
    row = file.read()
    file.close()

    data = row.split('\n')
    for i in range(len(data)):
        data[i] = data[i].split(',')
    
    evalDermatology(data)

    return data

def evalDermatology(data: List[List[any]])-> None:
    for i in range(len(data)):
        for j in range(len(data[i])):
            match data[i][j]:
                case '?':
                    data[i][j] = 0.0
                case _:
                    if data[i][j].isnumeric():
                        data[i][j] = float(data[i][j])
                    else:
                        data[i][j] = 0.0



def BreastCancerWisconsinDisease() -> List[List[float]]:
    file = open(".\\BreastCancerWisconsin\\wdbc.data", 'r')
    row = file.read()
    file.close()

    data = row.split('\n')
    for i in range(len(data)):
        data[i] = data[i].split(',')
    
    evalBreastCancerWisconsin(data)

    return data

def evalBreastCancerWisconsin(data: List[List[any]])-> None:
    for i in range(len(data)):
        for j in range(len(data[i])):
            match data[i][j]:
                case '?':
                    data[i][j] = 0.0
                case 'M':
                    data[i][j] = 1.0
                case 'B':
                    data[i][j] = 2.0
                case _:
                    if data[i][j].isnumeric():
                        data[i][j] = float(data[i][j])
                    else:
                        data[i][j] = 0.0
    
def Ionosphere() -> List[List[float]]:
    file = open(".\\ionosphere\\ionosphere.data", 'r')
    row = file.read()
    file.close()

    data = row.split('\n')
    for i in range(len(data)):
        data[i] = data[i].split(',')
    
    evalIonosphere(data)

    return data

def evalIonosphere(data: List[List[any]])-> None:
    for i in range(len(data)):
        for j in range(len(data[i])):
            match data[i][j]:
                case '?':
                    data[i][j] = 0.0
                case 'g':
                    data[i][j] = 1.0
                case 'b':
                    data[i][j] = 0.0
                case _:
                    if data[i][j].isnumeric():
                        data[i][j] = float(data[i][j])
                    else:
                        data[i][j] = 0.0