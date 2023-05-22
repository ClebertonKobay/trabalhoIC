from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# typagem de python
from typing import List, Callable

    # Carregue seus dados e faça o pré-processamento se necessário
    # data = ...
def perceptron(parametros:List[int],data: List[List[bool]]):
    # Separe os atributos (X) e os rótulos (y)
    data = np.array(data)
    X = data[:, :-1] # Atributos (todas as colunas, exceto a última)
    y = data[:, -1] # Rótulos (última coluna) # Rótulos na coluna 29
    X_parametros = X[:, parametros]

    learning_rate = 0.001  # Taxa de aprendizado
    max_iter = 500  # Número máximo de iterações

    # Divida os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_parametros, y, test_size=0.3,random_state=25)

    # Inicialize o perceptron usando a biblioteca scikit-learn
    perceptron_model = Perceptron(eta0=learning_rate,max_iter=max_iter, random_state=25)
    
    # Treine o perceptron
    perceptron_model.fit(X_train, y_train)

    # Faça previsões nos dados de teste
    y_pred = perceptron_model.predict(X_test)

    # Calcule a acurácia das previsões
    accuracy = accuracy_score(y_test, y_pred)
    # print("Taxa de acerto:", accuracy)

    return accuracy