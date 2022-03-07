import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import GridSearchCV

# Carregamento da base de dados e experimetnos
np.random.seed(123)
torch.manual_seed(123)

# carregamento dos arquivos da base de dados
previsores = pd.read_csv('data/entradas_breast.csv')
classe = pd.read_csv('data/saidas_breast.csv')

# passando do formato pandas para numpy
previsores = np.array(previsores, dtype=np.float32)
classe = np.array(classe, dtype=np.float32).squeeze(1)


class Classificador_torch(nn.Module):
    def __init__(self, activation, neurons, initializer):
        super().__init__()
        # 30 -> 16 -> 16 -> 1
        self.dense0 = nn.Linear(30, neurons)
        initializer(self.dense0.weight)
        self.activation0 = activation
        self.dense1 = nn.Linear(neurons, neurons)
        initializer(self.dense1.weight)
        self.activation1 = activation
        self.dense2 = nn.Linear(neurons, 1)
        initializer(self.dense2.weight)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.dense0(x)
        x = self.activation0(x)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dense2(x)
        x = self.output(x)
        return x


classficador_sklearn = NeuralNetBinaryClassifier(
    module=Classificador_torch,
    lr=0.001, optimizer__weight_decay=0.0001,
    train_split=False)

''' Tunning dos parâmetros '''
params = {
    'batch_size': [10, 30],
    'max_epochs': [100, 200],
    'optimizer': [torch.optim.Adam, torch.optim.SGD],
    'criterion': [torch.nn.BCELoss, torch.nn.HingeEmbeddingLoss],
    'module__activation': [F.relu, F.tanh],
    'module__neurons': [8, 16],
    'module__initializer': [torch.nn.init.uniform, torch.nn.init.normal]}

grid_search = GridSearchCV(
    estimator=classficador_sklearn,
    param_grid=params,
    scoring='accuracy',
    cv=2).fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
