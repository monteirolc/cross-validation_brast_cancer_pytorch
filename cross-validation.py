import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import cross_val_score

# Carregamento da base de dados
np.random.seed(123)
torch.manual_seed(123)

# carregamento dos arquivos da base de dados
previsores = pd.read_csv('data/entradas_breast.csv')
classe = pd.read_csv('data/saidas_breast.csv')

# plot do grafico com seaborn
sns.countplot(classe['0'])

# passando do formato pandas para numpy
previsores = np.array(previsores, dtype=np.float32)
classe = np.array(classe, dtype=np.float32).squeeze(1)


# class Classificador_torch(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 30 -> 16 -> 16 -> 1
#         self.dense0 = nn.Linear(30, 16)
#         torch.nn.init.uniform(self.dense0.weight)
#         self.activation0 = nn.ReLU()
#         self.dense1 = nn.Linear(16, 16)
#         torch.nn.init.uniform(self.dense1.weight)
#         self.activation1 = nn.ReLU()
#         self.dense2 = nn.Linear(16, 1)
#         torch.nn.init.uniform(self.dense2.weight)
#         self.output = nn.Sigmoid()

#     def forward(self, x):
#         x = self.dense0(x)
#         x = self.activation0(x)
#         x = self.dense1(x)
#         x = self.activation1(x)
#         x = self.dense2(x)
#         x = self.output(x)
#         return x


''' Usando Skorch '''

# classficador_sklearn = NeuralNetBinaryClassifier(
#     module=Classificador_torch,
#     criterion=torch.nn.BCELoss,
#     optimizer=torch.optim.Adam,
#     lr=0.001, optimizer__weight_decay=0.0001,
#     max_epochs=100,
#     batch_size=10,
#     train_split=False)

''' Validação Cruzada '''
# resultados = cross_val_score(
#     classficador_sklearn,
#     previsores,
#     classe,
#     cv=10,
#     scoring='accuracy')

# media = resultados.mean()  # média
# desvio = resultados.std()  # desvio padrão

''' Dropout '''


class Classificador_torch(nn.Module):
    def __init__(self):
        super().__init__()
        # 30 -> 16 -> 16 -> 1
        self.dense0 = nn.Linear(30, 16)
        torch.nn.init.uniform(self.dense0.weight)
        self.activation0 = nn.ReLU()
        self.dropout0 = nn.Dropout(0.2)  # Dropout
        self.dense1 = nn.Linear(16, 16)
        torch.nn.init.uniform(self.dense1.weight)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)  # Dropout
        self.dense2 = nn.Linear(16, 1)
        torch.nn.init.uniform(self.dense2.weight)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.dense0(x)
        x = self.activation0(x)
        x = self.dropout0(x)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.output(x)
        return x


classficador_sklearn = NeuralNetBinaryClassifier(
    module=Classificador_torch,
    criterion=torch.nn.BCELoss,
    optimizer=torch.optim.Adam,
    lr=0.001, optimizer__weight_decay=0.0001,
    max_epochs=100,
    batch_size=10,
    train_split=False)

resultados = cross_val_score(
    classficador_sklearn,
    previsores,
    classe,
    cv=10,
    scoring='accuracy')

media = resultados.mean()  # média
desvio = resultados.std()  # desvio padrão
