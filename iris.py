import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

dados = pd.read_csv("iris.csv")


fig = plt.figure(figsize=(15,8))
eixo = fig.add_axes([0,0,1,1])

cores = {'Iris-setosa': 'r', 'Iris-versicolor': 'b', 'Iris-virginica': 'g'}
marcadores = {'Iris-setosa': 'x', 'Iris-versicolor': 'o', 'Iris-virginica': 'v'}

for especie in dados['espécie'].unique():
    tmp = dados[dados['espécie'] == especie]
    eixo.scatter(tmp['comprimento_sépala'], tmp['largura_sépala'],
                 color=cores[especie], marker=marcadores[especie],
                 s=100)

eixo.set_title('Gráfico de dispersão', fontsize=25, pad=15)
eixo.set_xlabel('Comprimento da sépala', fontsize=15)
eixo.set_ylabel('Largura da sépala', fontsize=15)
eixo.tick_params(labelsize=15)
eixo.legend(cores, fontsize=20)
plt.show()


y_numpy = []
for i in np.array(dados["espécie"]):
    if i == "Iris-setosa":
        y_numpy.append(0)
    elif i == "Iris-versicolor":
        y_numpy.append(1)
    elif i == "Iris-virginica":
        y_numpy.append(2)

y_numpy = np.array(y_numpy)
x_numpy = np.array(dados[["comprimento_sépala","largura_sépala","comprimento_pétala","largura_pétala"]])

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.long()
x = x.view(x.shape[0], 4)

print(x.shape)
print(y.shape)
print(y)


class RegressaoSoftmax(nn.Module):
    def __init__(self, n_input, n_output):
        super(RegressaoSoftmax, self).__init__()
        self.Linear = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.Linear(x)


# DEFINICIÇÃO DE MODELO
input_size = 4
output_size = 3
model = RegressaoSoftmax(input_size, output_size)

# DEFINIÇÃO DA FUNÇAO DE CUSTO E OTIMIZADOR
learning_rate = 0.05
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# LOOP DE TREINAMENTO
num_epochs = 1000
contador_custo = []
for epoch in range(num_epochs):
    # forward pass and loos
    y_hat = model(x)
    loss = criterion(y_hat, y)
    contador_custo.append(loss)
    # print(y_hat)

    # backward pass (calcular gradientes)
    loss.backward()

    # update (atualizar os pesos)
    optimizer.step()

    # limpar o otimizador
    optimizer.zero_grad()

# PLOTANDO O GRÁFICO DA FUNÇÃO DE CUSTO
print("GRÁFICO DA FUNÇÃO DE CUSTO")
contador_custo = torch.tensor(contador_custo, requires_grad=True)
contador_custo = contador_custo.detach().numpy()
plt.plot(contador_custo, 'b')
plt.show()

"""#Fazer a predição"""

# fazer predição de teste
teste = np.array([[9.8,5.8,5.8,0.2],[1.1,2.8,2.8,2.8],[5.1,4.1,4.5,7.8],[2.4,2.4,1.4,0.3]])
t_teste = torch.from_numpy(teste.astype(np.float32))
t_teste = t_teste.view(t_teste.shape[0], 4)

with torch.no_grad():
    predicoes = model(t_teste)
    print(np.argmax(predicoes, axis=1).flatten())