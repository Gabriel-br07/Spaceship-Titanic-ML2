import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# Carregando os dados de treinamento e teste
treino = pd.read_csv("train.csv")
teste = pd.read_csv("test.csv")
test_id = teste["PassengerId"]

# Usando o LabelEncoder para codificar a coluna "Transported"
le = LabelEncoder()
treino["Transported"] = le.fit_transform(treino["Transported"])

# Configurando opções de exibição do pandas
pd.set_option("display.max_columns", None)


# Função para limpar os dados
def limpar(dados):
    # Dividindo a coluna "Cabin" em três novas colunas: "deck", "num" e "side"
    dados[["deck", "num", "side"]] = dados["Cabin"].str.split('/', expand=True)

    # Removendo colunas não necessárias
    dados = dados.drop(["Cabin", "Name", "PassengerId"], axis=1)

    # Usando o LabelEncoder para codificar várias colunas categóricas
    colunas_categoricas = ["CryoSleep", "VIP", "deck", "side", "HomePlanet", "Destination"]
    for coluna in colunas_categoricas:
        dados[coluna] = le.fit_transform(dados[coluna])

    # Preenchendo valores nulos com a mediana das respectivas colunas
    colunas_numericas = ['Age', 'num', 'HomePlanet', 'Destination', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
                         'VRDeck']
    for coluna in colunas_numericas:
        media_coluna = dados[coluna].median()
        dados[coluna].fillna(media_coluna, inplace=True)

    return dados


# Aplicando a função de limpeza aos dados de treinamento e teste
treino = limpar(treino)
teste = limpar(teste)

# Dividindo os dados de treinamento em features (x) e target (y)
x = treino.drop(["Transported"], axis=1)
y = treino["Transported"]

# Definindo uma semente aleatória para reproducibilidade
SEED = 311
np.random.seed(SEED)

# Criando um modelo RandomForestClassifier
modelo = RandomForestClassifier()

# Realizando validação cruzada
resultado = cross_validate(modelo, x, y, cv=10, return_train_score=False)

# Treinando o modelo com todos os dados de treinamento
modelo.fit(x, y)

# Realizando previsões nos dados de teste
resultado_final = modelo.predict(teste)

# Convertendo os resultados para True (1) ou False (0)
resultado_convertido = [True if valor == 1 else False for valor in resultado_final]

# Criando um DataFrame com os resultados e salvando em um arquivo CSV
df = pd.DataFrame({"PassengerId": test_id.values, "Transported": resultado_convertido})
df.to_csv("resultado.csv", index=False)
