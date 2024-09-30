import pandas as pd

# Carregando o dataset do Wine Quality
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

# Visualizando as primeiras linhas do dataset
print(data.head())

# Informações básicas sobre os dados
print(data.info())

# Estatísticas descritivas dos dados
print(data.describe())

# Verificar se há valores ausentes
print(data.isnull().sum())

# Separando os atributos (X) e o alvo (y)
X = data.drop('quality', axis=1)
y = data['quality']

from sklearn.preprocessing import StandardScaler

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertendo a qualidade do vinho em binária (0 para ruim e 1 para bom)
y = y.apply(lambda x: 1 if x >= 7 else 0)

# Verificando a distribuição das classes
print(y.value_counts())

from sklearn.model_selection import train_test_split

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Treinando o modelo de Árvore de Decisão
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Fazendo previsões
y_pred_dt = dt_model.predict(X_test)

# Avaliando o desempenho
print("Acurácia da Árvore de Decisão:", accuracy_score(y_test, y_pred_dt))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_dt))

from sklearn.ensemble import RandomForestClassifier

# Treinando o modelo de Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Fazendo previsões
y_pred_rf = rf_model.predict(X_test)

# Avaliando o desempenho
print("Acurácia do Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_rf))

from sklearn.neighbors import KNeighborsClassifier

# Treinando o modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Fazendo previsões
y_pred_knn = knn_model.predict(X_test)

# Avaliando o desempenho
print("Acurácia do KNN:", accuracy_score(y_test, y_pred_knn))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_knn))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Matriz de confusão para Random Forest
cm = confusion_matrix(y_test, y_pred_rf)

# Visualizando a matriz de confusão
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Random Forest')
plt.show()
