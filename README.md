# Projeto de Aprendizado Supervisionado com Python

## Descrição

Este projeto tem como objetivo aplicar técnicas de aprendizado supervisionado utilizando Python. Foi implementado um modelo de classificação para prever a qualidade do vinho com base em atributos físico-químicos, utilizando o **Wine Quality Dataset**.

Foram explorados três modelos de aprendizado supervisionado: Árvore de Decisão, Random Forest e K-Nearest Neighbors (KNN). O projeto foi estruturado desde a análise e pré-processamento dos dados até o treinamento, validação e avaliação dos modelos.

## Dataset

O dataset utilizado é o **Wine Quality Dataset**, disponível no [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). O dataset contém 1.599 amostras de vinho tinto, cada uma com 11 atributos físico-químicos e uma classificação de qualidade (variável alvo).

### Atributos:

- **fixed acidity**
- **volatile acidity**
- **citric acid**
- **residual sugar**
- **chlorides**
- **free sulfur dioxide**
- **total sulfur dioxide**
- **density**
- **pH**
- **sulphates**
- **alcohol**
- **quality** (variável alvo)

## Modelos Utilizados

Três modelos de classificação foram testados:

1. **Árvore de Decisão**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**

Os modelos foram avaliados usando métricas como Acurácia, Precisão, Recall e F1-Score.

## Estrutura do Projeto

- **data/**: Contém o dataset utilizado no projeto.
- **notebooks/**: Notebooks Jupyter com o código utilizado para análise, treinamento e avaliação dos modelos.
- **scripts/**: Scripts Python para execução do pré-processamento e treinamento dos modelos.
- **README.md**: Explicação do projeto.
  
## Pré-requisitos

Antes de executar o projeto, certifique-se de que as seguintes dependências estão instaladas:

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

Você pode instalar as dependências necessárias com o seguinte comando:

```bash
pip install -r requirements.txt
