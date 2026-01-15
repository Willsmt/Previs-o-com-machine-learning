

---

# 📊 Análise de Crédito de Clientes

Este projeto utiliza **Python**, **Pandas** e **Scikit-learn** para analisar clientes e prever se o **score de crédito** é **bom, ok ou ruim**.  
A ideia é treinar modelos de inteligência artificial com dados históricos e aplicar em novos clientes.

---

## 🚀 Tecnologias utilizadas
- [Python 3](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)

---

## 📌 Passo a passo

### 🔹 Passo 0: Entender o desafio
Prever a coluna **`score_credito`** (bom, ok ou ruim) a partir das variáveis:

---

### 🔹 Passo 1: Importar a base de dados
```python
import pandas as pd

tabela = pd.read_csv("clientes.csv")
display(tabela)
display(tabela.info())
```

---

### 🔹 Passo 2: Explorar e preparar os dados
Transformar colunas de texto em números usando **LabelEncoder**:
```python
from sklearn.preprocessing import LabelEncoder

codificador_profissao = LabelEncoder()
tabela["profissao"] = codificador_profissao.fit_transform(tabela["profissao"])

codificador_mix_credito = LabelEncoder()
tabela["mix_credito"] = codificador_mix_credito.fit_transform(tabela["mix_credito"])

codificador_comportamento_pagamento = LabelEncoder()
tabela["comportamento_pagamento"] = codificador_comportamento_pagamento.fit_transform(tabela["comportamento_pagamento"])
```

---

### 🔹 Passo 3: Separar em variáveis preditoras (X) e alvo (y)
```python
x = tabela.drop(columns=["score_credito","id_cliente"])
y = tabela["score_credito"]
```

---

### 🔹 Passo 4: Dividir em treino e teste
```python
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)
```

---

### 🔹 Passo 5: Criar e treinar os modelos
Modelos utilizados:
- **RandomForestClassifier** (árvore de decisão)
- **KNeighborsClassifier** (KNN - vizinhos mais próximos)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
```

---

### 🔹 Passo 6: Avaliar os modelos
```python
from sklearn.metrics import accuracy_score

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

print("Acurácia RandomForest:", accuracy_score(y_teste, previsao_arvoredecisao))
print("Acurácia KNN:", accuracy_score(y_teste, previsao_knn))
```

---

### 🔹 Passo 7: Fazer previsões em novos clientes
```python
novos_clientes = pd.read_csv("novos_clientes.csv")

novos_clientes["profissao"] = codificador_profissao.fit_transform(novos_clientes["profissao"])
novos_clientes["mix_credito"] = codificador_mix_credito.fit_transform(novos_clientes["mix_credito"])
novos_clientes["comportamento_pagamento"] = codificador_comportamento_pagamento.fit_transform(novos_clientes["comportamento_pagamento"])

previsao = modelo_arvoredecisao.predict(novos_clientes)
novos_clientes["previsao"] = previsao

display(novos_clientes)
```

---

## 📈 Melhorias possíveis
- Testar outros modelos (SVM, Gradient Boosting, etc.)
- Ajustar hiperparâmetros com **GridSearchCV**
- Obter mais dados para treinar

---

## ▶️ Como executar
1. Clone este repositório:
   ```bash
   git clone https://github.com/seuusuario/projeto-credito.git
   ```
2. Instale as dependências:
   ```bash
   pip install pandas scikit-learn
   ```
3. Execute o script principal:
   ```bash
   python analise_credito.py
   ```

---


