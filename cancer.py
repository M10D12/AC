# ==========================================================
# Aprendizagem Computacional 2025/2026
# Projeto: Breast Cancer Coimbra
# Parte 1 - Data Loading (adaptado do CovidML)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Leitura do dataset
# ----------------------------------------------------------

print("=== DATA LOADING ===")

# Ler o ficheiro Excel com os dados
data = pd.read_excel("dataR2.xlsx")

print("\nDimensões do dataset:", data.shape)
print("\nPrimeiras 5 amostras:")
print(data.head())

# ----------------------------------------------------------
# 2. Informação geral sobre as variáveis
# ----------------------------------------------------------

print("\n--- Informação sobre as variáveis ---")
print(data.info())

print("\n--- Estatísticas descritivas ---")
print(data.describe())

# ----------------------------------------------------------
# 3. Verificação de valores em falta
# ----------------------------------------------------------

print("\n--- Valores em falta por variável ---")
print(data.isnull().sum())

# ----------------------------------------------------------
# 4. Identificação da variável alvo (classe)
# ----------------------------------------------------------

target_col = data.columns[-1]
print(f"\nVariável alvo: {target_col}")

# ----------------------------------------------------------
# 5. Distribuição das classes
# ----------------------------------------------------------

y = data[target_col].values
classes, counts = np.unique(y, return_counts=True)

print("\n--- Distribuição das classes ---")
for c, n in zip(classes, counts):
    print(f"Classe {c}: {n} amostras ({n/len(y)*100:.1f}%)")

# ----------------------------------------------------------
# 6. Visualização simples das classes
# ----------------------------------------------------------

plt.figure(figsize=(6,4))
plt.bar(classes, counts, color=['skyblue', 'salmon'], edgecolor='black')
plt.title("Distribuição das Classes (1 = Saudável, 2 = Cancro)")
plt.xlabel("Classe")
plt.ylabel("Número de Amostras")
plt.xticks(classes)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 7. Visualização básica das variáveis numéricas
# ----------------------------------------------------------

plt.figure(figsize=(10,6))
plt.boxplot(data.iloc[:, :-1], labels=data.columns[:-1], vert=True)
plt.yscale('log')  # aplicar escala logarítmica
plt.title("Boxplots das Variáveis Numéricas (Escala Logarítmica)")
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 8. Conclusão
# ----------------------------------------------------------

print("\nData loading concluído com sucesso.")
print("Dataset pronto para análise e preparação dos dados.")






# ==========================================================
# Aprendizagem Computacional 2025/2026
# Projeto: Breast Cancer Coimbra
# Parte 2 - Data Partitioning (70% Treino / 30% Teste)
# ==========================================================


# ----------------------------------------------------------
# 1. Ler o dataset
# ----------------------------------------------------------

data = pd.read_excel("dataR2.xlsx")

# Separar features (X) e target (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print("=== DATA PARTITIONING ===")
print(f"Dimensões originais: {X.shape[0]} amostras, {X.shape[1]} variáveis")

# ----------------------------------------------------------
# 2. Divisão treino/teste (70/30)
# ----------------------------------------------------------

# Definir proporção de treino
train_ratio = 0.7
N = len(y)
train_size = int(train_ratio * N)

# Definir semente para reprodutibilidade
np.random.seed(42)

# Embaralhar os índices
indices = np.random.permutation(N)

# Selecionar índices para treino e teste
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# Criar conjuntos de treino e teste
X_train = X[train_idx, :]
y_train = y[train_idx]
X_test = X[test_idx, :]
y_test = y[test_idx]

# ----------------------------------------------------------
# 3. Resumo da divisão
# ----------------------------------------------------------

print(f"\nNúmero total de amostras: {N}")
print(f"Tamanho do conjunto de treino: {len(y_train)} ({len(y_train)/N*100:.1f}%)")
print(f"Tamanho do conjunto de teste: {len(y_test)} ({len(y_test)/N*100:.1f}%)")

# Distribuição de classes no treino
unique, counts = np.unique(y_train, return_counts=True)
print("\nDistribuição no conjunto de treino:")
for cls, cnt in zip(unique, counts):
    print(f"Classe {cls}: {cnt} amostras ({cnt/len(y_train)*100:.1f}%)")

# Distribuição de classes no teste
unique, counts = np.unique(y_test, return_counts=True)
print("\nDistribuição no conjunto de teste:")
for cls, cnt in zip(unique, counts):
    print(f"Classe {cls}: {cnt} amostras ({cnt/len(y_test)*100:.1f}%)")

# ----------------------------------------------------------
# 4. Visualização opcional das classes (treino e teste)
# ----------------------------------------------------------



plt.figure(figsize=(6,4))
plt.hist(y_train, bins=np.arange(y.min(), y.max()+2)-0.5,
         rwidth=0.8, color='lightgreen', edgecolor='black', label='Treino')
plt.hist(y_test, bins=np.arange(y.min(), y.max()+2)-0.5,
         rwidth=0.5, color='salmon', edgecolor='black', label='Teste', alpha=0.7)
plt.title("Distribuição das Classes (Treino vs Teste)")
plt.xlabel("Classe (1=Saudável, 2=Cancro)")
plt.ylabel("Número de Amostras")
plt.legend()
plt.show()

# ----------------------------------------------------------
# 5. Conclusão
# ----------------------------------------------------------

print("\nData partitioning concluído com sucesso.")
print("Conjuntos X_train, X_test, y_train e y_test prontos para treino e avaliação.")




