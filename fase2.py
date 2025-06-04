import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.lines import Line2D

# === CARGA DE DATOS ===
df = pd.read_csv('data/creditcard.csv')

# === MATRIZ DE CORRELACIÓN ===
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación")
plt.show()

# === CORRELACIÓN CON LA CLASE ===
cor_target = correlation_matrix["Class"].abs()

# === SEPARAR VARIABLES ===
X = df.drop(columns=["Class"])
y = df["Class"]

# === PCA EN 3D ===
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# === FUNCIÓN PARA GRAFICAR 3D ===
def graficar_3d(X_3d, labels, title, xlabel, ylabel, zlabel):
    colors = ['red' if label == 1 else 'blue' for label in labels]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=colors, alpha=0.7, s=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Clase 0', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Clase 1', markerfacecolor='red', markersize=8)
    ]
    ax.legend(handles=legend_elements)
    plt.show()

# === GRAFICAR PCA 3D ===
graficar_3d(X_pca, y, "PCA 3D", "PC1", "PC2", "PC3")

# === VARIABLES MÁS CORRELACIONADAS CON LA CLASE ===
correlaciones = X.apply(lambda col: np.corrcoef(col, y)[0, 1])
correlaciones_abs = correlaciones.abs()
top3_vars = correlaciones_abs.sort_values(ascending=False).head(3).index.tolist()
print("Variables con mayor correlación con la clase:", top3_vars)

X_top3 = X[top3_vars].values

# === GRAFICAR LAS 3 VARIABLES MÁS CORRELACIONADAS ===
graficar_3d(X_top3, y, "Top 3 variables correlacionadas con clase", *top3_vars)

# === LDA PARA VISUALIZACIÓN EN 1D ===
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X, y)

plt.scatter(X_lda, [0]*len(X_lda), c=y)
plt.title("LDA")
plt.show()

# === VERIFICACIÓN DE DATOS ===
print("\nDistribución de clases:")
print(df['Class'].value_counts())
print("\nValores nulos por columna:")
print(df.isnull().sum())

