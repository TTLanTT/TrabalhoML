import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y
df["target_name"] = df["target"].map({0: "malignant", 1: "benign"})

plt.figure(figsize=(4,3))
sns.countplot(x="target_name", data=df)
plt.tight_layout()
plt.savefig("distribuicao_classes.png", dpi=300)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelos = {
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "SVM (linear)": SVC(kernel="linear", random_state=42)
}

resultados = {}
matrizes = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    resultados[nome] = acc
    matrizes[nome] = cm
    print(nome)
    print("Acurácia:", acc)
    print(cm)
    print(classification_report(y_test, y_pred, target_names=data.target_names))

plt.figure(figsize=(5,3))
sns.barplot(x=list(resultados.keys()), y=list(resultados.values()))
plt.ylim(0.9, 1.0)
plt.tight_layout()
plt.savefig("acuracia_modelos.png", dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

for ax, (nome, cm) in zip(axes, matrizes.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(nome)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.savefig("matrizes_confusao.png", dpi=300)
plt.show()
