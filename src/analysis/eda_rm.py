import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/rm_cleaned_dataset.csv")

# Distribution Age

plt.figure()
df["PR_AGE1"].hist(bins=20)
plt.title("Distribution de l'âge")
plt.xlabel("Age")
plt.ylabel("Fréquence")
plt.show()


# Distribution Sexe

plt.figure()
df["PR_Q3D"].value_counts().plot(kind="bar")
plt.title("Distribution du sexe")
plt.xlabel("Sexe")
plt.ylabel("Nombre")
plt.show()


# Niveau de soutien global (PR_QQ)

plt.figure()
df["PR_QQ"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution du niveau de soutien (PR_QQ)")
plt.xlabel("Niveau soutien")
plt.ylabel("Nombre")
plt.show()


# Troubles neurologiques (PR_QN1_*)

trouble_cols = [col for col in df.columns if "PR_QN1_" in col]

trouble_counts = df[trouble_cols].sum()

plt.figure()
trouble_counts.plot(kind="bar")
plt.title("Fréquence des troubles neurologiques")
plt.xticks(rotation=45)
plt.ylabel("Nombre")
plt.show()


# Heatmap corrélation

# Encodage rapide pour corrélation
df_encoded = pd.get_dummies(df, drop_first=True)

plt.figure(figsize=(10,8))
sns.heatmap(df_encoded.corr(), cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()


# Boxplot âge vs niveau soutien

plt.figure()
sns.boxplot(x=df["PR_QQ"], y=df["PR_AGE1"])
plt.title("Age vs Niveau de soutien")
plt.xlabel("Niveau soutien")
plt.ylabel("Age")
plt.show()


# PCA Projection 2D

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.title("Projection PCA - Structure des profils RM")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.show()

print("Variance expliquée PCA :", pca.explained_variance_ratio_)