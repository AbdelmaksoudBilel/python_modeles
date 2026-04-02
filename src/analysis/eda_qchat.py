import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Charger dataset nettoyé
df = pd.read_csv("data/processed/qchat_cleaned.csv")

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())

print("\nTarget distribution:")
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True))

# Distribution âge
plt.hist(df["Age_Years"], bins=20)
plt.title("Distribution Age")
plt.xlabel("Age (Years)")
plt.ylabel("Count")
plt.show()

#Repartition des ASD
palette_color=['Blue','red']
plt.figure(figsize=(20, 10))
sns.countplot(x='target',data=df,palette=palette_color)
plt.title('repartition des ASD')
plt.show()

# Encoder proprement variables binaires
binary_map = {
    "m": 1,
    "f": 0,
}

df["Sex"] = df["Sex"].replace(binary_map)

# Vérifier qu'il reste pas de string
print(df.dtypes)

# Corrélation
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Matrice de Corrélation")
plt.show()

# Séparer X et y
X = df.drop('target', axis=1)
y = df['target']

# Sélectionner seulement les colonnes numériques dans X
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Modèle
X_train, X_test, y_train, y_test = train_test_split(
    X[num_features],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Feature importance
importances = pd.Series(clf.feature_importances_, index=num_features)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 5))
importances.plot(kind='bar')
plt.title("Importance des variables (Decision Tree)")
plt.ylabel("Score d'importance")
plt.tight_layout()
plt.show()