import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Chargement des données

PATH = "data/raw/developmental_services_survey_dataset_1.csv"

df = pd.read_csv(PATH)

print("Shape initial:", df.shape)


# Suppression colonnes inutiles

# Suppression UID (identifiant non informatif)
if "UID" in df.columns:
    df = df.drop(columns=["UID"])

print("Après suppression UID:", df.shape)


# Sélection des colonnes pertinentes

selected_columns = [
    # Démographie
    "PR_AGE1",
    "PR_Q3D",

    # Communication
    "PR_QF1A",
    "PR_QG1A",

    # Mobilité
    "PR_QH1A",
    "PR_QH1B",

    # Aides
    "PR_QI1",
    "PR_QJ1",
    "PR_QK1",

    # Niveau de soutien global
    "PR_QQ",

    # Troubles neurologiques
    "PR_QN1_A", "PR_QN1_B", "PR_QN1_C",
    "PR_QN1_D", "PR_QN1_E", "PR_QN1_F",
    "PR_QN1_G", "PR_QN1_H",

    # Comportements
    "PR_QO1_A_COMBINE",
    "PR_QO1_B_COMBINE",
    "PR_QO1_C_COMBINE",
    "PR_QO1_D_COMBINE",
    "PR_QO1_E_COMBINE"
]

# Garder seulement colonnes existantes
selected_columns = [col for col in selected_columns if col in df.columns]

df = df[selected_columns]

print("Après sélection colonnes:", df.shape)


# Gestion valeurs manquantes

# Remplacer -88 et -99 par NaN
df = df.replace([-88, -99], np.nan, inplace=True)

df = df.dropna()

print("Après suppression lignes trop incomplètes:", df.shape)


# Suppression des doublons

df = df.drop_duplicates()
print("Après suppression doublons:", df.shape)


# # Encodage des variables

# # Colonnes ordinales (gardées telles quelles)
# ordinal_columns = [
#     "PR_AGE1",
#     "PR_QH1A",
#     "PR_QI1",
#     "PR_QJ1",
#     "PR_QK1",
#     "PR_QQ"
# ]

# # Colonnes nominales → One-Hot
# nominal_columns = [col for col in df.columns if col not in ordinal_columns]

# df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)

# print("Après encodage:", df.shape)


# # Imputation

# imputer = SimpleImputer(strategy="median")
# df_imputed = imputer.fit_transform(df)

# df_clean = pd.DataFrame(df_imputed, columns=df.columns)

# print("Après imputation:", df_clean.shape)


# # Normalisation

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(df_clean)

# df_scaled = pd.DataFrame(X_scaled, columns=df_clean.columns)

# print("Après normalisation:", df_scaled.shape)


# Sauvegarde données nettoyées

OUTPUT_PATH = "data/processed/rm_cleaned_dataset.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("Dataset nettoyé sauvegardé dans:", OUTPUT_PATH)