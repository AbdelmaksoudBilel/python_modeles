import os
import math
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler

# Configuration
MODEL_PATH   = "models_saved/modele_rm_ml.pkl"
DATA_PATH    = "data/processed/rm_cleaned_dataset.csv"
THRESHOLD    = 0.0          # seuil sur le score d'anomalie (0 = frontière Isolation Forest)
CONTAMINATION = 0.20        # ratio Non-RM synthétiques pour évaluation

os.makedirs("outputs", exist_ok=True)

# Chargement modèle – FIX
raw = joblib.load(MODEL_PATH)

# Cas 1 : dictionnaire {'model': ..., 'scaler': ..., 'feature_cols': ...}
if isinstance(raw, dict):
    model        = raw['model']
    scaler       = raw.get('scaler', None)
    feature_cols = raw.get('feature_cols', None)
    model_name   = raw.get('model_name', type(model).__name__)

# Cas 2 : Pipeline sklearn
elif hasattr(raw, 'named_steps'):
    model, scaler = None, None

    print("  Étapes détectées dans le Pipeline :")
    for step_name, step_obj in raw.named_steps.items():
        print(f"    - '{step_name}' → {type(step_obj).__name__}")
        if isinstance(step_obj, StandardScaler):
            scaler = step_obj
        # Détecte IsolationForest, OneClassSVM, LocalOutlierFactor, etc.
        elif hasattr(step_obj, 'score_samples') or hasattr(step_obj, 'decision_function'):
            model = step_obj

    if model is None:
        print("  ⚠ Modèle non isolé → utilisation du Pipeline complet")
        model  = raw      # predict(X_raw) → scaler → modèle
        scaler = None     # déjà intégré dans le pipeline

    feature_cols = None
    model_name   = type(model).__name__

# Cas 3 : modèle seul
else:
    model        = raw
    scaler       = None
    feature_cols = None
    model_name   = type(model).__name__

print(f"\n Modèle chargé : {model_name}")
print(f"  Type artifact : {type(raw).__name__}")
print(f"  Scaler trouvé : {'Oui' if scaler else 'Non'}")
print(f"  Feature cols  : {'Oui' if feature_cols else 'Non (auto-détection)'}\n")

# Préparation des données
df = pd.read_csv(DATA_PATH)

if feature_cols is None:
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c.upper() != 'UID']
    print(f"  → {len(feature_cols)} colonnes détectées automatiquement")

X = df[feature_cols].fillna(df[feature_cols].median())

if scaler is not None:
    X_scaled = scaler.transform(X.values)   # .values → supprime le warning feature names
else:
    print("  ⚠ Pas de scaler séparé → StandardScaler appliqué à la volée")
    X_scaled = StandardScaler().fit_transform(X.values)

# Garder X_raw pour le cas pipeline complet (scaler déjà intégré)
X_raw = X.values  # numpy array non scalé

# Dataset d'évaluation
rng      = np.random.RandomState(42)
n_rm     = len(X_scaled)
n_non_rm = int(n_rm * CONTAMINATION)

# Non-RM synthétiques dans l'espace scalé
X_non_rm_scaled = rng.uniform(
    low  = X_scaled.min(axis=0) - 3.0,
    high = X_scaled.max(axis=0) + 3.0,
    size = (n_non_rm, X_scaled.shape[1])
)

X_eval_scaled = np.vstack([X_scaled, X_non_rm_scaled])
y_true        = np.array([1] * n_rm + [-1] * n_non_rm)
idx           = rng.permutation(len(X_eval_scaled))
X_eval_scaled = X_eval_scaled[idx]
y_true        = y_true[idx]

# Prédictions

y_pred = model.predict(X_eval_scaled)
scores = model.score_samples(X_eval_scaled)

y_true_b    = (y_true == 1).astype(int)
y_pred_b    = (y_pred  == 1).astype(int)
scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

# Métriques console
print(f"Accuracy  : {accuracy_score(y_true_b, y_pred_b):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_true_b, scores_norm):.4f}")
print(f"F1 Score  : {f1_score(y_true_b, y_pred_b):.4f}")
print(f"Precision : {precision_score(y_true_b, y_pred_b):.4f}")
print(f"Recall    : {recall_score(y_true_b, y_pred_b):.4f}\n")

print("Classification Report:")
print(classification_report(
    y_true_b, y_pred_b,
    target_names=['Non-RM (synthétique)', 'RM (réel)'],
    digits=4
))

# Matrice de confusion
cm = confusion_matrix(y_true_b, y_pred_b)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Prédit Non-RM', 'Prédit RM'],
            yticklabels=['Réel Non-RM',   'Réel RM'])
plt.title(f"Matrice de Confusion – {model_name}")
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=130)
plt.show()

# Distribution des scores
rm_scores     = scores[y_true == 1]
non_rm_scores = scores[y_true == -1]

plt.figure(figsize=(10, 5))
plt.hist(rm_scores,     bins=60, alpha=0.7, color='steelblue',
         label='RM (réels)', density=True)
plt.hist(non_rm_scores, bins=40, alpha=0.7, color='tomato',
         label='Non-RM (synthétiques)', density=True)
plt.axvline(THRESHOLD, color='black', linestyle='--',
            linewidth=2, label=f'Seuil = {THRESHOLD}')
plt.title(f"Distribution des Scores d'Anomalie – {model_name}")
plt.xlabel("Score d'anomalie")
plt.ylabel("Densité")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/score_distribution.png", dpi=130)
plt.show()

# Courbe ROC
fpr, tpr, _ = roc_curve(y_true_b, scores_norm)
auc_val      = roc_auc_score(y_true_b, scores_norm)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='steelblue', linewidth=2.5,
         label=f'ROC (AUC = {auc_val:.4f})')
plt.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Aléatoire')
plt.title(f"Courbe ROC – {model_name}")
plt.xlabel("Taux Faux Positifs (FPR)")
plt.ylabel("Taux Vrais Positifs (TPR)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=130)
plt.show()

# Learning Curve
sample_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
val_scores   = []

for frac in sample_sizes:
    n     = max(10, int(len(X_scaled) * frac))
    idx_s = rng.choice(len(X_scaled), size=n, replace=False)
    X_sub = X_scaled[idx_s]

    m_tmp = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    m_tmp.fit(X_sub)

    train_scores.append(m_tmp.score_samples(X_sub).mean())
    val_scores.append(m_tmp.score_samples(X_eval_scaled[:500]).mean())

plt.figure(figsize=(10, 5))
plt.plot([int(s * 100) for s in sample_sizes], train_scores,
         marker='o', label='Score Train (moyen)', color='steelblue', linewidth=2)
plt.plot([int(s * 100) for s in sample_sizes], val_scores,
         marker='s', label='Score Évaluation (moyen)', color='tomato', linewidth=2)
plt.title("Évolution du Score selon la Taille d'Entraînement")
plt.xlabel("% des données d'entraînement utilisées")
plt.ylabel("Score d'anomalie moyen")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/learning_curve.png", dpi=130)
plt.show()

print("\n Tous les graphiques sauvegardés dans outputs/")