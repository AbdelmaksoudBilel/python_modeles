import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, f1_score
import joblib
import os

df = pd.read_csv("data/processed/qchat_cleaned.csv")

# Séparer X et y
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Identifier colonnes numériques et catégorielles
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

# Transformer numérique → scaler, catégorielle → OneHot
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

all_models = {
    "Logistic Regression": Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=50, random_state=42))
    ]),
    "XGBoost": Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=42))
    ]),
    "SVM (Linear)": Pipeline([
        ('preprocessor', preprocessor),
        ('model', SVC(kernel='linear', probability=True, random_state=42))
    ]),
    "SVM (RBF)": Pipeline([
        ('preprocessor', preprocessor),
        ('model', SVC(kernel='rbf', probability=True, random_state=42))
    ])
}

best_roc_auc = 0
best_pipeline = None
best_model_name = ""

for model_name, model in all_models.items():
    print(f"\n=== {model_name} ===")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1-score:", f1)
        print("ROC-AUC:", roc_auc)
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_pipeline = model
            best_model_name = model_name
    except Exception as e:
        print(f"Erreur pour {model_name} : {e}")
        continue

print(f"\nMeilleur modèle : {best_model_name} — ROC-AUC : {best_roc_auc:.2f}")

save_directory = 'models_saved/' 

filename = 'modele_tsa_ml.pkl'

full_path = os.path.join(save_directory, filename)

joblib.dump(all_models[best_model_name], full_path)

print(f"Model saved to: {full_path}")