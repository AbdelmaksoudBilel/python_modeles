import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Chargement
df = pd.read_csv("data/processed/rm_cleaned_dataset.csv")

df.replace([-88, -99], np.nan, inplace=True)
df = df.dropna()
df = df.drop_duplicates()

df = pd.get_dummies(df, drop_first=True)

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

iso_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    ))
])

iso_pipeline.fit(X_train)

iso_scores = iso_pipeline.decision_function(X_test)

plt.figure()
plt.hist(iso_scores, bins=50)
plt.title("Distribution Scores - Isolation Forest")
plt.xlabel("Score anomalie")
plt.ylabel("Fréquence")
plt.show()

svm_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05
    ))
])

svm_pipeline.fit(X_train)

svm_scores = svm_pipeline.decision_function(X_test)

plt.figure()
plt.hist(svm_scores, bins=50)
plt.title("Distribution Scores - One-Class SVM")
plt.xlabel("Score anomalie")
plt.ylabel("Fréquence")
plt.show()

# Conversion vers numpy
X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# Normalisation (indépendante)
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# Tensor
X_train_tensor = torch.tensor(X_train_np)
X_test_tensor = torch.tensor(X_test_np)

train_loader = DataLoader(
    TensorDataset(X_train_tensor),
    batch_size=64,
    shuffle=True
)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_dim = X_train_np.shape[1]
model_ae = Autoencoder(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model_ae.parameters(), lr=0.001)

epochs = 30

for epoch in range(epochs):
    for batch in train_loader:
        data = batch[0]
        
        optimizer.zero_grad()
        output = model_ae(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.6f}")

model_ae.eval()

with torch.no_grad():
    reconstructed = model_ae(X_test_tensor)
    reconstruction_error = torch.mean(
        (X_test_tensor - reconstructed) ** 2,
        dim=1
    ).numpy()

plt.figure()
plt.hist(reconstruction_error, bins=50)
plt.title("Distribution Erreur Reconstruction - Autoencoder")
plt.xlabel("Erreur")
plt.ylabel("Fréquence")
plt.show()


from sklearn.preprocessing import MinMaxScaler

scores_dict = {
    "Isolation Forest": -iso_scores,
    "One-Class SVM": -svm_scores,
    "Autoencoder": reconstruction_error
}

for name, scores in scores_dict.items():
    scaler = MinMaxScaler()
    norm_scores = scaler.fit_transform(scores.reshape(-1,1))
    
    plt.figure()
    plt.hist(norm_scores, bins=50)
    plt.title(f"Normalized Scores - {name}")
    plt.show()
# all_models = {
#     "Isolation Forest": iso_pipeline,
#     "One-Class SVM": svm_pipeline
# }

# best_model_name = ""
# best_model = None
# best_anomaly_rate_diff = 999

# for name, pipeline in all_models.items():
#     pipeline.fit(X_train)
#     pred = pipeline.predict(X_test)
    
#     anomaly_rate = np.mean(pred == -1)
#     diff = abs(anomaly_rate - 0.05)  # contamination attendue
    
#     print(f"{name} - Anomaly Rate: {anomaly_rate:.4f}")
    
#     if diff < best_anomaly_rate_diff:
#         best_anomaly_rate_diff = diff
#         best_model = pipeline
#         best_model_name = name


# for name, pipeline in all_models.items():
#     pipeline.fit(X_train)
#     scores = pipeline.decision_function(X_test)
#     avg_score = np.mean(scores)
#     print(f"{name} - Average Score: {avg_score:.4f}")
#     plt.figure()
#     plt.hist(scores, bins=50)
#     plt.title(f"Distribution Scores - {name}")
#     plt.xlabel("Score anomalie")
#     plt.ylabel("Fréquence")
#     plt.show()
    
#     if avg_score > best_score:
#         best_score = avg_score
#         best_model_name = name
#         best_model = pipeline

# joblib.dump(best_model, "models_saved/modele_rm_ml.pkl")
# print(f"Meilleur modèle : {best_model_name} avec un score moyen de {best_anomaly_rate_diff:.4f} - Modèle sauvegardé dans models_saved/modele_rm_ml.pkl")