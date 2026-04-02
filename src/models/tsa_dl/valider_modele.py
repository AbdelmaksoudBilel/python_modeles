import os
import math
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Reproductibilité
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Hyperparamètres finaux
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 0.001
DROPOUT = 0.2
EPOCHS = 30
THRESHOLD = 0.5

TRAIN_DIR = "data/images/train"
VAL_DIR = "data/images/valid"
TEST_DIR = "data/images/test"
SAVE_PATH = "models_saved/modele_tsa_cnn.h5"

os.makedirs("models_saved", exist_ok=True)

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Modèle
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# EarlyStopping
early_stop = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=5,
    restore_best_weights=True
)

# Training
print("\n Entraînement du modèle final...\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1
)

# Sauvegarde modèle
model.save(SAVE_PATH)
print(f"\n Modèle sauvegardé dans {SAVE_PATH}")

# Évaluation Test
print("\n Évaluation sur le jeu de test...\n")

steps = math.ceil(test_gen.samples / BATCH_SIZE)
probs = model.predict(test_gen, steps=steps, verbose=0).ravel()
y_true = test_gen.classes
y_pred = (probs >= THRESHOLD).astype(int)

print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"ROC AUC : {roc_auc_score(y_true, probs):.4f}")
print(f"F1 score : {f1_score(y_true, y_pred):.4f}")
print(f"Precision : {precision_score(y_true, y_pred):.4f}")
print(f"Recall : {recall_score(y_true, y_pred):.4f}\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion")
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.show()

# Courbes Training
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Évolution Accuracy")
plt.xlabel("Époques")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Évolution Loss")
plt.xlabel("Époques")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()