import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

train_dir = "data/images/train"
val_dir = "data/images/valid"
test_dir = "data/images/test"

architectures_dict = {
    "MobileNetV2": MobileNetV2,
    "ResNet50": ResNet50,
    "EfficientNetB0": EfficientNetB0
}

dropouts = [0.2, 0.3, 0.5]
batchs = [16, 32]
learningRates = [1e-3, 1e-4]

IMG_SIZE = 224
EPOCHS = 15

best_architecture = None
best_dropout = None
best_batch = None
best_learningRate = None
best_roc_auc = 0
best_model = None


def get_generators(img_size, batch_size):

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary'
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def build_model(arch_name, img_size, dropout, lr):

    base_model = architectures_dict[arch_name](
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


for arch in architectures_dict.keys():
    for lr in learningRates:
        for batch in batchs:
            for dropout in dropouts:

                print(f"\nTesting {arch} | LR={lr} | Batch={batch} | Dropout={dropout}")

                train_gen, val_gen, test_gen = get_generators(IMG_SIZE, batch)

                model = build_model(arch, IMG_SIZE, dropout, lr)

                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )

                model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS,
                    callbacks=[early_stop],
                    verbose=0
                )

                loss, acc, auc = model.evaluate(test_gen, verbose=0)

                print(f"→ AUC: {auc:.4f}")

                if auc > best_roc_auc:
                    best_roc_auc = auc
                    best_architecture = arch
                    best_dropout = dropout
                    best_batch = batch
                    best_learningRate = lr
                    best_model = model


os.makedirs("models_saved", exist_ok=True)
best_model.save("models_saved/modele_tsa_cnn.h5")

print(
    f"\n La meilleure application est {best_architecture}, "
    f"Dropout {best_dropout}, "
    f"Batch size {best_batch}, "
    f"Learning rate {best_learningRate}, "
    f"Taille d'image {IMG_SIZE} "
    f"avec une ROC_AUC de {best_roc_auc:.4f}"
)