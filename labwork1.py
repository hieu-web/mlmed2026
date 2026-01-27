import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def build_model(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv1D(64, 5, activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(128, 5, activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(256, 3, activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling1D())

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


print("Đang tải dữ liệu...")
train_df = pd.read_csv("mitbih_train.csv", header=None)
test_df = pd.read_csv("mitbih_test.csv", header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(X_train.shape[0], 187, 1)
X_test = X_test.reshape(X_test.shape[0], 187, 1)

num_classes = len(np.unique(y_train))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

model = build_model((187, 1), num_classes)
model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[reduce_lr],
    verbose=1
)
