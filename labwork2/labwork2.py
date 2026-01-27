import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

TRAIN_IMG_DIR = r"C:\Users\Admin\Downloads\training_set"
CSV_PATH = "training_set_pixel_size_and_HC.csv"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

images = []
targets = []

for _, row in df.iterrows():
    img_path = os.path.join(TRAIN_IMG_DIR, row["filename"])
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    images.append(img)
    targets.append(row["head circumference (mm)"])

X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(targets)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

y_pred = model.predict(X_val).flatten()

plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()],
         [y_val.min(), y_val.max()], "r--")
plt.xlabel("Actual HC (mm)")
plt.ylabel("Predicted HC (mm)")
plt.title("Prediction vs Ground Truth")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_actual.png"))
plt.close()

errors = np.abs(y_pred - y_val)

plt.figure(figsize=(7,5))
plt.hist(errors, bins=30)
plt.xlabel("Absolute Error (mm)")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "error_distribution.png"))
plt.close()

plt.figure(figsize=(4,4))
plt.bar(["Actual", "Predicted"], [y_val[0], y_pred[0]])
plt.ylabel("HC (mm)")
plt.title("Sample Prediction")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_prediction.png"))
plt.close()

loss, mae = model.evaluate(X_val, y_val)
print("Validation MAE:", mae)

print("Saved figures at:", os.path.abspath(OUTPUT_DIR))
