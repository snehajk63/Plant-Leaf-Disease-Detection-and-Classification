#CNN
# Install dependencies (run once)
!pip install tensorflow matplotlib scikit-learn --quiet

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

#
# CONFIG
#
DATA_DIR = r"C:\Users\Shanmitha Satram\Downloads\wheat_leaf (1)\wheat_leaf" # change if needed
img_size = (128, 128)
batch_size = 32
epochs = 15

#
# STEP 1: Load Dataset
#
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

#
# STEP 2: Build CNN Model
#
model = models.Sequential([
    layers.Input(shape=(img_size[0], img_size[1], 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#
# STEP 3: Train Model
#
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

#
# STEP 4: Evaluate & Compute Metrics
#
y_true = []
y_pred = []
y_prob = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)            # shape: (batch, num_classes)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_prob.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# Confusion Matrix (for class-wise metrics)
cm = confusion_matrix(y_true, y_pred)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

# Specificity: compute per-class TN/(TN+FP) then macro-average
specificity_list = []
num_classes = len(class_names)
for i in range(num_classes):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    specificity_list.append(spec_i)
specificity = float(np.mean(specificity_list))

# Misclassification Rate
misclassification_rate = 1.0 - accuracy

# Macro AUC-ROC (one-vs-rest). Requires probability outputs and >1 class present.
try:
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
except Exception:
    auc = None

# Print in the same formatted block as your image
# Print in the same formatted block as your image (in percentage)
print("\n--- Evaluation Metrics (in %) ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")
print(f"Misclassification Rate: {misclassification_rate * 100:.2f}%")

if auc is not None:
    print(f"AUC-ROC: {auc * 100:.2f}%")
else:
    print("AUC-ROC: Not Available")


# Optional: classification report and confusion matrix (comment/uncomment as needed)
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names, digits=4))
print("\nConfusion Matrix:\n", cm)

#
# STEP 5: Plot Training Curves
#
plt.figure(figsize=(8, 5))
plt.plot(history.history.get("accuracy", []), label="train_acc")
plt.plot(history.history.get("val_accuracy", []), label="val_acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history.get("loss", []), label="train_loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()