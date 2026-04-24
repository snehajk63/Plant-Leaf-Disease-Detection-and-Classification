#rnn
# Install dependencies
!pip install tensorflow scikit-learn matplotlib

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_auc_score
)

#
# STEP 1: Load Dataset
#
DATA_DIR = r"C:\Users\Shanmitha Satram\Downloads\wheat_leaf (1)\wheat_leaf"  
img_size = (224, 224)
batch_size = 32

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
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

#
# STEP 2: Build CNN-RNN Model
#
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze pretrained CNN

model = models.Sequential([
    base_model,
    layers.Reshape((-1, 512)),  # Convert CNN output to sequence [time_steps, features]
    layers.LSTM(128),           # RNN (LSTM layer)
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#
# STEP 3: Train Model
#
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

#
# STEP 4: Evaluate
#
loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy (Keras): {acc:.4f}")

#
# STEP 5: Compute 7 Metrics
#
y_true = []
y_pred = []
y_prob = []

for images, labels in val_ds:
    probs = model.predict(images)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)
    y_prob.extend(probs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

# Specificity: TN / (TN + FP) per class → macro average
cm = confusion_matrix(y_true, y_pred)
specificity_list = []
for i in range(len(class_names)):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity_list.append(tn / (tn + fp))
specificity = np.mean(specificity_list)

# Misclassification Rate = 1 - Accuracy
misclassification_rate = 1 - accuracy

# AUC-ROC (macro, one-vs-rest)
try:
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
except:
    auc = None

print("\n--- Evaluation Metrics (in %) ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")
print(f"Misclassification Rate: {misclassification_rate * 100:.2f}%")
print(f"AUC-ROC: {auc * 100:.2f}%" if auc is not None else "AUC-ROC: Not Available")


#
# STEP 6: Predict on New Image
#
from tensorflow.keras.utils import load_img, img_to_array

def predict_image(img_path):
    img = load_img(img_path, target_size=img_size)
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    pred = model.predict(img_arr)
    return class_names[np.argmax(pred)]

# Example:
# test_img = r"C:\Users\Devaashish\Downloads\wheat_leaf\Stripe_Rust\sample.jpg"
# print("Prediction:", predict_image(test_img))
