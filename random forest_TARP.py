#random forest
# Random Forest Classifier with MobileNetV2 Feature Extraction

# Install dependencies if needed
!pip install tensorflow scikit-learn matplotlib --quiet

import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

#
# STEP 1: Load Dataset
#
DATA_DIR =  r"C:\Users\Shanmitha Satram\Downloads\wheat_leaf (1)\wheat_leaf"  # dataset path
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
# STEP 2: CNN Feature Extractor (MobileNetV2)
#
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"  # gives a flat feature vector
)

def extract_features(dataset):
    features = []
    labels = []
    for batch, label in dataset:
        feat = base_model(batch, training=False).numpy()
        features.append(feat)
        labels.append(label.numpy())
    return np.vstack(features), np.concatenate(labels)

print("\nExtracting features using MobileNetV2...")
X_train, y_train = extract_features(train_ds)
X_val, y_val = extract_features(val_ds)
print("Train feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)

#
# STEP 3: Train Random Forest
#
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=None,         # let trees grow deep
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Random Forest training complete!")

#
# STEP 4: Evaluate with Extra Metrics
#
print("\nEvaluating model...")
y_pred = rf_model.predict(X_val)
y_prob = rf_model.predict_proba(X_val)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:\n", cm)

# Metrics
acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, average='weighted')
rec = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')
auc = roc_auc_score(y_val, y_prob, multi_class="ovr")
logloss = log_loss(y_val, y_prob)

# Specificity (macro average)
specificities = []
for i in range(len(class_names)):
    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
    fp = np.sum(cm[:, i]) - cm[i, i]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)
macro_specificity = np.mean(specificities)

# Print performance metrics
# Print performance metrics (in %)
print("\n---- Performance Metrics (in %) ----")
print(f"Accuracy                    : {acc * 100:.2f}%")
print(f"Precision (Weighted)         : {prec * 100:.2f}%")
print(f"Recall (Weighted)            : {rec * 100:.2f}%")
print(f"F1-score (Weighted)          : {f1 * 100:.2f}%")
print(f"ROC-AUC (OVR)                : {auc * 100:.2f}%" if auc is not None else "ROC-AUC (OVR)                : Not Available")
print(f"Log Loss                     : {logloss * 100:.2f}%")
print(f"Specificity (Macro)          : {macro_specificity * 100:.2f}%")
