# -----------------------------
# DL + Feature Fusion for Leaf Disease Classification
# -----------------------------

# Install dependencies (only run once)
!pip install opencv-python tensorflow scikit-learn matplotlib

# -----------------------------
# Imports
# -----------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

import numpy as np
import cv2
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, recall_score, precision_score, f1_score
)
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
DATA_DIR = r"C:\Users\Shanmitha Satram\Downloads\wheat_leaf (1)\wheat_leaf" # Change to your dataset path

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

# -----------------------------
# STEP 2: CNN Feature Extractor (VGG16)
# -----------------------------
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

def extract_deep_features(dataset):
    features, labels = [], []
    for batch, lbls in dataset:
        batch = preprocess_input(batch)
        feat = base_model.predict(batch, verbose=0)
        feat = feat.reshape(feat.shape[0], -1)
        features.append(feat)
        labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)

X_train_cnn, y_train = extract_deep_features(train_ds)
X_val_cnn, y_val = extract_deep_features(val_ds)

print("CNN Train features:", X_train_cnn.shape)

# -----------------------------
# STEP 3: Handcrafted Features (Color Histogram)
# -----------------------------
def color_hist_features(dataset):
    features, labels = [], []
    for batch, lbls in dataset:
        for img in batch:
            img = img.numpy().astype("uint8")
            hist = cv2.calcHist([img], [0, 1, 2], None, [8,8,8], [0,256,0,256,0,256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
        labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)

X_train_hist, _ = color_hist_features(train_ds)
X_val_hist, _ = color_hist_features(val_ds)

print("Color Histogram Train features:", X_train_hist.shape)

# -----------------------------
# STEP 4: Feature Fusion
# -----------------------------
X_train = np.hstack([X_train_cnn, X_train_hist])
X_val = np.hstack([X_val_cnn, X_val_hist])

print("Fused Train Features:", X_train.shape)

# -----------------------------
# STEP 5: Train Classifier (SVM)
# -----------------------------
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
y_prob = clf.predict_proba(X_val)

# -----------------------------
# STEP 6: Evaluation Metrics
# -----------------------------
acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, average="macro")
rec = recall_score(y_val, y_pred, average="macro")
f1 = f1_score(y_val, y_pred, average="macro")

cm = confusion_matrix(y_val, y_pred)
TN = np.sum(np.diag(cm)) - np.diag(cm)
FP = np.sum(cm, axis=0) - np.diag(cm)
spec = np.mean(TN / (TN + FP))

mis_rate = 1 - acc

try:
    auc = roc_auc_score(y_val, y_prob, multi_class="ovr")
except:
    auc = None

print("\n---- Evaluation Metrics ----")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Specificity: {spec:.4f}")
print(f"Misclassification Rate: {mis_rate:.4f}")
print(f"AUC-ROC: {auc:.4f}" if auc else "AUC-ROC: Not Available")

print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=class_names))
print("\nConfusion Matrix:\n", cm)

# -----------------------------
# STEP 7: ROC Curve (for multiclass)
# -----------------------------
if auc:
    fpr, tpr, _ = roc_curve(y_val, y_prob[:,1], pos_label=1)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
