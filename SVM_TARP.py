#svm
# Install dependencies if needed
!pip install tensorflow scikit-learn matplotlib --quiet

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt

#
# CONFIG
#
DATA_DIR = r"C:\Users\Shanmitha Satram\Downloads\wheat_leaf (1)\wheat_leaf"  # update if needed
img_size = (224, 224)
batch_size = 32
FEATURE_CACHE_DIR = "./feature_cache"
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

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
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

#
# STEP 2: Prepare (attempt to load pretrained VGG16; fallback if no internet)
#
print("\nLoading VGG16 base model (attempting imagenet weights)...")
try:
    base_cnn = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    print("Loaded VGG16 with imagenet weights.")
except Exception as e:
    print("Warning: could not download pretrained weights (network/proxy issue).")
    print("Falling back to VGG16 with weights=None (random init).")
    base_cnn = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))

# We'll use GlobalAveragePooling2D to reduce feature dimensionality (better for SVM)
feature_extractor = models.Sequential([
    base_cnn,
    layers.GlobalAveragePooling2D()
])

# Freeze base CNN (if pretrained)
feature_extractor.trainable = False

#
# STEP 3: Feature extraction helper (with caching)
#
def extract_features_with_cache(dataset, split_name):
    cache_X = os.path.join(FEATURE_CACHE_DIR, f"X_{split_name}.npy")
    cache_y = os.path.join(FEATURE_CACHE_DIR, f"y_{split_name}.npy")
    if os.path.exists(cache_X) and os.path.exists(cache_y):
        print(f"Loading cached features for {split_name} from {cache_X}")
        X = np.load(cache_X)
        y = np.load(cache_y)
        return X, y

    print(f"Extracting features for {split_name} ...")
    features_list = []
    labels_list = []
    for batch_images, batch_labels in dataset:
        # ensure proper preprocessing
        batch_images = preprocess_input(batch_images.numpy())
        feats = feature_extractor.predict(batch_images, verbose=0)  # shape: (B, feature_dim)
        features_list.append(feats)
        labels_list.append(batch_labels.numpy())

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)

    # cache
    np.save(cache_X, X)
    np.save(cache_y, y)
    print(f"Cached features to {cache_X} and {cache_y}")
    return X, y

X_train, y_train = extract_features_with_cache(train_ds, "train")
X_val, y_val = extract_features_with_cache(val_ds, "val")

print("Train feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)

#
# STEP 4: Train SVM
#
print("\nTraining linear SVM (this may take a while depending on data size)...")
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)
print("SVM trained.")

#
# STEP 5: Evaluate
#
y_pred = svm_model.predict(X_val)
y_prob = svm_model.predict_proba(X_val)

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred, target_names=class_names, digits=4))

# Custom metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average="macro", zero_division=0)
recall = recall_score(y_val, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

# Specificity per class and macro-average
specificity_scores = []
for i in range(len(class_names)):
    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
    fp = np.sum(cm[:, i]) - cm[i, i]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    specificity_scores.append(spec)
specificity = np.mean(specificity_scores)

misclassification_rate = 1 - accuracy

try:
    auc_roc = roc_auc_score(y_val, y_prob, multi_class="ovr")
except Exception:
    auc_roc = None

print("\n--- Evaluation Metrics (in %) ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision (macro): {precision * 100:.2f}%")
print(f"Recall (macro): {recall * 100:.2f}%")
print(f"F1-score (macro): {f1 * 100:.2f}%")
print(f"Specificity (macro): {specificity * 100:.2f}%")
print(f"Misclassification Rate: {misclassification_rate * 100:.2f}%")
print(f"AUC-ROC (macro, OVR): {auc_roc * 100:.2f}%" if auc_roc is not None else "AUC-ROC (macro, OVR): Not available")


#
# STEP 6: Utility: predict single image
#
from tensorflow.keras.utils import load_img, img_to_array

def predict_image(img_path):
    img = load_img(img_path, target_size=img_size)
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    feat = feature_extractor.predict(img_arr).reshape(1, -1)
    pred = svm_model.predict(feat)[0]
    return class_names[pred]

# Example:
# test_img = r"C:\Users\Devaashish\Downloads\wheat_leaf\wheat_leaf\Stripe_Rust\sample.jpg"
# print("Prediction:", predict_image(test_img))
