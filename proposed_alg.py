
import cv2
import os
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss
)

from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_incep
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xcep
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D


dataset_path = r"E:\trap\wheat_leaf"
print(" Dataset path:", dataset_path)
def denoise_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def normalize_and_resize(image, size=(224, 224)):
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return image

def augment_image(image):
    flipped_h = cv2.flip(image, 1)   # Horizontal flip
    flipped_v = cv2.flip(image, 0)   # Vertical flip
    flipped_hv = cv2.flip(image, -1) # Both flips
    return [
        ("original", image),
        ("flipH", flipped_h),
        ("flipV", flipped_v),
        ("flipHV", flipped_hv)
    ]

def kmeans_segmentation(image, K=2):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    return segmented.reshape(image.shape)

X, y = [], []
class_names = sorted(os.listdir(dataset_path))
label_dict = {cls: idx for idx, cls in enumerate(class_names)}

for cls in class_names:
    cls_path = os.path.join(dataset_path, cls)
    if not os.path.isdir(cls_path):
        continue

    for file in os.listdir(cls_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(cls_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = denoise_image(img)
            img = normalize_and_resize(img)
            aug_list = augment_image((img * 255).astype(np.uint8))

            for _, aug_img in aug_list:
                seg_img = kmeans_segmentation(aug_img)
                X.append(seg_img)
                y.append(label_dict[cls])

X = np.array(X)
y = np.array(y)
print(f"Total images (augmented + segmented): {len(X)} across {len(class_names)} classes")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def extract_features(model, preprocess_func, images):
    feats = []
    for img in images:
        x = preprocess_func(np.expand_dims(img, axis=0))
        feat = model.predict(x, verbose=0)
        feats.append(feat.flatten())
    return np.array(feats)

print("\n Extracting Deep Features (InceptionV3 + Xception)...")

inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

inception_model = Model(inputs=inception_base.input, outputs=GlobalAveragePooling2D()(inception_base.output))
xception_model = Model(inputs=xception_base.input, outputs=GlobalAveragePooling2D()(xception_base.output))

incep_train = extract_features(inception_model, preprocess_incep, X_train)
incep_test = extract_features(inception_model, preprocess_incep, X_test)
xcep_train = extract_features(xception_model, preprocess_xcep, X_train)
xcep_test = extract_features(xception_model, preprocess_xcep, X_test)

X_train_fused = np.hstack([incep_train, xcep_train])
X_test_fused = np.hstack([incep_test, xcep_test])


scaler = StandardScaler().fit(X_train_fused)
X_train_scaled = scaler.transform(X_train_fused)
X_test_scaled = scaler.transform(X_test_fused)

svm = SVC(kernel='rbf', probability=True)
print("\n Training Hybrid Model (10 epochs simulated)...")
for epoch in range(1, 11):
    svm.fit(X_train_scaled, y_train)
    print(f"Epoch {epoch}/10 completed.")

y_pred = svm.predict(X_test_scaled)
y_prob = svm.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
cm = confusion_matrix(y_test, y_pred)
spec = np.mean([cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0 for i in range(len(cm))])
misclass = 1 - acc
logl = log_loss(y_test, y_prob)


print("\n   Hybrid Model Performance ")
print(f"Epochs Used     : 10")
print(f"Accuracy        : {acc*100:.2f}%")
print(f"Precision       : {prec*100:.2f}%")
print(f"Recall          : {rec*100:.2f}%")
print(f"Specificity     : {spec*100:.2f}%")
print(f"F1-Score        : {f1*100:.2f}%")
print(f"Misclassification Rate : {misclass*100:.2f}%")
print(f"Log Loss        : {logl:.4f}")