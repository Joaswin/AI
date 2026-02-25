# train_model_dl.py
import os
import re
import math
import json
import logging
from collections import Counter, defaultdict

import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from core.features import extract_features_from_file, PRIMARY_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "malware_dl_model.keras")
PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessing.pkl")
RANDOM_STATE = 42

# ------------------ Feature Helpers ------------------
def shannon_entropy(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        if not data:
            return 0.0
        counts = Counter(data)
        probs = [c / len(data) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)
    except:
        return 0.0

def extract_printable_strings(file_path, min_len=4):
    try:
        raw = open(file_path, "rb").read().decode("latin1")
        return re.findall(r'[\x20-\x7E]{%d,}' % min_len, raw)
    except:
        return []

def vectorize(feats, file_path):
    combined = []
    for key in [
        "HTTP","FTP","SMTP","DNS",
        "os.system","subprocess","eval","exec",
        "open","socket","shutil","ctypes","getenv"
    ]:
        if key in feats:
            combined.extend([key]*feats.get(key,0))

    funcs = feats.get("functions", [])
    imports = feats.get("imports", [])
    combined.extend(funcs)
    combined.extend(imports)

    counts = [combined.count(f) for f in PRIMARY_FEATURES]

    try:
        file_size = os.path.getsize(file_path)
    except:
        file_size = 0
    entropy = shannon_entropy(file_path)
    num_strings = len(feats.get("strings",[])) or len(extract_printable_strings(file_path))
    num_imports = len(imports)
    num_functions = len(funcs)
    num_params = sum(len(p) for p in feats.get("params",{}).values()) if feats.get("params") else 0

    return counts + [file_size, entropy, num_strings, num_imports, num_functions, num_params]

# ------------------ Main ------------------
def main():
    X, y = [], []

    logging.info("[+] Scanning dataset: %s", BASE_DIR)
    if not os.path.isdir(BASE_DIR):
        logging.error("[!] Dataset folder not found")
        return

    for family in os.listdir(BASE_DIR):
        family_dir = os.path.join(BASE_DIR, family)
        if not os.path.isdir(family_dir):
            continue
        for fname in os.listdir(family_dir):
            file_path = os.path.join(family_dir, fname)
            try:
                feats = extract_features_from_file(file_path)
                if not feats:
                    continue
                X.append(vectorize(feats, file_path))
                y.append(family)
            except Exception as e:
                logging.warning("[!] Skipped %s: %s", file_path, e)

    if not X:
        logging.error("[!] No samples collected")
        return

    X = np.array(X)
    y = np.array(y)
    logging.info("[+] Collected %d samples (%d classes)", len(y), len(set(y)))

    # ------------------ Balance Classes ------------------
    by_label = defaultdict(list)
    for xv,label in zip(X,y):
        by_label[label].append(xv)

    max_count = max(len(v) for v in by_label.values())
    X_bal, y_bal = [], []
    for label,samples in by_label.items():
        samples = resample(samples, replace=True, n_samples=max_count, random_state=RANDOM_STATE) if len(samples)<max_count else samples
        X_bal.extend(samples)
        y_bal.extend([label]*len(samples))

    X = np.array(X_bal)
    y = np.array(y_bal)
    logging.info("[+] Balanced dataset: %d samples per class", max_count)

    # ------------------ Encode & Scale ------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    num_classes = len(label_encoder.classes_)
    input_dim = X_scaled.shape[1]

    # ------------------ 10-Fold Cross-Validation ------------------
    logging.info("[+] Running 10-fold cross-validation...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    fold = 1
    acc_scores = []
    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in skf.split(X_scaled, y_encoded):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        if fold==1:
            logging.info("[+] Model Summary:\n")
            model.summary(print_fn=logging.info)

        # Train
        model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, verbose=0)

        y_pred = np.argmax(model.predict(X_test), axis=1)
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        logging.info("[Fold %d] Accuracy: %.4f", fold, acc)
        fold += 1

    logging.info("[+] 10-fold CV Accuracy: %.4f ± %.4f", np.mean(acc_scores), np.std(acc_scores))

    # ------------------ Final Classification Report ------------------
    y_true_names = label_encoder.inverse_transform(all_y_true)
    y_pred_names = label_encoder.inverse_transform(all_y_pred)
    logging.info("\n===== Classification Report =====\n" + classification_report(y_true_names, y_pred_names))

    # ------------------ Confusion Matrix & Graph ------------------
    cm = confusion_matrix(y_true_names, y_pred_names, labels=label_encoder.classes_)
    logging.info("[+] Confusion Matrix:\n%s", cm)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("DL Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR,"dl_confusion_matrix.png"))
    plt.close()

    # ------------------ Save Final Model & Pipeline ------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump({"scaler": scaler, "label_encoder": label_encoder}, PIPELINE_PATH)
    with open(os.path.join(MODEL_DIR,"primary_features.json"), "w") as f:
        json.dump(PRIMARY_FEATURES,f)

    logging.info("[+] Saved DL model to %s", MODEL_PATH)
    logging.info("[+] Saved preprocessing pipeline and PRIMARY_FEATURES")

if __name__ == "__main__":
    main()