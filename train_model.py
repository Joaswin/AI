# train_model.py
import os
import re
import math
import joblib
import logging
from collections import Counter, defaultdict
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

from core.features import extract_features_from_file, PRIMARY_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = "dataset"
MODEL_PATH = "models/malware_pipeline.pkl"
RANDOM_STATE = 42

# -------------------------
# Feature extraction helpers
# -------------------------
def shannon_entropy(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        if not data:
            return 0.0
        counts = Counter(data)
        probs = [c / len(data) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)
    except Exception:
        return 0.0

def extract_printable_strings(file_path, min_len=4):
    try:
        with open(file_path, "rb") as f:
            raw = f.read().decode("latin1")
        return re.findall(r'[\x20-\x7E]{%d,}' % min_len, raw)
    except Exception:
        return []

def vectorize(feats, file_path):
    combined = []

    for key in ["HTTP", "FTP", "SMTP", "DNS",
                "os.system", "subprocess", "eval", "exec",
                "open", "socket", "shutil", "ctypes", "getenv"]:
        if key in feats:
            combined.extend([key] * feats.get(key, 0))

    funcs = feats.get("functions", [])
    imports = feats.get("imports", [])
    combined.extend(funcs)
    combined.extend(imports)

    counts = [combined.count(f) for f in PRIMARY_FEATURES]

    try:
        file_size = os.path.getsize(file_path)
    except Exception:
        file_size = 0

    entropy = shannon_entropy(file_path)
    num_strings = len(feats.get("strings", [])) or len(extract_printable_strings(file_path))
    num_imports = len(imports)
    num_functions = len(funcs)
    num_params = sum(len(p) for p in feats.get("params", {}).values()) if feats.get("params") else 0

    return counts + [file_size, entropy, num_strings, num_imports, num_functions, num_params]

# -------------------------
# CV-style classification report
# -------------------------
from sklearn.metrics import precision_recall_fscore_support

def cv_classification_report(model, X, y, label_encoder, cv=10):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    classes = label_encoder.classes_
    n_classes = len(classes)

    # Collect metrics per fold
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        p, r, f, s = precision_recall_fscore_support(
            y_test, y_pred, labels=list(range(n_classes)), zero_division=0
        )
        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)
        support_list.append(s)

    # Average over folds
    precision_avg = np.mean(precision_list, axis=0)
    recall_avg = np.mean(recall_list, axis=0)
    f1_avg = np.mean(f1_list, axis=0)
    support_total = np.sum(support_list, axis=0)

    report = {}
    for i, c in enumerate(classes):
        report[c] = {
            "precision": precision_avg[i],
            "recall": recall_avg[i],
            "f1-score": f1_avg[i],
            "support": int(support_total[i])
        }

    overall_accuracy = sum([v["support"]*v["recall"] for v in report.values()]) / sum([v["support"] for v in report.values()])

    macro_avg = {
        "precision": np.mean([v["precision"] for v in report.values()]),
        "recall": np.mean([v["recall"] for v in report.values()]),
        "f1-score": np.mean([v["f1-score"] for v in report.values()]),
        "support": sum([v["support"] for v in report.values()])
    }
    weighted_avg = {
        "precision": np.average([v["precision"] for v in report.values()], weights=[v["support"] for v in report.values()]),
        "recall": np.average([v["recall"] for v in report.values()], weights=[v["support"] for v in report.values()]),
        "f1-score": np.average([v["f1-score"] for v in report.values()], weights=[v["support"] for v in report.values()]),
        "support": sum([v["support"] for v in report.values()])
    }

    return report, overall_accuracy, macro_avg, weighted_avg

# -------------------------
# Main training pipeline
# -------------------------
def main():
    X = []
    y = []

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
                if not feats or not isinstance(feats, dict):
                    continue
                vec = vectorize(feats, file_path)
                X.append(vec)
                y.append(family)
            except Exception as e:
                logging.warning("[!] Skipped %s: %s", file_path, e)

    if not X:
        logging.error("[!] No valid features found")
        return

    X = np.array(X)
    y = np.array(y)
    logging.info("[+] Collected %d samples across %d classes", len(y), len(set(y)))

    # -------------------------
    # Balance Classes
    # -------------------------
    by_label = defaultdict(list)
    for xv, label in zip(X, y):
        by_label[label].append(xv)

    max_count = max(len(lst) for lst in by_label.values())
    X_balanced, y_balanced = [], []

    for label, items in by_label.items():
        items_res = resample(
            items,
            replace=True,
            n_samples=max_count,
            random_state=RANDOM_STATE
        ) if len(items) < max_count else items

        X_balanced.extend(items_res)
        y_balanced.extend([label] * len(items_res))

    X = np.array(X_balanced)
    y_str = np.array(y_balanced)
    logging.info("[+] Balanced dataset to %d per class (total %d)", max_count, len(y_str))

    # -------------------------
    # Encode + Scale
    # -------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # Train/Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # -------------------------
    # Train Model
    # -------------------------
    logging.info("[+] Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluate on Test Set
    # -------------------------
    y_pred = model.predict(X_test)
    y_test_names = label_encoder.inverse_transform(y_test)
    y_pred_names = label_encoder.inverse_transform(y_pred)

    print("\n===== Classification Report (Test Set) =====")
    print(classification_report(y_test_names, y_pred_names))

    cm = confusion_matrix(y_test_names, y_pred_names, labels=label_encoder.classes_)

    # -------------------------
    # 10-Fold CV Summary
    # -------------------------
    logging.info("[+] Running 10-fold cross-validation...")
    report, accuracy, macro_avg, weighted_avg = cv_classification_report(
        model, X_scaled, y, label_encoder, cv=10
    )

    print("\n===== Classification Report (10-Fold CV) =====")
    print(f"{'':<15}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n")
    for label, metrics in report.items():
        print(f"{label:<15}{metrics['precision']:>10.2f}{metrics['recall']:>10.2f}{metrics['f1-score']:>10.2f}{metrics['support']:>10}")
    
    print(f"\n{'accuracy':<15}{'':>10}{'':>10}{accuracy:>10.2f}{sum([v['support'] for v in report.values()]):>10}")
    print(f"{'macro avg':<15}{macro_avg['precision']:>10.2f}{macro_avg['recall']:>10.2f}{macro_avg['f1-score']:>10.2f}{macro_avg['support']:>10}")
    print(f"{'weighted avg':<15}{weighted_avg['precision']:>10.2f}{weighted_avg['recall']:>10.2f}{weighted_avg['f1-score']:>10.2f}{weighted_avg['support']:>10}")

    # -------------------------
    # Graphs
    # -------------------------
    # Fold accuracy bar chart (approx using CV accuracy)
    scores = cross_val_score(model, X_scaled, y, cv=10, scoring="accuracy", n_jobs=1)
    plt.figure(figsize=(10,5))
    plt.bar(range(1,11), scores, color="skyblue")
    plt.axhline(scores.mean(), color='red', linestyle='--', label='Mean Accuracy')
    plt.title("10-Fold Cross Validation Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.xticks(range(1,11))
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Accuracy distribution
    plt.figure(figsize=(6,4))
    sns.boxplot(y=scores)
    plt.title("Cross Validation Accuracy Distribution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

    # Confusion matrix heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Feature importance
    feature_names = PRIMARY_FEATURES + [
        "file_size", "entropy", "num_strings",
        "num_imports", "num_functions", "num_params"
    ]
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(12,6))
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), np.array(feature_names)[sorted_idx], rotation=90)
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Save Model + Features
    # -------------------------
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder
    }, MODEL_PATH)
    logging.info("[+] Saved pipeline to %s", MODEL_PATH)

    features_path = os.path.join(os.path.dirname(MODEL_PATH), "primary_features.json")
    with open(features_path, "w") as f:
        json.dump(PRIMARY_FEATURES, f)
    logging.info("[+] Saved PRIMARY_FEATURES to %s", features_path)

if __name__ == "__main__":
    main()