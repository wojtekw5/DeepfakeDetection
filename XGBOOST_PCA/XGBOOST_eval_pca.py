import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve)

# Ścieżki
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_folder = os.path.join(desktop_path, "xgboost_feature_pca_3")
validation_folder = os.path.join(model_folder, "validation")
os.makedirs(validation_folder, exist_ok=True)

# Ścieżki do zapisanych cech
feature_real_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature_EVAL", "REAL")
feature_fake_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature_EVAL", "FAKE")


def load_features_from_folder(folder_path, label):
    features_path = os.path.join(folder_path, "features.npy")
    filenames_path = os.path.join(folder_path, "filenames.npy")

    if not os.path.exists(features_path) or not os.path.exists(filenames_path):
        print(f"Brakuje danych w folderze: {folder_path}")
        return np.array([]), np.array([]), []

    X = np.load(features_path)
    filenames = np.load(filenames_path).tolist()
    y = np.full(len(X), label)

    print(f"Wczytano {len(X)} próbek z {folder_path}")
    return X, y, filenames


# Wczytanie cech
X_real, y_real, filenames_real = load_features_from_folder(feature_real_folder, 0)
X_fake, y_fake, filenames_fake = load_features_from_folder(feature_fake_folder, 1)

X_test = np.concatenate((X_real, X_fake), axis=0)
y_test = np.concatenate((y_real, y_fake), axis=0)
filenames = filenames_real + filenames_fake

# Wczytanie modelu, scalera i PCA
model_path = os.path.join(model_folder, "xgboost_model.pkl")
scaler_path = os.path.join(model_folder, "scaler.pkl")
pca_path = os.path.join(model_folder, "pca.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("Nie znaleziono modelu!")
if not os.path.exists(scaler_path):
    raise FileNotFoundError("Nie znaleziono scalera!")
if not os.path.exists(pca_path):
    raise FileNotFoundError("Nie znaleziono PCA!")

with open(model_path, "rb") as f:
    xgb_model = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
with open(pca_path, "rb") as f:
    pca = pickle.load(f)

print("Model, scaler i PCA załadowane poprawnie!")

# Normalizacja i PCA
X_test = scaler.transform(X_test)
X_test = pca.transform(X_test)

# Predykcja
y_pred = xgb_model.predict(X_test)
y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]

def plot_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Macierz Pomyłek")
    plt.savefig(os.path.join(validation_folder, "confusion_matrix.png"))
    plt.show()


def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(validation_folder, "roc_curve.png"))
    plt.show()

def plot_precision_recall_curve():
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(validation_folder, "precision_recall_curve.png"))
    plt.show()

plot_confusion_matrix()
plot_roc_curve()
plot_precision_recall_curve()

print(f"\n Walidacja zakończona. Wyniki zapisane w: {validation_folder}")
