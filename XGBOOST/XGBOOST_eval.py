import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

# Ścieżki i parametry
SAMPLE_RATE = 8000
DURATION = 7
SAMPLES = SAMPLE_RATE * DURATION

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_folder = os.path.join(desktop_path, "xgboost_feature_new_npy")
validation_folder = os.path.join(model_folder, "validation")
os.makedirs(validation_folder, exist_ok=True)

model_path = os.path.join(model_folder, "xgboost_model.pkl")
scaler_path = os.path.join(model_folder, "scaler.pkl")

# Ścieżki do zapisanych cech
feature_real_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature_EVAL", "REAL")
feature_fake_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature_EVAL", "FAKE")


# Funkcja do wczytywania cech z plików .npy
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


# Wczytanie modelu
if not os.path.exists(model_path):
    raise FileNotFoundError("Nie znaleziono modelu!")
with open(model_path, "rb") as f:
    xgb_model = pickle.load(f)
print("Model załadowany poprawnie!")

# Wczytanie danych cech
X_real, y_real, filenames_real = load_features_from_folder(feature_real_folder, 0)
X_fake, y_fake, filenames_fake = load_features_from_folder(feature_fake_folder, 1)

# Połączenie danych
X_test = np.concatenate((X_real, X_fake), axis=0)
y_test = np.concatenate((y_real, y_fake), axis=0)
filenames = filenames_real + filenames_fake

# Wczytanie scalera i normalizacja
if not os.path.exists(scaler_path):
    raise FileNotFoundError("Brak pliku scaler.pkl – uruchom ponownie trenowanie modelu!")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
print("Scaler załadowany poprawnie!")

# Normalizacja
X_test = scaler.transform(X_test)

# Predykcja
y_pred = xgb_model.predict(X_test)
y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]


# Wizualizacje
def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Macierz Pomyłek")
    plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_test, y_pred_probs, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Krzywa ROC")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


plot_confusion_matrix(y_test, y_pred, os.path.join(validation_folder, "confusion_matrix.png"))
plot_roc_curve(y_test, y_pred_probs, os.path.join(validation_folder, "roc_curve.png"))

print(f"\n Wyniki zapisane w: {validation_folder}")
