import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from sklearn.metrics import confusion_matrix, roc_curve

# Ścieżki
SAMPLE_RATE = 8000
DURATION = 7
SAMPLES = SAMPLE_RATE * DURATION

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_folder = os.path.join(desktop_path, "xgboost_feature_npy_4")
validation_folder = os.path.join(model_folder, "validation")
os.makedirs(validation_folder, exist_ok=True)

model_path = os.path.join(model_folder, "xgboost_model.pkl")
scaler_path = os.path.join(model_folder, "scaler.pkl")

eval_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "EVAL")
real_audio_folder = os.path.join(eval_folder, "REAL")
fake_audio_folder = os.path.join(eval_folder, "FAKE")

def extract_features(file):
    try:
        y, sr = librosa.load(file, sr=SAMPLE_RATE)
        y = np.pad(y, (0, max(0, SAMPLES - len(y))), mode="constant")[:SAMPLES]

        features = [
            np.mean(y), np.std(y), np.min(y), np.max(y), np.median(y),
            np.sqrt(np.mean(y ** 2)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        ]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        return np.array(features)

    except Exception as e:
        print(f"Błąd przetwarzania pliku {file}: {e}")
        return None

# Wczytanie modelu
if not os.path.exists(model_path):
    raise FileNotFoundError("Nie znaleziono modelu!")
with open(model_path, "rb") as f:
    xgb_model = pickle.load(f)
print("Model załadowany poprawnie!")

def load_audio_files_from_folder(folder_path, label):
    X, y, filenames = [], [], []
    if not os.path.exists(folder_path):
        print(f"Folder nie istnieje: {folder_path}")
        return np.array(X), np.array(y), filenames

    files = []
    for f in os.listdir(folder_path):
        if f.endswith(".wav"):
            files.append(f)

    print(f"Znaleziono {len(files)} plików w {folder_path}")

    for idx, file in enumerate(files, start=1):
        print(f'Przetwarzanie {idx}/{len(files)}: {file}')
        file_path = os.path.join(folder_path, file)
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(label)
            filenames.append(file)

    return np.array(X), np.array(y), filenames

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
    plt.plot([0, 1], [0, 1], linestyle="--", color="orange")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Krzywa ROC")
    plt.legend()
    plt.savefig(save_path)
    plt.show()

X_real, y_real, filenames_real = load_audio_files_from_folder(real_audio_folder, 0)
X_fake, y_fake, filenames_fake = load_audio_files_from_folder(fake_audio_folder, 1)

X_test = np.concatenate((X_real, X_fake), axis=0)
y_test = np.concatenate((y_real, y_fake), axis=0)

# Wczytanie scalera i normalizacja
if not os.path.exists(scaler_path):
    raise FileNotFoundError("Brak pliku scaler.pkl – uruchom ponownie trenowanie modelu!")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
print("Scaler załadowany poprawnie!")

X_test = scaler.transform(X_test)  # Normalizacja

# Predykcja
y_pred = xgb_model.predict(X_test)
y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]

# Wizualizacje
plot_confusion_matrix(y_test, y_pred, os.path.join(validation_folder, "confusion_matrix.png"))
plot_roc_curve(y_test, y_pred_probs, os.path.join(validation_folder, "roc_curve.png"))

print(f"Wyniki zapisane w: {validation_folder}")