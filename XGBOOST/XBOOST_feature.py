import os
import librosa
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

SAMPLE_RATE = 8000
DURATION = 7
SAMPLES = int(SAMPLE_RATE * DURATION)

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
real_audio_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "TRAIN", "REAL")
fake_audio_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "TRAIN", "FAKE")

def get_unique_folder_name(base_path):
    if not os.path.exists(base_path):
        return base_path
    counter = 1
    new_path = f"{base_path}_{counter}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}"
    return new_path

model_folder = os.path.join(desktop_path, "xgboost_feature_npy")
model_folder = get_unique_folder_name(model_folder)
os.makedirs(model_folder, exist_ok=True)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = np.pad(y, (0, max(0, SAMPLES - len(y))), mode="constant")[:SAMPLES]

        features = [
            ("Mean", np.mean(y)),
            ("Std", np.std(y)),
            ("Min", np.min(y)),
            ("Max", np.max(y)),
            ("Median", np.median(y)),
            ("RMS", np.sqrt(np.mean(y ** 2))),
            ("ZeroCrossingRate", np.mean(librosa.feature.zero_crossing_rate(y))),
            ("SpectralCentroid", np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            ("SpectralRolloff", np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
            ("SpectralBandwidth", np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        ]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features.append((f"MFCC {i+1}", np.mean(mfcc[i, :])))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features.append((f"Chroma {i+1}", np.mean(chroma[i, :])))

        values = []
		names = []
		
		for f in features:
			values.append(f[1])
			names.append(f[0])
			
		return values, names


    except Exception as e:
        print(f"Błąd przetwarzania {file_path}: {e}")
        return None, None

def load_audio_files_from_folder(folder_path, label):
    X, y, filenames, feature_names = [], [], [], None
    if not os.path.exists(folder_path):
        print(f"Folder nie istnieje: {folder_path}")
        return X, y, filenames, feature_names

    files = []
    for f in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, f)):
            files.append(f)

    total_files = len(files)
    print(f"Znaleziono {total_files} plików w {folder_path}")

    for idx, filename in enumerate(files, start=1):
        file_path = os.path.join(folder_path, filename)
        features, feature_names = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(label)
            filenames.append(filename)
        print(f"Przetworzono {idx}/{total_files} plików")

    return np.array(X), np.array(y), filenames, feature_names

X_real, y_real, filenames_real, feature_names = load_audio_files_from_folder(real_audio_folder, 0)
X_fake, y_fake, filenames_fake, _ = load_audio_files_from_folder(fake_audio_folder, 1)

X = np.concatenate((X_real, X_fake), axis=0)
y = np.concatenate((y_real, y_fake), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.01,
    reg_lambda=1,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

def plot_feature_importance(model, feature_names, model_folder):
    importance = model.get_booster().get_score(importance_type="gain")

    # sortowanie cech wg ważności
    sorted_idx = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Pobranie wartości ważności cech
    sorted_values = []
    for x in sorted_idx:
        sorted_values.append(x[1])

    # Pobranie indeksów cech z formatu 'f0', 'f1', 'f2' itd.
    feature_indices = []
    for k, _ in sorted_idx:
        feature_indices.append(int(k[1:]))

    # Zamiana indeksów cech na rzeczywiste nazwy
    sorted_features = []
    for i in feature_indices:
        sorted_features.append(feature_names[i])

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_values, align="center")
    plt.xlabel("Feature Importance (Gain)")
    plt.ylabel("Features")
    plt.title("Ważność cech w modelu XGBoost")
    plt.gca().invert_yaxis()  # najważniejsze cechy na górze

    importance_path = os.path.join(model_folder, "feature_importance.png")
    plt.savefig(importance_path)
    plt.show()


y_pred = xgb_model.predict(X_test)
y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
    "AUC-ROC": roc_auc_score(y_test, y_pred_probs)
}

metrics_df = pd.DataFrame(metrics, index=[0])
metrics_csv_path = os.path.join(model_folder, "classification_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)


def plot_roc_curve(y_test, y_pred_probs, model_folder):
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['AUC-ROC']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="orange")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Krzywa ROC")
    plt.legend()
    roc_curve_path = os.path.join(model_folder, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.show()


def plot_precision_recall_curve(y_test, y_pred_probs, model_folder):
    prec, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(recall, prec, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Krzywa Precision-Recall")
    plt.legend()
    precision_recall_curve_path = os.path.join(model_folder, "precision_recall_curve.png")
    plt.savefig(precision_recall_curve_path)
    plt.show()

def plot_confusion_matrix(y_test, y_pred, model_folder):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Macierz Pomyłek")
    confusion_matrix_path = os.path.join(model_folder, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.show()

plot_feature_importance(xgb_model, feature_names, model_folder)
plot_roc_curve(y_test, y_pred_probs, model_folder)
plot_precision_recall_curve(y_test, y_pred_probs, model_folder)
plot_confusion_matrix(y_test, y_pred, model_folder)

# zapis scaleru
scaler_path = os.path.join(model_folder, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

# zapis modelu
xgb_model.save_model(os.path.join(model_folder, "xgboost_model.json"))
pickle.dump(xgb_model, open(os.path.join(model_folder, "xgboost_model.pkl"), "wb"))
print(f"Model i wykresy zapisane w: {model_folder}")


