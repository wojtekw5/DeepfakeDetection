import os
import librosa
import numpy as np

SAMPLE_RATE = 8000
DURATION = 7
SAMPLES = int(SAMPLE_RATE * DURATION)

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
real_audio_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "TRAIN", "REAL")
fake_audio_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "TRAIN", "FAKE")

real_output_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "REAL")
fake_output_folder = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "FAKE")
os.makedirs(real_output_folder, exist_ok=True)
os.makedirs(fake_output_folder, exist_ok=True)

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

def process_and_save(folder_path, output_folder):
    X, filenames, feature_names = [], [], None

    if not os.path.exists(folder_path):
        print(f"Folder nie istnieje: {folder_path}")
        return

    files = []
    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)
        if os.path.isfile(file_path):
            files.append(f)

    total_files = len(files)
    print(f"Znaleziono {total_files} plików w {folder_path}")

    for idx, filename in enumerate(files, 1):
        file_path = os.path.join(folder_path, filename)
        features, feature_names = extract_features(file_path)
        if features is not None:
            X.append(features)
            filenames.append(filename)
        print(f"Przetworzono {idx}/{total_files} plików")

    if X:
        np.save(os.path.join(output_folder, "features.npy"), np.array(X))
        np.save(os.path.join(output_folder, "feature_names.npy"), np.array(feature_names))
        np.save(os.path.join(output_folder, "filenames.npy"), np.array(filenames))
        print(f"Cechy zapisane w: {output_folder}")
    else:
        print(f"Brak poprawnych danych w folderze: {folder_path}")

# Przetwarzanie REAL i FAKE
process_and_save(real_audio_folder, real_output_folder)
process_and_save(fake_audio_folder, fake_output_folder)

print("\n Zakończono ekstrakcję i zapis cech.")
