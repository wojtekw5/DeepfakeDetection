import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

compare_path = os.path.join(os.path.expanduser("~"), "Desktop", "compare")

audio_pairs = [
    (
        os.path.join(compare_path, "common_voice_en_19475363.wav"),
        os.path.join(compare_path, "common_voice_en_19475363_VCTK_V1.wav"),
    ),
    (
        os.path.join(compare_path, "common_voice_en_19662755.wav"),
        os.path.join(compare_path, "common_voice_en_19662755_VCTK_V1.wav"),
    ),
    (
        os.path.join(compare_path, "common_voice_en_17288439.wav"),
        os.path.join(compare_path, "common_voice_en_17288439_VCTK_V1.wav"),
    ),
]

mel_diffs = []          # różnicowe mel-spektrogramy (dB)
freq_profiles = []      # (freqs_hz, mean_orig_db_per_mel, mean_hifi_db_per_mel, nazwa_pliku)
wave_data = []          # waveformy

n_fft = 1024
hop_length = 256
n_mels = 80

plt.rcParams.update({
    "font.size": 18,         # rozmiar ogólny
    "axes.titlesize": 20,    # tytuły wykresów
    "axes.labelsize": 18,    # etykiety osi
    "xtick.labelsize": 16,   # etykiety osi X
    "ytick.labelsize": 16,   # etykiety osi Y
    "legend.fontsize": 16,   # legenda
    "figure.titlesize": 20   # tytuł całej figury
})

# przetwarzanie każdej pary
for orig_path, hifi_path in audio_pairs:
    print(f"Przetwarzanie:\n  - {os.path.basename(orig_path)}\n  - {os.path.basename(hifi_path)}")

    # wczytanie audio (sampling rate bez zmian)
    y_orig, sr_orig = librosa.load(orig_path, sr=None)
    y_hifi, sr_hifi = librosa.load(hifi_path, sr=None)

    # dopasowanie długości sygnałów
    min_len = min(len(y_orig), len(y_hifi))
    y_orig = y_orig[:min_len]
    y_hifi = y_hifi[:min_len]

    # Zachowaj waveformy
    wave_data.append((y_orig, y_hifi, sr_orig, os.path.basename(orig_path)))

    # Oblicz mel-spektrogramy (moc)
    mel_orig = librosa.feature.melspectrogram(
        y=y_orig, sr=sr_orig, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_hifi = librosa.feature.melspectrogram(
        y=y_hifi, sr=sr_hifi, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # konwersja do dB
    mel_orig_db = librosa.power_to_db(mel_orig, ref=np.max)
    mel_hifi_db = librosa.power_to_db(mel_hifi, ref=np.max)

    # dopasowanie długości ramek czasowych
    min_frames = min(mel_orig_db.shape[1], mel_hifi_db.shape[1])
    mel_orig_db = mel_orig_db[:, :min_frames]
    mel_hifi_db = mel_hifi_db[:, :min_frames]

    # różnica mel-spektrogramów (dB)
    mel_diff = mel_orig_db - mel_hifi_db
    mel_diffs.append(mel_diff)

    # średnia głośność w dB dla każdego pasma Mel
    mean_orig = mel_orig_db.mean(axis=1)
    mean_hifi = mel_hifi_db.mean(axis=1)

    # częstotliwości (Hz) odpowiadające pasmom Mel
    freqs_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr_orig / 2.0)

    freq_profiles.append((freqs_hz, mean_orig, mean_hifi, os.path.basename(orig_path)))

# melspektrogramy - różnice
plt.figure(figsize=(10, 12))
for i, diff in enumerate(mel_diffs):
    plt.subplot(3, 1, i + 1)
    plt.imshow(diff, aspect="auto", origin="lower", cmap="bwr")
    plt.colorbar(label="Różnica amplitudy (dB)")
    filename = wave_data[i][3]
    plt.title(f"Różnica Mel-spektrogramów - Para {i+1} ({filename})", fontsize=16)
    plt.xlabel("Ramki", fontsize=14)
    plt.ylabel("Częstotliwości Mel", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(compare_path, "mel_compare.png"))
plt.close()


# profil częstotliwośc
plt.figure(figsize=(10, 12))
for i, (freqs_hz, mean_orig, mean_hifi, filename) in enumerate(freq_profiles):
    plt.subplot(3, 1, i + 1)
    plt.plot(freqs_hz, mean_orig, label="Oryg.", alpha=0.9)
    plt.plot(freqs_hz, mean_hifi, label="Hifi", alpha=0.9)
    plt.title(f"Profil głośności vs. częstotliwość - Para {i+1} ({filename})", fontsize=16)
    plt.xlabel("Częstotliwość [Hz]", fontsize=14)
    plt.ylabel("Średnia amplituda (dB)", fontsize=14)
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(compare_path, "freq_gain.png"))
plt.close()

# waveform
plt.figure(figsize=(10, 12))
for i, (y1, y2, sr, filename) in enumerate(wave_data):
    plt.subplot(3, 1, i + 1)
    time = np.linspace(0, len(y1) / sr, len(y1))
    plt.plot(time, y1, label="Oryg.", alpha=0.7)
    plt.plot(time, y2, label="Hifi", alpha=0.7)
    plt.title(f"Waveform - Para {i+1} ({filename})", fontsize=16)
    plt.xlabel("Czas [s]", fontsize=14)
    plt.ylabel("Amplituda", fontsize=14)
    plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(compare_path, "waveformy.png"))
plt.close()
