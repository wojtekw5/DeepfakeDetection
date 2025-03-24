import librosa
import numpy as np
import os

# Parametry HiFi-GAN
sampling_rate = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
fmin = 0
fmax = 8000

input_folder = "C:/Users/wojtek/Desktop/AUDIO_PROCCESSING/2.AUDIO_TO_MELSPECTROGRAM/INPUT"
output_folder = "C:/Users/wojtek/Desktop/AUDIO_PROCCESSING/2.AUDIO_TO_MELSPECTROGRAM/OUTPUT"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(('.wav', '.mp3', '.flac')):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".npy")

        try:
            audio, sr_loaded = librosa.load(input_path, sr=sampling_rate)
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=sampling_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                power=1.0
            )

            # Logarytmowanie/unikanie zera
            log_mel_spectrogram = np.log(np.maximum(1e-5, mel_spectrogram))

            # Dodanie wymiaru batch_size / [n_mels, time_steps] -> [1, n_mels, time_steps]
            log_mel_spectrogram_with_batch = np.expand_dims(log_mel_spectrogram, axis=0)

            # zapis do pliku npy
            np.save(output_path, log_mel_spectrogram_with_batch)
            print(f"Mel-spektrogram zapisany: {output_path}, wymiary: {log_mel_spectrogram_with_batch.shape}")
        except Exception as e:
            print(f"Błąd przetwarzania pliku {file_name}: {e}")
