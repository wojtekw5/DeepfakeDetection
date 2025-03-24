import os
import librosa
import numpy as np
import librosa.display

SAMPLE_RATE = 22050
N_MELS = 80
HOP_LENGTH = 512
FRAMES = 264

dataset_path = r"C:\Users\wojtek\Desktop\DATASET\EVAL"

real_audio_folder = os.path.join(dataset_path, "REAL", "AUDIO_FINAL")
real_mel_folder = os.path.join(dataset_path, "REAL", "MELSKEKTROGRAM")

fake_audio_folder = os.path.join(dataset_path, "FAKE", "AUDIO_FAKE_FINAL")
fake_mel_folder = os.path.join(dataset_path, "FAKE", "MELSKEKTROGRAM")

os.makedirs(real_mel_folder, exist_ok=True)
os.makedirs(fake_mel_folder, exist_ok=True)


def audio_to_mel(file_path):

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)

    # konwersja do skali dB
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # dopasowanie długości (padding / cropping)
    if mel_spectrogram.shape[1] < FRAMES:
        # jeśli za krótki → uzupełnia zerami
        pad_width = FRAMES - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spectrogram.shape[1] > FRAMES:
        # jeśli za długi → ucina
        mel_spectrogram = mel_spectrogram[:, :FRAMES]

    # dodanie wymiaru dla CNN: (80, 264, 1)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)

    return mel_spectrogram


def process_audio_folder(input_folder, output_folder):

    for file_name in os.listdir(input_folder):
        if file_name.endswith((".wav", ".mp3", ".flac")):
            file_path = os.path.join(input_folder, file_name)

            mel_spectrogram = audio_to_mel(file_path)

            output_file = os.path.join(output_folder,
                                       file_name.replace(".wav", ".npy").replace(".mp3", ".npy").replace(".flac",
                                                                                                         ".npy"))
            np.save(output_file, mel_spectrogram)

            print(f"Zapisano: {output_file}")


print("Przetwarzanie REAL...")
process_audio_folder(real_audio_folder, real_mel_folder)

print("Przetwarzanie FAKE...")
process_audio_folder(fake_audio_folder, fake_mel_folder)

print("Wszystkie pliki skonwertowane!")
