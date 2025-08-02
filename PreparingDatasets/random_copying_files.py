import os
import shutil
import random

source_folder = r"C:\Users\wojtek\Desktop\DATASET\FAKE\AUDIO_BEFORE_SELECTION"
target_folders = [
    r"C:\Users\wojtek\Desktop\DATASET\FAKE\AUDIO_AFTER_SELECTION\LJ_1",
    r"C:\Users\wojtek\Desktop\DATASET\FAKE\AUDIO_AFTER_SELECTION\LJ_2",
    r"C:\Users\wojtek\Desktop\DATASET\FAKE\AUDIO_AFTER_SELECTION\UNIVERSAL",
    r"C:\Users\wojtek\Desktop\DATASET\FAKE\AUDIO_AFTER_SELECTION\VCTK_1",
    r"C:\Users\wojtek\Desktop\DATASET\FAKE\AUDIO_AFTER_SELECTION\VCTK_2"
]

# Upewnij się, że foldery istnieją
for folder in target_folders:
    os.makedirs(folder, exist_ok=True)

# Zbierz listę plików
files = []
for f in os.listdir(source_folder):
    full_path = os.path.join(source_folder, f)
    if os.path.isfile(full_path):
        files.append(f)

# Sprawdź, czy da się podzielić pliki równo na 5 części
total_files = len(files)
folder_count = len(target_folders)

# Pomieszaj pliki
random.shuffle(files)

# Rozdziel równo i kopiuj
chunk_size = total_files // folder_count

for i in range(folder_count):
    start_index = i * chunk_size
    end_index = start_index + chunk_size
    current_files = files[start_index:end_index]

    for file_name in current_files:
        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(target_folders[i], file_name)
        shutil.copy(src_path, dst_path)

print("Pliki audio zostały równomiernie i losowo skopiowane.")
