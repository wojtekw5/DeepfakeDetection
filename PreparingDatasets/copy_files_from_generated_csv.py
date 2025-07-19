import os
import shutil
import pandas as pd


source_folder = r"C:\Users\wojtek\Downloads\cv-corpus\cv-corpus-19.0-2024-09-13\en\clips"

destination_folders = {
    "dataset_1": r"C:\Users\wojtek\Desktop\DATASET\REAL\AUDIO",
    "dataset_2": r"C:\Users\wojtek\Desktop\DATASET\FAKE\AUDIO",
    "dataset_3": r"C:\Users\wojtek\Desktop\DATASET\EVAL\AUDIO"
}

def copy_files(tsv_file, dataset_name):
    df = pd.read_csv(os.path.join(r"C:\Users\wojtek\Desktop", tsv_file), sep="\t")

    dest_folder = destination_folders[dataset_name]
    os.makedirs(dest_folder, exist_ok=True)

    for filename in df["clip"]:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(dest_folder, filename)

        # sprawdzenie, czy plik istnieje
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
        else:
            print(f"Brak pliku: {source_path}")

copy_files("dataset_1.tsv", "dataset_1")
copy_files("dataset_2.tsv", "dataset_2")
copy_files("dataset_3.tsv", "dataset_3")

print("Kopiowanie zakończone!")
