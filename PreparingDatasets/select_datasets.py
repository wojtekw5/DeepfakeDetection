import pandas as pd
import os


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
lengths_df = pd.read_csv(os.path.join(desktop_path, "clip_durations.tsv"), sep="\t")
metadata_df = pd.read_csv(os.path.join(desktop_path, "validated.tsv"), sep="\t", low_memory=False)

# konwersja czasu z ms na s
lengths_df["duration"] = lengths_df["duration[ms]"] / 1000

# Mapowanie wieku
age_mapping = {
    "teens": 15,
    "twenties": 25,
    "thirties": 35,
    "fourties": 45,
    "fifties": 55,
    "sixties": 65,
    "seventies": 75,
    "eighties": 85
}
metadata_df["age"] = metadata_df["age"].map(age_mapping)

# usuwanie brakujących danych i filtracja płci
metadata_df = metadata_df.dropna(subset=["age", "gender"])
metadata_df = metadata_df[metadata_df["gender"].isin(["male_masculine", "female_feminine"])]

# połączenie danych
df = pd.merge(lengths_df, metadata_df, left_on="clip", right_on="path")
df = df[df["duration"] <= 8].copy()
df["gender"] = df["gender"].replace({"male_masculine": "male", "female_feminine": "female"})

# grupowanie wieku
def categorize_age(age):
    if age < 20:
        return "below_20"
    elif age < 40:
        return "20_39"
    elif age < 60:
        return "40_59"
    else:
        return "60_plus"
df["age_group"] = df["age"].apply(categorize_age)

# tworzenie grupy: gender + age_group
df["group"] = df["gender"] + "_" + df["age_group"]

target_per_group = 1500 # 1500 * 8(gender*age) = 12000 potrzebnych próbek

samples = []
for group, group_df in df.groupby("group"):
    sample = group_df.sample(target_per_group, random_state=42)
    samples.append(sample)
samples = pd.concat(samples, ignore_index=True)

samples = samples.reset_index(drop=True)

def categorize_length(duration):
    if duration <= 2:
        return "very_short"
    elif duration <= 4:
        return "short"
    elif duration <= 6:
        return "medium"
    else:
        return "long"
samples["length_group"] = samples["duration"].apply(categorize_length)

# podział na zbiory
def split_sets(df, sizes):
    sets = []
    available = df.copy()
    for size in sizes:
        part = available.sample(size, random_state=42)
        available = available.drop(part.index)
        sets.append(part)
    return sets

dataset_1, dataset_2, dataset_3 = split_sets(samples, [5000, 5000, 2000])

# zapis plików
columns = ["clip", "duration", "age", "gender", "age_group", "length_group"]
for i, dataset in enumerate([dataset_1, dataset_2, dataset_3], 1):
    path = os.path.join(desktop_path, f"dataset_{i}.tsv")
    dataset[columns].to_csv(path, sep="\t", index=False)

print("Zbiory zapisane na pulpicie.")

# Statystyki
def stats(dataset):
    return {
        "Średnia długość": f"{dataset['duration'].mean():.2f} s",
        "Średni wiek": f"{dataset['age'].mean():.2f}",
        "Płeć": dataset["gender"].value_counts(normalize=True).mul(100).round(1).to_dict(),
        "Wiek": dataset["age_group"].value_counts(normalize=True).mul(100).round(1).to_dict(),
        "Długość": dataset["length_group"].value_counts(normalize=True).mul(100).round(1).to_dict()
    }

# statystyki dla dataset_1
print("\n Dataset 1")
stats_1 = stats(dataset_1)
for k, v in stats_1.items():
    print(f"{k}: {v}")

# statystyki dla dataset_2
print("\n Dataset 2")
stats_2 = stats(dataset_2)
for k, v in stats_2.items():
    print(f"{k}: {v}")

# statystyki dla dataset_3
print("\n Dataset 3")
stats_3 = stats(dataset_3)
for k, v in stats_3.items():
    print(f"{k}: {v}")

def show_group_distribution(dataset, label):
    print(f"\n Procentowy udział grup demograficznych w {label}:")
    group_percent = dataset["gender"] + "_" + dataset["age_group"]
    percent = group_percent.value_counts(normalize=True).sort_index() * 100
    percent = percent.round(2)
    print(percent.to_string())

show_group_distribution(dataset_1, "Dataset 1")
show_group_distribution(dataset_2, "Dataset 2")
show_group_distribution(dataset_3, "Dataset 3")
