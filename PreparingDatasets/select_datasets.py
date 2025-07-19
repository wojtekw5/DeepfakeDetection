import pandas as pd
import os

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

lengths_df = pd.read_csv(os.path.join(desktop_path, "clip_durations.tsv"), sep="\t")
metadata_df = pd.read_csv(os.path.join(desktop_path, "validated.tsv"), sep="\t", low_memory=False)

# konwersja czasu trwania na sekundy
lengths_df["duration"] = lengths_df["duration[ms]"] / 1000

# mapowanie kategorii wiekowych
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
metadata_df = metadata_df.dropna(subset=["age", "gender"])
metadata_df = metadata_df[metadata_df["gender"].isin(["male_masculine", "female_feminine"])]

merged_df = pd.merge(lengths_df, metadata_df, left_on="clip", right_on="path")
filtered_df = merged_df[merged_df["duration"] <= 8].copy()
filtered_df["gender"] = filtered_df["gender"].replace({"male_masculine": "male", "female_feminine": "female"})

def categorize_age(age):
    if age < 20:
        return "below_20"
    elif age < 40:
        return "20_39"
    elif age < 60:
        return "40_59"
    else:
        return "60_plus"
filtered_df["age_group"] = filtered_df["age"].apply(categorize_age)

def categorize_length(duration):
    if duration <= 2:
        return "very_short"
    elif duration <= 4:
        return "short"
    elif duration <= 6:
        return "medium"
    else:
        return "long"
filtered_df["length_group"] = filtered_df["duration"].apply(categorize_length)

# wyrównanie liczby kobiet i mężczyzn
gender_counts = filtered_df["gender"].value_counts()
target_count = min(gender_counts.get("male", 0), gender_counts.get("female", 0), 6000)
if target_count == 0:
    print("Brak wystarczających próbek do wyrównania płci.")
    exit()

balanced_df = filtered_df.groupby("gender", group_keys=False).apply(lambda x: x.sample(min(len(x), target_count), random_state=42))

sizes = [5000, 5000, 2000]
def get_balanced_disjoint_subsets(df, sizes):
    available_samples = df.copy()
    selected_samples = []
    for size in sizes:
        sampled = available_samples.sample(size, random_state=42)
        available_samples = available_samples.loc[~available_samples.index.isin(sampled.index)]
        selected_samples.append(sampled)
    return selected_samples

dataset_1, dataset_2, dataset_3 = get_balanced_disjoint_subsets(balanced_df, sizes)

# Wybór wymaganych kolumn
columns = ["clip", "duration", "age", "gender", "age_group", "length_group"]
dataset_1 = dataset_1[columns]
dataset_2 = dataset_2[columns]
dataset_3 = dataset_3[columns]

# zapis do plików
output_paths = [
    os.path.join(desktop_path, "dataset_1.tsv"),
    os.path.join(desktop_path, "dataset_2.tsv"),
    os.path.join(desktop_path, "dataset_3.tsv"),
]

dataset_1.to_csv(output_paths[0], sep="\t", index=False)
dataset_2.to_csv(output_paths[1], sep="\t", index=False)
dataset_3.to_csv(output_paths[2], sep="\t", index=False)

print("Zbiory zostały zapisane na pulpicie.")


def compute_averages(dataset):
    avg_duration = dataset["duration"].mean()
    avg_age = dataset["age"].dropna().mean()
    gender_counts = dataset["gender"].value_counts().to_dict()
    return avg_duration, avg_age, gender_counts


avg_d1, age_d1, gender_d1 = compute_averages(dataset_1)
avg_d2, age_d2, gender_d2 = compute_averages(dataset_2)
avg_d3, age_d3, gender_d3 = compute_averages(dataset_3)

print(f"Średnia długość nagrania - Dataset 1: {avg_d1:.2f} s, Dataset 2: {avg_d2:.2f} s, Dataset 3: {avg_d3:.2f} s")
print(f"Średni wiek - Dataset 1: {age_d1:.2f}, Dataset 2: {age_d2:.2f}, Dataset 3: {age_d3:.2f}")
print("Liczba kobiet i mężczyzn w zbiorach:")
print("Dataset 1:", gender_d1)
print("Dataset 2:", gender_d2)
print("Dataset 3:", gender_d3)
