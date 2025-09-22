import os
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
)
import csv
import time

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
real_feature_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "REAL")
fake_feature_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "FAKE")

X_real = np.load(os.path.join(real_feature_path, "features.npy"))
X_fake = np.load(os.path.join(fake_feature_path, "features.npy"))
feature_names = np.load(os.path.join(real_feature_path, "feature_names.npy"))

y_real = np.zeros(X_real.shape[0])
y_fake = np.ones(X_fake.shape[0])

X = np.concatenate((X_real, X_fake), axis=0)
y = np.concatenate((y_real, y_fake), axis=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

def get_unique_folder_name(base_path):
    if not os.path.exists(base_path):
        return base_path
    counter = 1
    new_path = f"{base_path}_{counter}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}"
    return new_path

model_folder = os.path.join(desktop_path, "xgboost_feature_new_npy")
model_folder = get_unique_folder_name(model_folder)
os.makedirs(model_folder, exist_ok=True)

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.6,
    gamma=0.7,
    reg_alpha=0.1,
    reg_lambda=7.0,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    use_label_encoder=False,
    eval_metric="logloss"
)
start_time = time.time()

xgb_model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"Czas treningu: {training_time:.2f} sekund ({training_time/60:.2f} minut)")


time_df = pd.DataFrame({
    "training_time_sec": [training_time],
    "training_time_min": [training_time / 60]
})
time_df.to_csv(os.path.join(model_folder, "training_time.csv"), index=False)

# Metryki końcowe na walidacji (zapis do CSV)
y_pred = xgb_model.predict(X_val)
y_pred_probs = xgb_model.predict_proba(X_val)[:, 1]

history = {
    "epoch": [],
    "train_accuracy": [], "val_accuracy": [],
    "train_precision": [], "val_precision": [],
    "train_recall": [], "val_recall": [],
    "train_f1_score": [], "val_f1_score": [],
    "train_auc": [], "val_auc": [],
    "train_loss": [], "val_loss": []
}

for i in range(1, xgb_model.n_estimators + 1):
    # predykcje PROB do metryk ciągłych
    proba_tr = xgb_model.predict_proba(X_train, iteration_range=(0, i))[:, 1]
    proba_va = xgb_model.predict_proba(X_val,   iteration_range=(0, i))[:, 1]
    # predykcje binarne przy progu 0.5
    pred_tr  = (proba_tr > 0.5).astype(int)
    pred_va  = (proba_va > 0.5).astype(int)

    history["epoch"].append(i)
    # accuracy
    history["train_accuracy"].append(accuracy_score(y_train, pred_tr))
    history["val_accuracy"].append(accuracy_score(y_val,   pred_va))
    # precision
    history["train_precision"].append(precision_score(y_train, pred_tr, zero_division=0))
    history["val_precision"].append(precision_score(y_val,   pred_va, zero_division=0))
    # recall
    history["train_recall"].append(recall_score(y_train, pred_tr))
    history["val_recall"].append(recall_score(y_val,   pred_va))
    # f1
    history["train_f1_score"].append(f1_score(y_train, pred_tr))
    history["val_f1_score"].append(f1_score(y_val,   pred_va))
    # auc
    history["train_auc"].append(roc_auc_score(y_train, proba_tr))
    history["val_auc"].append(roc_auc_score(y_val,   proba_va))
    # logloss
    history["train_loss"].append(log_loss(y_train, proba_tr, labels=[0, 1]))
    history["val_loss"].append(log_loss(y_val,   proba_va, labels=[0, 1]))

history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(model_folder, "training_history_train_val.csv"), index=False)

def plot_metrics(history_df, model_info, save_path):

    #powiększenie czcionek
    plt.rcParams.update({
        "font.size": 18,         # rozmiar ogólny
        "axes.titlesize": 20,    # tytuły wykresów
        "axes.labelsize": 18,    # etykiety osi
        "xtick.labelsize": 16,   # etykiety osi X
        "ytick.labelsize": 16,   # etykiety osi Y
        "legend.fontsize": 16,   # legenda
        "figure.titlesize": 20   # tytuł całej figury
    })

    metrics = [
        ("accuracy", "Accuracy"),
        ("loss", "Loss"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
        ("auc", "AUC"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    epochs = history_df["epoch"].values

    for i, (metric, title) in enumerate(metrics):
        axes[i].plot(epochs, history_df[f"train_{metric}"], label="Treningowa", marker="o")
        axes[i].plot(epochs, history_df[f"val_{metric}"],   label="Walidacyjna", marker="o")
        axes[i].set_title(title)
        axes[i].set_xlabel("Epochs")
        axes[i].set_ylabel(title)
        axes[i].grid(True)
        axes[i].legend()

    fig.suptitle(model_info, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path)
    plt.show()
    plt.close()

model_info = f"n_estimators={xgb_model.n_estimators}, max_depth=3, lr=0.02"
plot_metrics(history_df, model_info, os.path.join(model_folder, "metrics.png"))

def plot_feature_importance_total_gain(model, feature_names, model_folder):

    plt.rcParams.update(plt.rcParamsDefault)

    booster = model.get_booster()

    # suma redukcji straty
    total_gain_dict = booster.get_score(importance_type="total_gain")
    # liczba splitów
    weight = booster.get_score(importance_type="weight")

    # normalizacja do %
    total_sum = sum(total_gain_dict.values()) if len(total_gain_dict) > 0 else 1.0
    normalized_total_gain = {k: (v / total_sum) * 100 for k, v in total_gain_dict.items()}

    # sortowanie malejąco
    sorted_items = sorted(normalized_total_gain.items(), key=lambda x: x[1], reverse=True)

    sorted_values = []
    sorted_features = []
    csv_rows = []

    for k, percent_total_gain in sorted_items:
        feature_idx = int(k[1:])  # 'f123' -> 123
        feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else k

        sorted_features.append(feature_name)
        sorted_values.append(percent_total_gain)

        split_count = int(weight.get(k, 0))
        raw_total_gain = float(total_gain_dict[k])

        csv_rows.append({
            "Cecha": feature_name,
            "Total Gain (%)": round(percent_total_gain, 2),
            "Total Gain (surowy)": round(raw_total_gain, 6),
            "Liczba splitów": split_count
        })

    # zapis do CSV
    os.makedirs(model_folder, exist_ok=True)
    csv_path = os.path.join(model_folder, "feature_importance_total_gain.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Cecha", "Total Gain (%)", "Total Gain (surowy)", "Liczba splitów"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # wykres
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_values)
    plt.xlabel("Total Gain (%)")
    plt.ylabel("Cechy")
    plt.title("Ważność cech [TOTAL GAIN] – udział w całkowitej redukcji loss")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig_path = os.path.join(model_folder, "feature_importance_total_gain.png")
    plt.savefig(fig_path)
    plt.show()
    plt.close()

plot_feature_importance_total_gain(xgb_model, feature_names, model_folder)


# zapis modelu i skalera
pickle.dump(scaler, open(os.path.join(model_folder, "scaler.pkl"), "wb"))
pickle.dump(xgb_model, open(os.path.join(model_folder, "xgboost_model.pkl"), "wb"))

print(f"\n Model i wykresy zapisane w: {model_folder}")