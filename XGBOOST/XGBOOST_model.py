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

# Ścieżki do gotowych cech
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
real_feature_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "REAL")
fake_feature_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "FAKE")

# Wczytywanie cech i nazw
X_real = np.load(os.path.join(real_feature_path, "features.npy"))
X_fake = np.load(os.path.join(fake_feature_path, "features.npy"))
feature_names = np.load(os.path.join(real_feature_path, "feature_names.npy"))

y_real = np.zeros(X_real.shape[0])
y_fake = np.ones(X_fake.shape[0])

# Łączenie danych
X = np.concatenate((X_real, X_fake), axis=0)
y = np.concatenate((y_real, y_fake), axis=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# Unikalny folder na wyniki
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

# Trening modelu XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.02,
    subsample=0.6,
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

metrics = {
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision": precision_score(y_val, y_pred),
    "Recall": recall_score(y_val, y_pred),
    "F1-score": f1_score(y_val, y_pred),
    "AUC-ROC": roc_auc_score(y_val, y_pred_probs)
}
metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_csv(os.path.join(model_folder, "classification_metrics_val.csv"), index=False)

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

model_info = f"XGBoost | n_estimators={xgb_model.n_estimators}, max_depth=6, lr=0.01"
plot_metrics(history_df, model_info, os.path.join(model_folder, "metrics.png"))

def plot_feature_importance(model, feature_names, model_folder):

    plt.rcParams.update(plt.rcParamsDefault)

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    total_gain = sum(gain.values()) if len(gain) > 0 else 1.0
    normalized_gain = {k: (v / total_gain) * 100 for k, v in gain.items()}
    sorted_items = sorted(normalized_gain.items(), key=lambda x: x[1], reverse=True)

    sorted_values = []
    sorted_features = []
    csv_rows = []

    for k, percent_gain in sorted_items:
        feature_idx = int(k[1:])  # 'f123' -> 123
        feature_name = feature_names[feature_idx]
        sorted_features.append(feature_name)
        sorted_values.append(percent_gain)
        split_count = weight.get(k, 0)
        csv_rows.append({
            "Cecha": feature_name,
            "Gain (%)": round(percent_gain, 2),
            "Gain (surowy)": round(gain[k], 6),
            "Liczba splitów": int(split_count)
        })

    with open(os.path.join(model_folder, "feature_importance.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Cecha", "Gain (%)", "Gain (surowy)", "Liczba splitów"])
        writer.writeheader()
        writer.writerows(csv_rows)

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_values)
    plt.xlabel("Gain (%)")
    plt.ylabel("Cechy")
    plt.title("Ważność cech [GAIN] %")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "feature_importance.png"))
    plt.show()
    plt.close()

plot_feature_importance(xgb_model, feature_names, model_folder)

# Zapis modelu i skalera
pickle.dump(scaler, open(os.path.join(model_folder, "scaler.pkl"), "wb"))
xgb_model.save_model(os.path.join(model_folder, "xgboost_model.json"))
pickle.dump(xgb_model, open(os.path.join(model_folder, "xgboost_model.pkl"), "wb"))

print(f"\n Model i wykresy zapisane w: {model_folder}")
