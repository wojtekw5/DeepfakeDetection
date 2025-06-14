import os
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import csv

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

# Podział na trening/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Skalowanie
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.01,
    reg_lambda=1,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# Metryki i predykcje
y_pred = xgb_model.predict(X_test)
y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
    "AUC-ROC": roc_auc_score(y_test, y_pred_probs)
}

metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_csv(os.path.join(model_folder, "classification_metrics.csv"), index=False)

# Wykres ważności cech
def plot_feature_importance(model, feature_names, model_folder):
    # Pobierz booster i metryki: gain i weight
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")

    # całkowity gain
    total_gain = sum(gain.values())

    # gain do postaci procentowej
    normalized_gain = {}
    for k, v in gain.items():
        normalized_gain[k] = (v / total_gain) * 100

    # sortowanie po %
    sorted_items = sorted(normalized_gain.items(), key=lambda x: x[1], reverse=True)

    sorted_values = []
    sorted_features = []

    csv_rows = []

    for k, percent_gain in sorted_items:
        feature_idx = int(k[1:])  # 'f0' → 0
        feature_name = feature_names[feature_idx]
        sorted_features.append(feature_name)
        sorted_values.append(percent_gain)

        # liczba splitów (weight) – może być 0, jeśli cecha nie została użyta
        split_count = weight.get(k, 0)

        csv_rows.append({
            "Cecha": feature_name,
            "Gain (%)": round(percent_gain, 2),
            "Gain (surowy)": round(gain[k], 6),
            "Liczba splitów": int(split_count)
        })

    # Zapisz do CSV
    csv_path = os.path.join(model_folder, "feature_importance.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["Cecha", "Gain (%)", "Gain (surowy)", "Liczba splitów"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Wykres
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_values, align="center")
    plt.xlabel("Gain (%)")
    plt.ylabel("Features")
    plt.title("Udział cech w redukcji straty (XGBoost Gain %)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "feature_importance.png"))
    plt.show()



def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC-ROC']:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(model_folder, "roc_curve.png"))
    plt.show()

def plot_precision_recall_curve_fn():
    prec, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(recall, prec, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(model_folder, "precision_recall_curve.png"))
    plt.show()

def plot_confusion_matrix_fn():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Macierz Pomyłek")
    plt.savefig(os.path.join(model_folder, "confusion_matrix.png"))
    plt.show()

# Wykresy
plot_feature_importance(xgb_model, feature_names, model_folder)
plot_roc_curve()
plot_precision_recall_curve_fn()
plot_confusion_matrix_fn()

# Zapis modelu i skalera
pickle.dump(scaler, open(os.path.join(model_folder, "scaler.pkl"), "wb"))
xgb_model.save_model(os.path.join(model_folder, "xgboost_model.json"))
pickle.dump(xgb_model, open(os.path.join(model_folder, "xgboost_model.pkl"), "wb"))


print(f"\n Model i wykresy zapisane w: {model_folder}")
