import os
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

# Ścieżki do cech
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
real_feature_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "REAL")
fake_feature_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "FAKE")

# Wczytywanie danych
X_real = np.load(os.path.join(real_feature_path, "features.npy"))
X_fake = np.load(os.path.join(fake_feature_path, "features.npy"))
feature_names = np.load(os.path.join(real_feature_path, "feature_names.npy"))

y_real = np.zeros(X_real.shape[0])
y_fake = np.ones(X_fake.shape[0])

X = np.concatenate((X_real, X_fake), axis=0)
y = np.concatenate((y_real, y_fake), axis=0)

# Podział na trening/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Skalowanie
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA – redukcja do 20 cech
pca = PCA(n_components=25, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Folder na model
def get_unique_folder_name(base_path):
    if not os.path.exists(base_path):
        return base_path
    counter = 1
    new_path = f"{base_path}_{counter}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}"
    return new_path

model_folder = os.path.join(desktop_path, "xgboost_feature_pca")
model_folder = get_unique_folder_name(model_folder)
os.makedirs(model_folder, exist_ok=True)


# Trening modelu
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

xgb_model.fit(X_train_pca, y_train)

# Predykcje i metryki
y_pred = xgb_model.predict(X_test_pca)
y_pred_probs = xgb_model.predict_proba(X_test_pca)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
    "AUC-ROC": roc_auc_score(y_test, y_pred_probs)
}
pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(model_folder, "classification_metrics.csv"), index=False)

# Wykresy
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC-ROC']:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(model_folder, "roc_curve.png"))
    plt.show()

def plot_precision_recall_curve():
    prec, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(recall, prec, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(model_folder, "precision_recall_curve.png"))
    plt.show()

def plot_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Macierz Pomyłek")
    plt.savefig(os.path.join(model_folder, "confusion_matrix.png"))
    plt.show()

plot_roc_curve()
plot_precision_recall_curve()
plot_confusion_matrix()

# Zapis modelu i PCA
xgb_model.save_model(os.path.join(model_folder, "xgboost_model.json"))
pickle.dump(xgb_model, open(os.path.join(model_folder, "xgboost_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(model_folder, "scaler.pkl"), "wb"))
pickle.dump(pca, open(os.path.join(model_folder, "pca.pkl"), "wb"))

print(f"\n Model wytrenowany. Wszystko zapisane w: {model_folder}")
