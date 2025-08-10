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

# Ścieżki i dane
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Skalowanie
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Folder wyników
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

# PCA analiza
explained_ratios = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_ratios)
n_components_95 = pca.n_components_

print("\n--- PCA Explainability ---")
i = 0
while i < len(explained_ratios) and i < 35:
    print("PC{}: {:.4f}".format(i + 1, explained_ratios[i]))
    i += 1

print(f"\nNoise variance: {pca.noise_variance_:.4f}")
print(f"Sum of explained variance: {np.sum(explained_ratios):.4f}")
print(f"Liczba komponentów odpowiadających za 95% wariancji: {n_components_95}")

# Główne cechy w PC1 i PC2
components = pca.components_
pc1_weights = components[0]
pc2_weights = components[1]

pc1_df = pd.Series(pc1_weights, index=feature_names).abs().sort_values(ascending=False)
pc2_df = pd.Series(pc2_weights, index=feature_names).abs().sort_values(ascending=False)

print("\nTop 10 cech wpływających na PC1:")
print(pc1_df.head(10))
print("\nTop 10 cech wpływających na PC2:")
print(pc2_df.head(10))

pc1_df.head(20).to_csv(os.path.join(model_folder, "top_features_PC1.csv"))
pc2_df.head(20).to_csv(os.path.join(model_folder, "top_features_PC2.csv"))

# Wykres explained variance
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_ratios) + 1), explained_ratios, alpha=0.7, label="Wyjaśniona wariancja")
plt.plot(range(1, len(explained_ratios) + 1), cumulative_var, marker='o', label="Skumulowana wyjaśniona\nwariancja")
plt.axhline(y=0.95, color='r', linestyle='--', label="Próg 95%")
plt.xlabel("Komponenty (PCA)")
plt.ylabel("Stosunek wyjaśnionej wariancji")
plt.title("Udział wyjaśnionej wariancji przez składowe PCA")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(model_folder, "explained_variance_pca.png"))
plt.show()

# Skumulowany wpływ cech
n_top_components = 20
cumulative_weights = np.abs(pca.components_[:n_top_components, :]).sum(axis=0)
cumulative_df = pd.DataFrame({
    'Cecha': feature_names,
    'CumulativeInfluence': cumulative_weights
}).sort_values(by="CumulativeInfluence", ascending=False)

print("\nTop 10 cech wg skumulowanego wpływu na pierwsze 20 komponentów PCA:")
print(cumulative_df.head(10))

cumulative_df.to_csv(os.path.join(model_folder, "cumulative_feature_influence.csv"), index=False)

# Zmiana wpływu na %
cumulative_df["PercentInfluence"] = 100 * cumulative_df["CumulativeInfluence"] / cumulative_df["CumulativeInfluence"].sum()
cumulative_df.to_csv(os.path.join(model_folder, "percent_feature_influence.csv"), index=False)

# Wykres procentowego wpływu
plt.figure(figsize=(10, len(feature_names) * 0.3))
sns.barplot(
    data=cumulative_df.sort_values(by="PercentInfluence", ascending=False),
    y='Feature',
    x='PercentInfluence',
    color='steelblue'
)
plt.title('Skumulowany wpływ cech na pierwsze 20 komponentów PCA w %')
plt.xlabel('Wpływ [%]')
plt.ylabel('Cecha')
plt.tight_layout()
plt.savefig(os.path.join(model_folder, "percent_feature_influence.png"))
plt.show()

# Wpływ cech na wszystkie PCA
pc_columns = []
pc_idx = 1
while pc_idx <= pca.n_components_:
    pc_columns.append(f"PC{pc_idx}")
    pc_idx += 1

component_weights = pd.DataFrame(
    np.abs(pca.components_.T),
    index=feature_names,
    columns=pc_columns
)

plt.figure(figsize=(12, len(feature_names) * 0.25))
sns.heatmap(component_weights, cmap="coolwarm", center=0)
plt.title("Wpływ cech na główne składowe PCA")
plt.xlabel("Składowa główna (PC)")
plt.ylabel("Oryginalna cecha")
plt.tight_layout()
plt.savefig(os.path.join(model_folder, "feature_vs_pc_heatmap.png"))
plt.show()

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
    eval_metric="logloss"
)

xgb_model.fit(X_train_pca, y_train)

# Metryki końcowe
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


history = {"epoch": [], "accuracy": [], "precision": [], "recall": [], "f1_score": [], "auc": []}
for i in range(1, xgb_model.n_estimators + 1):
    y_proba_iter = xgb_model.predict_proba(X_test_pca, iteration_range=(0, i))[:, 1]
    y_pred_iter = (y_proba_iter > 0.5).astype(int)
    history["epoch"].append(i)
    history["accuracy"].append(accuracy_score(y_test, y_pred_iter))
    history["precision"].append(precision_score(y_test, y_pred_iter, zero_division=0))
    history["recall"].append(recall_score(y_test, y_pred_iter))
    history["f1_score"].append(f1_score(y_test, y_pred_iter))
    history["auc"].append(roc_auc_score(y_test, y_proba_iter))

history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(model_folder, "training_history.csv"), index=False)


def plot_single_metric(df, metric, title, filename):
    plt.figure()
    plt.plot(df["epoch"], df[metric], label=metric.capitalize())
    plt.xlabel("Iteracja Boostingu")
    plt.ylabel("Wartość metryki")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, filename))
    plt.show()

def plot_metric_pair(df, metric1, metric2, title, filename):
    plt.figure()
    plt.plot(df["epoch"], df[metric1], label=metric1.capitalize())
    plt.plot(df["epoch"], df[metric2], label=metric2.capitalize())
    plt.xlabel("Iteracja Boostingu")
    plt.ylabel("Wartość metryki")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, filename))
    plt.show()

plot_single_metric(history_df, "accuracy", "Accuracy — XGBOOST PCA", "accuracy_history.png")
plot_metric_pair(history_df, "precision", "recall", "Precision i Recall — XGBOOST PCA", "precision_recall_history.png")
plot_metric_pair(history_df, "f1_score", "auc", "F1 i AUC — XGBOOST PCA", "f1_auc_history.png")

# Wykresy końcowe
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC-ROC']:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Fałszywie pozytywne (%)")
    plt.ylabel("Prawdziwie pozytywne (%)")
    plt.title("XGBOOST PCA")
    plt.legend()
    plt.savefig(os.path.join(model_folder, "roc_curve.png"))
    plt.show()

def plot_precision_recall_curve():
    prec, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    plt.figure()
    plt.plot(recall, prec, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall XGBOOST PCA")
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

# Zapis modelu
xgb_model.save_model(os.path.join(model_folder, "xgboost_model.json"))
pickle.dump(xgb_model, open(os.path.join(model_folder, "xgboost_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(model_folder, "scaler.pkl"), "wb"))
pickle.dump(pca, open(os.path.join(model_folder, "pca.pkl"), "wb"))

print(f"\n Model wytrenowany. Wszystko zapisane w: {model_folder}")
