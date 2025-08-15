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
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
)
import time

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

pca_folder = os.path.join(model_folder, "pca_analysis")
os.makedirs(pca_folder, exist_ok=True)


# PCA analiza
explained_ratios = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_ratios)
n_components_95 = pca.n_components_

# PCA Explainability do CSV
pca_explain_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(explained_ratios))],
    "ExplainedVariance": explained_ratios
})

pca_explain_df = pca_explain_df.head(35)

# + noise variance i suma
extra_rows = pd.DataFrame({
    "PC": ["NoiseVariance", "SumExplainedVariance", "NComponents_95pct"],
    "ExplainedVariance": [pca.noise_variance_,
                          np.sum(explained_ratios),
                          pca.n_components_]
})

pca_explain_df = pd.concat([pca_explain_df, extra_rows], ignore_index=True)

# Zapis do CSV
pca_explain_df.to_csv(os.path.join(pca_folder, "pca_explainability.csv"), index=False)


print(f"\nNoise variance: {pca.noise_variance_:.4f}")
print(f"Suma explained variance: {np.sum(explained_ratios):.4f}")
print(f"Liczba komponentów odpowiadających za 95% wariancji: {n_components_95}")

# Główne cechy w PC1 i PC2
components = pca.components_
pc1_weights = components[0]
pc2_weights = components[1]

pc1_df = pd.Series(pc1_weights, index=feature_names).abs().sort_values(ascending=False)
pc2_df = pd.Series(pc2_weights, index=feature_names).abs().sort_values(ascending=False)

pc1_df.head(20).to_csv(os.path.join(pca_folder, "top_features_PC1.csv"))
pc2_df.head(20).to_csv(os.path.join(pca_folder, "top_features_PC2.csv"))

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
plt.savefig(os.path.join(pca_folder, "explained_variance_pca.png"))
plt.show()

# Skumulowany wpływ cech
n_top_components = 20
cumulative_weights = np.abs(pca.components_[:n_top_components, :]).sum(axis=0)
cumulative_df = pd.DataFrame({
    'Cecha': feature_names,
    'CumulativeInfluence': cumulative_weights
}).sort_values(by="CumulativeInfluence", ascending=False)

cumulative_df.to_csv(os.path.join(pca_folder, "cumulative_feature_influence.csv"), index=False)

# Zmiana wpływu na %
cumulative_df["PercentInfluence"] = 100 * cumulative_df["CumulativeInfluence"] / cumulative_df["CumulativeInfluence"].sum()
cumulative_df.to_csv(os.path.join(pca_folder, "percent_feature_influence.csv"), index=False)

# Wykres procentowego wpływu
plt.figure(figsize=(10, len(feature_names) * 0.3))
sns.barplot(
    data=cumulative_df.sort_values(by="PercentInfluence", ascending=False),
    y='Cecha',
    x='PercentInfluence',
    color='steelblue'
)
plt.title('Skumulowany wpływ cech na pierwsze 20 komponentów PCA w %')
plt.xlabel('Wpływ [%]')
plt.ylabel('Cecha')
plt.tight_layout()
plt.savefig(os.path.join(pca_folder, "percent_feature_influence.png"))
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
plt.savefig(os.path.join(pca_folder, "feature_vs_pc_heatmap.png"))
plt.show()


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

xgb_model.fit(X_train_pca, y_train)

training_time = time.time() - start_time
print(f"Czas treningu: {training_time:.2f} sekund ({training_time/60:.2f} minut)")


pd.DataFrame({
    "training_time_sec": [training_time],
    "training_time_min": [training_time / 60]
}).to_csv(os.path.join(model_folder, "training_time.csv"), index=False)

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

    proba_tr = xgb_model.predict_proba(X_train_pca, iteration_range=(0, i))[:, 1]
    proba_va = xgb_model.predict_proba(X_test_pca,  iteration_range=(0, i))[:, 1]

    pred_tr  = (proba_tr > 0.5).astype(int)
    pred_va  = (proba_va > 0.5).astype(int)

    history["epoch"].append(i)
    history["train_accuracy"].append(accuracy_score(y_train, pred_tr))
    history["val_accuracy"].append(accuracy_score(y_test,  pred_va))
    history["train_precision"].append(precision_score(y_train, pred_tr, zero_division=0))
    history["val_precision"].append(precision_score(y_test,  pred_va, zero_division=0))
    history["train_recall"].append(recall_score(y_train, pred_tr))
    history["val_recall"].append(recall_score(y_test,  pred_va))
    history["train_f1_score"].append(f1_score(y_train, pred_tr))
    history["val_f1_score"].append(f1_score(y_test,  pred_va))
    history["train_auc"].append(roc_auc_score(y_train, proba_tr))
    history["val_auc"].append(roc_auc_score(y_test,  proba_va))
    history["train_loss"].append(log_loss(y_train, proba_tr, labels=[0, 1]))
    history["val_loss"].append(log_loss(y_test,  proba_va, labels=[0, 1]))

history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(model_folder, "training_history.csv"), index=False)


def plot_metrics(df, save_path):
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 20
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
    epochs = df["epoch"].values

    for i, (key, title) in enumerate(metrics):
        ax = axes[i]
        ax.plot(epochs, df[f"train_{key}"], label="Treningowa", marker="o")
        ax.plot(epochs, df[f"val_{key}"],   label="Walidacyjna", marker="o")
        ax.set_title(title)
        ax.set_xlabel("Iteracje boostingu")
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

plot_metrics(history_df, os.path.join(model_folder, "metrics.png"))


# Zapis modelu
pickle.dump(xgb_model, open(os.path.join(model_folder, "xgboost_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(model_folder, "scaler.pkl"), "wb"))
pickle.dump(pca, open(os.path.join(model_folder, "pca.pkl"), "wb"))

print(f"\n Model wytrenowany. Wszystko zapisane w: {model_folder}")
