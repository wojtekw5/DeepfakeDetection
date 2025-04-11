import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight, shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

# Ścieżki do danych cech
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
real_feature_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "REAL")
fake_feature_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature", "FAKE")

# Parametry
batch_size = 8
epochs = 20

# Folder zapisu
def get_unique_folder_name(base_path):
    if not os.path.exists(base_path):
        return base_path
    counter = 1
    new_path = f"{base_path}_{counter}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}"
    return new_path

model_folder = os.path.join(desktop_path, "mlp_model_npy")
model_folder = get_unique_folder_name(model_folder)
os.makedirs(model_folder, exist_ok=True)
print(f"⚪ Wyniki będą zapisane w: {model_folder}")

# Wczytywanie cech
X_real = np.load(os.path.join(real_feature_path, "features.npy"))
X_fake = np.load(os.path.join(fake_feature_path, "features.npy"))

y_real = np.zeros(X_real.shape[0])
y_fake = np.ones(X_fake.shape[0])

X = np.concatenate((X_real, X_fake), axis=0)
y = np.concatenate((y_real, y_fake), axis=0)

# Shuffle i normalizacja
X, y = shuffle(X, y, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# class_weights
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Datasety
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# Własna metryka F1
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

input_dim = X.shape[1]
model = tf.keras.Sequential([
    Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(input_dim,)),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),

    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), F1Score()]
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# Trening
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr]
)

# Zapis modelu i skalera
model.save(os.path.join(model_folder, "mlp_model.h5"))
pickle.dump(scaler, open(os.path.join(model_folder, "scaler.pkl"), "wb"))

# Zapis historii
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(model_folder, "training_history.csv"), index=False)

# Informacja o modelu
model_info = (
    f"Epoki: {epochs}   "
    f"Batch: {batch_size}   "
    f"Warstwy: [128, 64]   "
    f"Wejście: cechy z XGBoost ({input_dim})"
)

# Wykresy
def plot_metric(metric_name, label, title, filename):
    epochs = range(1, len(history.history[metric_name]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history[metric_name], label=f"Train {label}", marker="o")
    plt.plot(epochs, history.history[f"val_{metric_name}"], label=f"Validation {label}", marker="o")
    plt.xlabel("Epoki")
    plt.ylabel(label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(list(epochs))
    plt.figtext(0.5, 0.05, model_info, ha="center", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(os.path.join(model_folder, filename))
    plt.close()

plot_metric("accuracy", "Accuracy", "Dokładność treningowa i walidacyjna", "accuracy_plot.png")
plot_metric("loss", "Loss", "Strata treningowa i walidacyjna", "loss_plot.png")
plot_metric("precision", "Precision", "Precyzja treningowa i walidacyjna", "precision_plot.png")
plot_metric("recall", "Recall", "Czułość treningowa i walidacyjna", "recall_plot.png")
plot_metric("f1_score", "F1-score", "F1-score treningowy i walidacyjny", "f1_score_plot.png")
plot_metric("auc", "AUC", "AUC treningowe i walidacyjne", "auc_plot.png")

print(f"\n Zakończono. Wszystko zapisane w: {model_folder}")
