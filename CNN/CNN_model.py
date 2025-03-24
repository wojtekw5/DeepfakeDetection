import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import shuffle

img_size = (80, 264)
batch_size = 32
epochs = 15

# tworzenie folderu do zapisu
def get_unique_folder_name(base_path):
    if not os.path.exists(base_path):
        return base_path
    counter = 1
    new_path = f"{base_path}_{counter}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}"
    return new_path

# ścieżki do danych
desktop_path = r"C:\Users\wojtek\Desktop"
model_folder = os.path.join(desktop_path, "cnn_model_npy")
model_folder = get_unique_folder_name(model_folder)
os.makedirs(model_folder, exist_ok=True)

print(f"⚪ Wyniki będą zapisane w: {model_folder}")

train_real_dir = os.path.join(desktop_path, "data_cnn", "train", "real")
train_fake_dir = os.path.join(desktop_path, "data_cnn", "train", "fake")

# wczytywanie melspektrogramów
def load_npy_data(folder, label):
    X, y = [], []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} nie istnieje!")

    files = []
    for f in os.listdir(folder):
        if f.endswith(".npy"):
            files.append(f)

    print(f"⚪ Znaleziono {len(files)} plików w {folder}")

    for file in files:
        file_path = os.path.join(folder, file)
        try:
            mel_spectrogram = np.load(file_path)
            if mel_spectrogram.shape != (80, 264, 1):
                continue  # Pominięcie melspektrogramów o złych wymiarach
            X.append(mel_spectrogram)
            y.append(label)
        except Exception as e:
            print(f"🔴 Błąd przy wczytywaniu {file}: {e}")

    return np.array(X), np.array(y)

X_real, y_real = load_npy_data(train_real_dir, 0)
X_fake, y_fake = load_npy_data(train_fake_dir, 1)

# połączenie danych
X = np.concatenate((X_real, X_fake), axis=0)
y = np.concatenate((y_real, y_fake), axis=0)

# wymieszanie danych
X, y = shuffle(X, y, random_state=42)

# normalizacja do zakresu [0,1]
X = (X - np.mean(X)) / np.std(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# obliczenie `class_weight`
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.002), input_shape=(80, 264, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.002)),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.002)),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

# kompilacja modelu
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), F1Score()]
)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# trenowanie modelu
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
                    class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr])

# zapis modelu
model_path = os.path.join(model_folder, "cnn_fake_detector_npy.h5")
model.save(model_path)
print(f"⚪ Model zapisany jako: {model_path}")


# zapis historii trenowania do CSV
history_df = pd.DataFrame(history.history)
history_csv_path = os.path.join(model_folder, "training_history.csv")
history_df.to_csv(history_csv_path, index=False)

kernel_info = "3x3"
leyers_info = "3/1"
filters_info = "[32,64,128]"

model_info = (
    f"Epoki: {epochs}   "
    f"Batch: {batch_size}   "
    f"Kernel: {kernel_info}   "
    f"Sieci (cnn/dense): {leyers_info}   "
    f"Filtry: {filters_info}   "
)

# Accuracy
def plot_accuracy(history, folder):
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['accuracy'], label="Train Accuracy", marker="o")
    plt.plot(epochs, history.history['val_accuracy'], label="Validation Accuracy", marker="o")
    plt.xlabel("Epoki")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.grid()
    plt.title("Dokładność treningowa i walidacyjna")
    plt.xticks(list(epochs))
    plt.subplots_adjust(bottom=0.2)  # Zwiększenie miejsca na dole
    plt.figtext(0.5, 0.05, model_info, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    plt.savefig(os.path.join(folder, "accuracy_plot.png"))
    plt.close()

# Loss
def plot_loss(history, folder):
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['loss'], label="Train Loss", marker="o")
    plt.plot(epochs, history.history['val_loss'], label="Validation Loss", marker="o")
    plt.xlabel("Epoki")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid()
    plt.title("Strata treningowa i walidacyjna")
    plt.xticks(list(epochs))
    plt.subplots_adjust(bottom=0.2)  # Zwiększenie miejsca na dole
    plt.figtext(0.5, 0.05, model_info, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    plt.savefig(os.path.join(folder, "loss_plot.png"))
    plt.close()

# Precision
def plot_precision(history, folder):
    epochs = range(1, len(history.history['precision']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['precision'], label="Train Precision", marker="o")
    plt.plot(epochs, history.history['val_precision'], label="Validation Precision", marker="o")
    plt.xlabel("Epoki")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.title("Precyzja (Precision) treningowa i walidacyjna")
    plt.xticks(list(epochs))
    plt.subplots_adjust(bottom=0.2)  # Zwiększenie miejsca na dole
    plt.figtext(0.5, 0.05, model_info, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    plt.savefig(os.path.join(folder, "precision_plot.png"))
    plt.close()

# Recall
def plot_recall(history, folder):
    epochs = range(1, len(history.history['recall']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['recall'], label="Train Recall", marker="o")
    plt.plot(epochs, history.history['val_recall'], label="Validation Recall", marker="o")
    plt.xlabel("Epoki")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid()
    plt.title("Czułość (Recall) treningowa i walidacyjna")
    plt.xticks(list(epochs))
    plt.subplots_adjust(bottom=0.2)  # Zwiększenie miejsca na dole
    plt.figtext(0.5, 0.05, model_info, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    plt.savefig(os.path.join(folder, "recall_plot.png"))
    plt.close()

# F1-score
def plot_f1(history, folder):
    epochs = range(1, len(history.history['f1_score']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['f1_score'], label="Train F1-score", marker="o")
    plt.plot(epochs, history.history['val_f1_score'], label="Validation F1-score", marker="o")
    plt.xlabel("Epoki")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid()
    plt.title("F1-score treningowy i walidacyjny")
    plt.xticks(list(epochs))
    plt.subplots_adjust(bottom=0.2)  # Zwiększenie miejsca na dole
    plt.figtext(0.5, 0.05, model_info, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    plt.savefig(os.path.join(folder, "f1_score_plot.png"))
    plt.close()

# AUC
def plot_auc(history, folder):
    epochs = range(1, len(history.history['auc']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['auc'], label="Train AUC", marker="o")
    plt.plot(epochs, history.history['val_auc'], label="Validation AUC", marker="o")
    plt.xlabel("Epoki")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid()
    plt.title("AUC treningowe i walidacyjne")
    plt.xticks(list(epochs))
    plt.subplots_adjust(bottom=0.2)  # Zwiększenie miejsca na dole
    plt.figtext(0.5, 0.05, model_info, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    plt.savefig(os.path.join(folder, "auc_plot.png"))
    plt.close()

# generowanie wszystkich wykresów
plot_accuracy(history, model_folder)
plot_loss(history, model_folder)
plot_precision(history, model_folder)
plot_recall(history, model_folder)
plot_f1(history, model_folder)
plot_auc(history, model_folder)

print(f"⚪ Zakończono")
