import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
from tensorflow.keras.metrics import Precision
import csv

desktop_path = r"C:\Users\wojtek\Desktop"
model_folder_name = "cnn_model_npy_3"
model_folder = os.path.join(desktop_path, model_folder_name)
model_path = os.path.join(desktop_path, model_folder_name, "cnn_fake_detector_npy.h5")
val_real_dir = os.path.join(desktop_path, "data_cnn", "val", "real")
val_fake_dir = os.path.join(desktop_path, "data_cnn", "val", "fake")

validation_folder = os.path.join(model_folder, "validation")
os.makedirs(validation_folder, exist_ok=True)

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

def load_npy_data(folder, label):
    X, y, filenames = [], [], []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} nie istnieje!")
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    print(f"⚪ Znaleziono {len(files)} plików w {folder}")
    for file in files:
        file_path = os.path.join(folder, file)
        try:
            mel_spectrogram = np.load(file_path)
            if mel_spectrogram.shape != (80, 264, 1):
                continue
            X.append(mel_spectrogram)
            y.append(label)
            filenames.append(file)
        except Exception as e:
            print(f"Błąd przy wczytywaniu {file_path}: {e}")
    return np.array(X), np.array(y), filenames

X_real, y_real, filenames_real = load_npy_data(val_real_dir, 0)
X_fake, y_fake, filenames_fake = load_npy_data(val_fake_dir, 1)

X_val = np.concatenate((X_real, X_fake), axis=0)
y_val = np.concatenate((y_real, y_fake), axis=0)
filenames = filenames_real + filenames_fake

# normalizacja
X_val = (X_val - np.mean(X_val)) / np.std(X_val)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Plik modelu {model_path} nie istnieje!")

try:
    model = tf.keras.models.load_model(model_path, custom_objects={"F1Score": F1Score})
    print(f"⚪ Poprawnie wczytano model: {model_path}")
except Exception as e:
    print(f"Błąd przy wczytywaniu modelu: {e}")
    raise

# predykcje
y_pred_probs = model.predict(X_val).ravel()   # << spłaszczone do 1D
y_pred = (y_pred_probs > 0.5).astype("int32")


def plot_confusion_matrix(y_true, y_hat, out_dir):
    cm = confusion_matrix(y_true, y_hat)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Macierz konfuzji")
    plt.savefig(os.path.join(out_dir, "conf_matrix.png"))
    plt.show()

def plot_roc_curve(y_true, y_scores, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"Krzywa ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2, label="Losowy klasyfikator")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Krzywa ROC - Detekcja FAKE")
    plt.legend(loc="lower right"); plt.grid()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.show()

def plot_precision_recall_curve_fig(y_true, y_scores, out_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color="blue", lw=2,
             label=f"Precision–Recall (AP = {ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Krzywa Precision–Recall - Detekcja FAKE")
    plt.legend(loc="lower left"); plt.grid()
    plt.savefig(os.path.join(out_dir, "precision_recall_curve.png"))
    plt.show()


plot_confusion_matrix(y_val, y_pred, validation_folder)
plot_roc_curve(y_val, y_pred_probs, validation_folder)
plot_precision_recall_curve_fig(y_val, y_pred_probs, validation_folder)


def save_predictions_to_csv(filenames, y_true, y_hat, y_scores, out_dir):
    output_csv_file = os.path.join(out_dir, "predictions.csv")
    headers = ['Filename', 'Real_Probability', 'Fake_Probability',
               'Predicted_Label', 'Actual_Label', 'Classification', 'Accuracy_Status']
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)a
        writer.writerow(headers)
        for i in range(len(filenames)):
            real_prob = float(1 - y_scores[i])
            fake_prob = float(y_scores[i])
            predicted_class = "Fake" if y_hat[i] == 1 else "Real"
            status = "INACCURATELY" if y_hat[i] != y_true[i] else ""
            writer.writerow([filenames[i], real_prob, fake_prob,
                             int(y_hat[i]), int(y_true[i]), predicted_class, status])
    print(f"Predykcje zapisane do {output_csv_file}")

save_predictions_to_csv(filenames, y_val, y_pred, y_pred_probs, validation_folder)

print(f"\nWszystko zapisane w folderze: {validation_folder}")
