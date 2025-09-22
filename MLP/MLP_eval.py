import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import csv
import pickle

desktop_path = r"C:\Users\wojtek\Desktop"
model_folder_name = "mlp_model_npy"
model_folder = os.path.join(desktop_path, model_folder_name)
model_path = os.path.join(model_folder, "mlp_model.h5")
real_features_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature_EVAL", "REAL", "features.npy")
fake_features_path = os.path.join(desktop_path, "XGBOOST_DATASET", "Feature_EVAL", "FAKE", "features.npy")

validation_folder = os.path.join(model_folder, "validation")
os.makedirs(validation_folder, exist_ok=True)

scaler_path = os.path.join(model_folder, "scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler nie istnieje: {scaler_path}")

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

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

X_real = np.load(real_features_path)
X_fake = np.load(fake_features_path)
y_real = np.zeros(len(X_real))
y_fake = np.ones(len(X_fake))

X_val = np.concatenate((X_real, X_fake), axis=0)
y_val = np.concatenate((y_real, y_fake), axis=0)

filenames = []
for i in range(len(X_real)):
    filenames.append(f"real_{i}.npy")

for i in range(len(X_fake)):
    filenames.append(f"fake_{i}.npy")

X_val = scaler.transform(X_val)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Plik modelu {model_path} nie istnieje!")

model = tf.keras.models.load_model(model_path, custom_objects={"F1Score": F1Score})
print(f"Wczytano model: {model_path}")

y_pred_probs = model.predict(X_val).ravel()
y_pred = (y_pred_probs > 0.5).astype("int32")

def plot_confusion_matrix(y_val, y_pred, folder_path):
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Macierz Pomyłek")
    plt.savefig(os.path.join(folder_path, "conf_matrix.png"))
    plt.show()

def plot_roc_curve(y_val, y_pred_probs, folder_path):
    fpr, tpr, _ = roc_curve(y_val, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, label="Losowy klasyfikator")
    plt.xlabel("Wskaźnik fałszywie pozytywnych")
    plt.ylabel("Wskaźnik prawdziwie pozytywnych (czułość)")
    plt.title("Krzywa ROC - MLP ")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(folder_path, "roc_curve.png"))
    plt.show()

def save_predictions_to_csv(filenames, y_val, y_pred, y_pred_probs, folder_path):
    output_csv_file = os.path.join(folder_path, "predictions.csv")
    headers = ['Filename', 'Real_Probability', 'Fake_Probability', 'Predicted_Label', 'Actual_Label', 'Classification', 'Accuracy_Status']

    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for i in range(len(filenames)):
            real_prob = 1 - y_pred_probs[i]
            fake_prob = y_pred_probs[i]

            if y_pred[i] == 1:
                predicted_class = "Fake"
            else:
                predicted_class = "Real"

            if y_pred[i] != y_val[i]:
                status = "INACCURATELY"
            else:
                status = ""

            writer.writerow([filenames[i], real_prob, fake_prob, y_pred[i], y_val[i], predicted_class, status])

    print(f"CSV z predykcjami zapisany w: {output_csv_file}")

plot_confusion_matrix(y_val, y_pred, validation_folder)
plot_roc_curve(y_val, y_pred_probs, validation_folder)
save_predictions_to_csv(filenames, y_val, y_pred, y_pred_probs, validation_folder)

print(f"\n Wszystko zapisane w folderze: {validation_folder}")