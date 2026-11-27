import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import load_model

from data_preprocessing import xu_ly_du_lieu
from cnn_model import tao_mo_hinh_cnn

# CPU Config
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

script_dir = os.path.dirname(os.path.abspath(__file__))
duong_dan_csv = os.path.join(script_dir, "..", "..", "data", "raw", "Modified_UNSW-NB15_full.csv")
thu_muc_checkpoint = os.path.join(script_dir, "..", "..", "models", "checkpoints")
thu_muc_artifacts = os.path.join(script_dir, "..", "..", "models", "artifacts")
os.makedirs(thu_muc_checkpoint, exist_ok=True)
os.makedirs(thu_muc_artifacts, exist_ok=True)

# luu qua trinh train
duong_dan_checkpoint = os.path.join(thu_muc_checkpoint, "inception_balanced_best.h5")
duong_dan_model = os.path.join(script_dir, "..", "..", "models", "saved", "Final_IDS_Model.h5")
duong_dan_ketqua = os.path.join(script_dir, "..", "..", "outputs", "results", "training_results.csv")
duong_dan_scaler = os.path.join(thu_muc_artifacts, "scaler.pkl")
duong_dan_encoders = os.path.join(thu_muc_artifacts, "feature_encoders.pkl")
duong_dan_log = os.path.join(script_dir, "..", "..", "outputs", "logs", "training_log.csv")

# --- HAM VE BIEU DO ---
def ve_bieu_do_history(log_path):
    if not os.path.exists(log_path):
        print(f"Khong tim thay file log: {log_path}")
        return

    data = pd.read_csv(log_path)
    epochs = range(1, len(data) + 1)

    plt.figure(figsize=(14, 6))

    # Bieu do Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, data['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, data['val_accuracy'], 'r--', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Bieu do Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, data['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, data['val_loss'], 'r--', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(script_dir, "..", "..", "outputs", "plots", "training_history_plot.png")
    plt.savefig(plot_path)
    print(f"Da luu bieu do training tai: {plot_path}")
    # plt.show() # Bo comment neu chay tren may ca nhan co man hinh

print("=== Bat dau huan luyen (CHIEN THUAT CAN BANG - STANDARD SCALER) ===")

X_train, X_test, y_train, y_test, so_lop, scaler, ma_hoa = xu_ly_du_lieu(duong_dan_csv)
if X_train is None: exit()
input_shape = X_train.shape[1:]

joblib.dump(scaler, duong_dan_scaler)
joblib.dump(ma_hoa, duong_dan_encoders)

label_col = "attack_cat" if "attack_cat" in ma_hoa else "Label"
encoder_nhan = ma_hoa[label_col]
class_names = encoder_nhan.classes_

# --- DIEU CHINH TRONG SO MEM MAI HON ---
print("Dang thiet lap trong so...")

class_weight_dict = {i: 1.0 for i in range(len(class_names))}

for i, name in enumerate(class_names):
    # x3
    if name in ['analysis', 'backdoor', 'shellcode', 'worms']:
        class_weight_dict[i] = 3.0 
        print(f" > Lop '{name}': Weight x3.0 (Giam de bot bao dong gia)")
    
    # x2
    elif name in ['dos', 'fuzzers', 'reconnaissance']:
        class_weight_dict[i] = 2.0  
        print(f" > Lop '{name}': Weight x2.0")
    
    # 0.8
    elif name in ['exploits', 'generic']:
        class_weight_dict[i] = 0.8  
        print(f" > Lop '{name}': Weight x0.8 (Can bang lai)")
    else:
        print(f" > Lop '{name}': Weight x1.0")

# Tao model
model = tao_mo_hinh_cnn(input_shape, so_lop)

callbacks = [
    ModelCheckpoint(duong_dan_checkpoint, monitor="val_accuracy", save_best_only=True, verbose=1, mode='max'),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    CSVLogger(duong_dan_log)
]

BATCH_SIZE = 128 
EPOCHS = 30  

print(f"Training voi Batch Size {BATCH_SIZE}...")
model.fit(
    X_train, y_train,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    class_weight=class_weight_dict, 
    verbose=1
)

model.save(duong_dan_model)

print("=== Danh gia ===")
model = load_model(duong_dan_checkpoint)

y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
# màu confusion Matrix
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Balanced Strategy)")
plt.tight_layout()
cm_path = os.path.join(script_dir, "..", "..", "outputs", "plots", "confusion_matrix.png")
plt.savefig(cm_path)

print(classification_report(y_test, y_pred, target_names=class_names))
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv(duong_dan_ketqua)

# --- VE BIEU DO HISTORY ---
ve_bieu_do_history(duong_dan_log)

print("Hoan tat.")