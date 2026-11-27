import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Đọc dữ liệu đã tiền xử lý
script_dir = os.path.dirname(os.path.abspath(__file__))
preprocessed_path = os.path.join(script_dir, "..", "..", "data", "preprocessed_data_rf.pkl")
with open(preprocessed_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']

# Loại bỏ cột nhãn nếu còn sót lại trong X_train
for col in ['Label', 'attack_cat']:
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])

print("Cột dùng để train model:", X_train.columns.tolist())
print("Số lượng cột:", len(X_train.columns))

# Train lại mô hình
rf_model = RandomForestClassifier(
    n_estimators=1,  # Số lượng cây nhỏ để so sánh với mô hình chính
    random_state=123,
    max_features=1,
    max_depth=10,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Vẽ ma trận nhầm lẫn (confusion matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# Load encoder để lấy tên nhãn gốc
encoders_path = os.path.join(script_dir, "..", "..", "models", "artifacts", "feature_encoders_rf.pkl")
encoders = joblib.load(encoders_path)
attack_cat_encoder = encoders['attack_cat']
class_names = attack_cat_encoder.classes_

# Tính toán ma trận nhầm lẫn
cm = confusion_matrix(y_train, rf_model.predict(X_train))
# Tính tỉ lệ phần trăm trên từng hàng
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True, linewidths=0.5)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.tight_layout()
cm_path = os.path.join(script_dir, "..", "..", "outputs", "plots", "confusion_matrix_rf.png")
plt.savefig(cm_path)
plt.close()

# Lưu lại mô hình
model_path = os.path.join(script_dir, "..", "..", "models", "saved", "Final_IDS_Model_rf.pkl")
joblib.dump(rf_model, model_path)

# Tính toán các chỉ số đánh giá mô hình
pred_train = rf_model.predict(X_train)
accuracy = accuracy_score(y_train, pred_train)
precision = precision_score(y_train, pred_train, average='weighted', zero_division=0)
recall = recall_score(y_train, pred_train, average='weighted', zero_division=0)
f1 = f1_score(y_train, pred_train, average='weighted', zero_division=0)


# In ra bảng số liệu đánh giá mô hình
report_text = []
report_text.append("Bảng số liệu đánh giá mô hình Random Forest:")
report_text.append(f"Accuracy:  {accuracy:.4f}")
report_text.append(f"Precision: {precision:.4f}")
report_text.append(f"Recall:    {recall:.4f}")
report_text.append(f"F1-Score:  {f1:.4f}")
report_text.append("")
report_text.append("Classification Report:")
class_report = classification_report(y_train, pred_train, zero_division=0)
report_text.append(class_report)

print("\n" + "\n".join(report_text))

# Lưu bảng số liệu vào file
report_path = os.path.join(script_dir, "..", "..", "outputs", "results", "rf_evaluation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_text))

print(f"Đã lưu lại mô hình RandomForest tại: {model_path}")
print(f"Đã lưu báo cáo đánh giá tại: {report_path}")