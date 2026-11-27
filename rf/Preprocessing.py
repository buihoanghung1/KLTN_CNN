import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Tải dữ liệu từ file csv
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "..", "data", "raw", "Modified_UNSW-NB15_full.csv")
data = pd.read_csv(csv_path)

# Xóa khoảng trằng thừa
data.columns = data.columns.str.replace(' ', '', regex=True)

# Hiển thị tên các cột để kiểm tra
print("Các cột trong DataFrame:")
print(data.columns.tolist())
print("=" * 30)

# Xóa khoảng trắng thừa trong cột 'attack_cat'
data['attack_cat'] = data['attack_cat'].str.strip()
# Gộp nhãn 'Backdoor' và 'Backdoors' thành 'Backdoor'
data['attack_cat'] = data['attack_cat'].replace({'Backdoors': 'Backdoor', 'Backdoors': 'Backdoor', 'Backdoor': 'Backdoor'})

# In kiểu dữ liệu trước khi chuyển đổi
print("Kiểu dữ liệu trước khi chuyển đổi:")
print(data.dtypes)
print("=" * 30)

# Lưu trữ Encoder cho từng cột

encoders = {}
categorical_columns = ['proto', 'service', 'state', 'attack_cat']
label_column = ['Label']

# các cột còn lại 
numerical_columns = [col for col in data.columns if col not in categorical_columns + label_column]

# Mã hóa các cột phân loại
for col in categorical_columns:
    le = LabelEncoder()
    
    data[col] = data[col].astype(str)
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Xử lý cột nhãn
for col in label_column:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].replace([np.inf, -np.inf], 0)  # Thay thế vô hạn bằng 0
    data[col] = data[col].fillna(0)  # Điền giá trị NaN bằng 0
    data[col] = data[col].astype(int)  # Chuyển đổi sang int
    

# Các cột còn lại
for col in numerical_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# In kiểu dữ liệu sau khi chuyển đổi
print("Kiểu dữ liệu sau khi chuyển đổi:")
print(data.dtypes)
print("=" * 30)

# Kiểm tra và xử lý giá trị thiếu
missing_values = pd.DataFrame(data.isnull().sum())
print("Giá trị bị thiếu trong DataFrame:")
print("=" * 30)
print(missing_values)

# Điền giá trị bị thiếu
data = data.fillna(0)
print("Dữ liệu sau khi điền giá trị bị thiếu:")
print(data.isnull().sum())
print("=" * 30)

# Số lượng nhãn sau khi mã hóa
print("Số lượng nhãn sau khi mã hóa:")
print(data['Label'].value_counts())
print("=" * 30)

# Bản đồ mã hóa (Số -> Nhãn gốc) cho cột 'attack_cat'
print("Bản đồ mã hóa cho cột 'attack_cat':")
ls = encoders['attack_cat']
for i, label in enumerate(ls.classes_):
    print(f"{i:2} -> {label}")
print("=" * 30)

# Cân bằng dữ liệu

# In kích thước dữ liệu sau khi tiền xử lý
print("Kích thước dữ liệu sau khi tiền xử lý:")
print(data.shape)
print("=" * 30)

# Tách đặt trưng và nhãn
X = data.drop('attack_cat', axis=1)
y = data['attack_cat']

# Chia tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Lưu dữ liệu đã tiền xử lý
import pickle

# Lưu các tập dữ liệu dạng pickle
preprocessed_path = os.path.join(script_dir, "..", "..", "data", "preprocessed_data_rf.pkl")
with open(preprocessed_path, 'wb') as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'encoders': encoders,
        'feature_names': X.columns.tolist()
    }, f)

# Lưu encoders ra file riêng
import joblib
encoders_path = os.path.join(script_dir, "..", "..", "models", "artifacts", "feature_encoders_rf.pkl")
joblib.dump(encoders, encoders_path)
print(f"Đã lưu encoders vào '{encoders_path}'")

# Lưu feature_names và scaler (nếu có) ra file riêng
feature_names = [col for col in X.columns if col not in ['Label', 'attack_cat']]
scaler_path = os.path.join(script_dir, "..", "..", "models", "artifacts", "scaler_rf.pkl")
joblib.dump({
    'scaler': None,
    'top_features': feature_names
}, scaler_path)
print(f"Đã lưu feature_names vào '{scaler_path}'")

print(f"Đã lưu dữ liệu vào '{preprocessed_path}'")
print("=" * 30)
