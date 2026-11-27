import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for

script_dir = os.path.dirname(os.path.abspath(__file__))

# Khởi tạo Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(script_dir, "..", "..", "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



# --- CẤU HÌNH MODEL ---
MODEL_PATH = os.path.join(script_dir, "..", "..", "models", "saved", "Final_IDS_Model_rf.pkl")
SCALER_PATH = os.path.join(script_dir, "..", "..", "models", "artifacts", "scaler_rf.pkl")
ENCODERS_PATH = os.path.join(script_dir, "..", "..", "models", "artifacts", "feature_encoders_rf.pkl")

print("Đang sử dụng mô hình Random Forest để dự đoán!")
print("Đang load Model và Artifacts...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)
print("Load xong!")


def preprocess_data(df):
    """Tiền xử lý dữ liệu giống lúc train"""
    error_message = None
    try:
        # Xóa cột nhãn nếu có
        nhan_col = "attack_cat" if "attack_cat" in encoders else "Label"
        if nhan_col in df.columns:
            df = df.drop(columns=[nhan_col])
        # Luôn loại bỏ cột 'Label' nếu có (tránh dư cột so với scaler/model)
        if 'Label' in df.columns:
            df = df.drop(columns=['Label'])

        # Chuẩn hóa các cột phân loại
        for cot, le in encoders.items():
            if cot == nhan_col:
                continue
            if cot in df.columns:
                df[cot] = df[cot].astype(str).str.lower().str.strip()
                if type(le) is dict:
                    df[cot] = df[cot].map(le).fillna(0).astype(int)
                elif isinstance(le, LabelEncoder):
                    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    df[cot] = df[cot].map(mapping).fillna(0).astype(int)

        # Chuyển tất cả các cột còn lại sang kiểu số và điền giá trị thiếu bằng 0
        categorical_columns = list(encoders.keys())
        numeric_columns = [col for col in df.columns if col not in categorical_columns]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Tự động thêm các cột còn thiếu với giá trị 0
        scaler_features = scaler.get('top_features', df.columns.tolist()) if isinstance(scaler, dict) else getattr(scaler, 'feature_names_in_', df.columns.tolist())
        print(f"Cột scaler/model yêu cầu ({len(scaler_features)}): {scaler_features}")
        print(f"Cột thực tế ({len(df.columns)}): {df.columns.tolist()}")
        for col in scaler_features:
            if col not in df.columns:
                print(f"Thiếu cột: {col}, sẽ thêm với giá trị 0")
                df[col] = 0
        # Đảm bảo thứ tự cột đúng
        df = df[scaler_features]

        # Scale
        try:
            if isinstance(scaler, dict) and scaler.get('scaler') is not None:
                X_scaled = scaler['scaler'].transform(df)
            else:
                # Không dùng scaler, trả về DataFrame giữ tên cột
                X_scaled = df
        except Exception as e:
            error_message = f"Lỗi scaler: {e}"
            return None, error_message
        return X_scaled, None
    except Exception as e:
        error_message = f"Lỗi tiền xử lý: {e}"
        return None, error_message


@app.route('/', methods=['GET', 'POST'])
def index():
    model_info = "Đang sử dụng mô hình Random Forest để dự đoán!"
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.replace(' ', '', regex=True)
                df_display = df.copy()
                # Tiền xử lý
                X_rf, error_message = preprocess_data(df)
                if X_rf is None:
                    return render_template('index.html', model_info=model_info, error=f"Lỗi: {error_message}")
                # Dự đoán
                y_pred_indices = model.predict(X_rf)
                if hasattr(model, "predict_proba"):
                    y_pred_probs = model.predict_proba(X_rf)
                    confidence = np.max(y_pred_probs, axis=1)
                else:
                    confidence = [None] * len(y_pred_indices)
                # Giải mã nhãn
                nhan_col = "attack_cat" if "attack_cat" in encoders else "Label"
                le_target = encoders[nhan_col]
                if isinstance(le_target, LabelEncoder):
                    ket_qua_chu = le_target.inverse_transform(y_pred_indices)
                elif type(le_target) is dict:
                    reverse_map = {int(v): k for k, v in le_target.items()}
                    ket_qua_chu = [reverse_map.get(int(idx), 'Unknown') for idx in y_pred_indices]
                else:
                    ket_qua_chu = y_pred_indices
                # Thêm kết quả vào DataFrame
                df_display['Pred_Attack'] = ket_qua_chu
                df_display['Confidence'] = [f"{x*100:.2f}%" if x is not None else "N/A" for x in confidence]
                # Thống kê cho biểu đồ
                stats = df_display['Pred_Attack'].value_counts().to_dict()
                # Chỉ lấy 50 dòng đầu tiên để hiển thị 
                table_html = df_display.head(50).to_html(classes='table table-striped', index=False)
                return render_template('result.html', table=table_html, stats=stats, model_info=model_info)
            except Exception as e:
                return render_template('index.html', model_info=model_info, error=f"Đã có lỗi xảy ra: {str(e)}")
    return render_template('index.html', model_info=model_info)

if __name__ == '__main__':
    app.run(debug=True, port=5000)