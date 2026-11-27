import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model

script_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(script_dir, "..", "..", "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- CAU HINH ---
MODEL_PATH = os.path.join(script_dir, "..", "..", "models", "saved", "Final_IDS_Model.h5")
SCALER_PATH = os.path.join(script_dir, "..", "..", "models", "artifacts", "scaler.pkl")
ENCODERS_PATH = os.path.join(script_dir, "..", "..", "models", "artifacts", "feature_encoders.pkl")

print("=== DANG KHOI DONG SERVER ===")
try:
    # Load model o che do compile=False 
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print(">>> OK: Model & Artifacts da san sang!")
except Exception as e:
    print(f"!!! LOI: Khong the load model. Chi tiet: {e}")

def preprocess_data(df):
    """Chuan hoa du lieu dau vao y het luc train"""
    # 1. Xoa cot Label (neu file upload co label)
    nhan_col = "attack_cat" if "attack_cat" in encoders else "Label"
    if nhan_col in df.columns:
        df = df.drop(columns=[nhan_col])

    # 2. Xu ly cot Categorical (Map string -> int)
    for cot, le in encoders.items():
        if cot == nhan_col: continue
        if cot in df.columns:
            df[cot] = df[cot].astype(str).str.lower().str.strip()
            # Map an toan: Gia tri la -> 0
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            df[cot] = df[cot].map(mapping).fillna(0).astype(int)
    
    # 3. Scale du lieu 
    try:
        X_scaled = scaler.transform(df)
    except ValueError as e:
        print(f"Loi Scale: {e}")
        return None

    # 4. Reshape cho CNN (them chieu thu 3)
    X_cnn = np.expand_dims(X_scaled, axis=2)
    return X_cnn

@app.route('/', methods=['GET', 'POST'])
def index():
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
                df_display = df.copy() # Ban sao de hien thi
                
                # Tien xu ly
                X_cnn = preprocess_data(df)
                if X_cnn is None:
                    return "<h3>Lỗi Dữ Liệu:</h3><p>File CSV thiếu cột hoặc sai định dạng so với lúc train.</p>"

                # --- DU DOAN ---
                # verbose=0 de khong in log rac ra terminal
                y_pred_probs = model.predict(X_cnn, verbose=0)
                y_pred_indices = np.argmax(y_pred_probs, axis=1)
                
                # Giai ma (So 0,1,2 -> Ten tan cong)
                nhan_col = "attack_cat" if "attack_cat" in encoders else "Label"
                le_target = encoders[nhan_col]
                ket_qua_chu = le_target.inverse_transform(y_pred_indices)
                
                # --- GAN KET QUA VAO BANG ---
                df_display['Pred_Attack'] = ket_qua_chu
                df_display['Confidence_raw'] = np.max(y_pred_probs, axis=1)
                df_display['Confidence'] = df_display['Confidence_raw'].apply(lambda x: f"{x*100:.2f}%")

                # --- TAO THONG KE (Cho bieu do) ---
                stats = df_display['Pred_Attack'].value_counts().to_dict()

                # Tao bang HTML (Lay 200 dong dau)
                table_html = df_display.head(200).to_html(
                    classes='table table-bordered table-striped table-hover align-middle',
                    index=False,
                    justify='center'
                )

                return render_template('result.html', table=table_html, stats=stats)

            except Exception as e:
                return f"<h3>Lỗi Hệ Thống:</h3><pre>{str(e)}</pre>"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)