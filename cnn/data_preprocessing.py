import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
# Doi MinMaxScaler -> StandardScaler

def xu_ly_du_lieu(csv_path):
    print("=== [Standard Mode] Xu ly du lieu (StandardScaler + ROS) ===")
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Loi doc file: {e}")
        return None, None, None, None, None, None, None

#---LOẠI BỎ DỮ LIỆU TRÙNG LẶP ---
    trc_khi_xoa = len(df)
    df = df.drop_duplicates()
    sau_khi_xoa = len(df)
    if trc_khi_xoa > sau_khi_xoa:
        print(f"Canh bao: Da xoa {trc_khi_xoa - sau_khi_xoa} dong trung lap (Tranh Data Leakage)!")
    # ---------------------------------------------------

    nhan = "attack_cat" if "attack_cat" in df.columns else "Label"
    df[nhan] = df[nhan].astype(str).str.lower().str.strip()
    map_nhan = {
        'backdoors': 'backdoor', 'shellcode ': 'shellcode', 
        'fuzzers ': 'fuzzers', 'reconnaissance ': 'reconnaissance'
    }
    df[nhan] = df[nhan].replace(map_nhan)
    df = df.fillna(0)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    ma_hoa = {}
    for cot in cat_cols:
        if cot == nhan: continue
        le = LabelEncoder()
        df[cot] = df[cot].astype(str).str.lower().str.strip()
        df[cot] = le.fit_transform(df[cot])
        ma_hoa[cot] = le

    le_target = LabelEncoder()
    df[nhan] = le_target.fit_transform(df[nhan])
    ma_hoa[nhan] = le_target

    X = df.drop(nhan, axis=1)
    y = df[nhan]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Dung StandardScaler de giu nguyen phan phoi du lieu 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Dang chay RandomOverSampler (Nhan ban du lieu)...")
    # Nhan ban du lieu -> Cac lop nho se co so luong bang lop lon nhat
    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_scaled, y_train)
    
    print(f"Kich thuoc Train sau can bang: {X_train_bal.shape}")

    X_train_cnn = np.expand_dims(X_train_bal, axis=2)
    X_test_cnn = np.expand_dims(X_test_scaled, axis=2)
    so_lop = len(np.unique(y))

    return X_train_cnn, X_test_cnn, y_train_bal, y_test, so_lop, scaler, ma_hoa