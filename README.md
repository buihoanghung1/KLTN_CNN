# Hệ Thống Phát Hiện và Phân Loại Tấn Công Mạng (CNN-IDS)

Dự án nghiên cứu và xây dựng Hệ thống phát hiện xâm nhập (IDS) sử dụng mô hình Mạng Nơ-ron Tích chập (CNN), tối ưu hóa bằng kiến trúc Inception-CNN 1D đa luồng để phân loại các luồng dữ liệu mạng bình thường và nhận diện các dạng tấn công phức tạp.

Đây là dự án Khóa luận Cử nhân ngành An toàn Thông tin - Trường Đại học Công Thương TP. HCM (HUIT).

## Hướng dẫn cài đặt và khới chạy

### Yêu cầu hệ thống
*   Python 3.8+
*   TensorFlow / Keras
*   Flask, Pandas, NumPy, Scikit-learn, Imbalanced-learn

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Khởi chạy Ứng dụng Web (Demo)
```bash
python app.py
```
Sau khi server chạy, truy cập vào trình duyệt tại địa chỉ: `http://127.0.0.1:5000`

### Hướng dẫn sử dụng
1.  Truy cập giao diện Web Demo IDS.
2.  Tải lên tệp dữ liệu mạng định dạng `.csv` (chuẩn UNSW-NB15).
3.  Nhấn nút **Phân Tích Ngay**.
4.  Hệ thống sẽ tiền xử lý dữ liệu và đưa ra **Báo Cáo Phân Tích Mạng** với thống kê chi tiết các lưu lượng bất thường và biểu đồ phân bổ các loại tấn công cùng độ tin cậy.

## So sánh hiệu năng

| Tiêu chí | Random Forest (ML truyền thống) | Inception-CNN 1D (Dự án) |
| :--- | :--- | :--- |
| **Độ chính xác (Accuracy)** | 74% | **87%** |
| **Recall (Shellcode)** | 50% | **83%** |
| **Recall (Worms)** | 12% | **85%** |
| **Báo động giả (False Alarm)**| Thấp | **Rất thấp (Gần 0%)** |

