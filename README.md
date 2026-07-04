# Softmax Regression

Thư mục này chứa code xây dựng mô hình Softmax Regression bằng Python thuần, không cần cài thêm thư viện ngoài. Chương trình đọc dữ liệu từ file CSV trong thư mục `data`.

## Cách chạy

```bash
cd AI-exam-code
python3 softmax_regression.py
```

## Dataset

- File dữ liệu: `data/softmax_demo.csv`
- Số mẫu: 60
- Số lớp: 3 lớp (`Loai_A`, `Loai_B`, `Loai_C`)
- Ý nghĩa: bộ dữ liệu 2 chiều dùng để minh họa bài toán phân loại nhiều lớp. Mỗi lớp có các điểm dữ liệu tập trung quanh một vùng khác nhau, phù hợp để demo Softmax Regression.

## Cột dữ liệu

- `x1`: đặc trưng thứ nhất của mẫu.
- `x2`: đặc trưng thứ hai của mẫu.
- `label`: nhãn lớp cần dự đoán.

## Nội dung chính

- Tính điểm tuyến tính cho từng lớp.
- Chuyển điểm thành xác suất bằng hàm Softmax.
- Tính loss bằng Cross-Entropy.
- Tính gradient theo trọng số và bias.
- Cập nhật tham số bằng Gradient Descent.
- Dự đoán nhãn và tính độ chính xác.

## File

- `softmax_regression.py`: code chính của mô hình, đọc CSV, huấn luyện và đánh giá.
- `data/softmax_demo.csv`: dữ liệu dùng để chạy thử mô hình.
