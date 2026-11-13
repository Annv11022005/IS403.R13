<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>PHÂN TÍCH DỮ LIỆU KINH DOANH - IS403</b></h1>
<h3 align="center"><b>ĐỀ TÀI: PHÂN TÍCH VÀ DỰ BÁO GIÁ THUÊ CĂN HỘ TẠI THÀNH PHỐ HỒ CHÍ MINH</b></h3>

## GIỚI THIỆU MÔN HỌC

- **Tên môn học:** Phân tích dữ liệu kinh doanh
- **Mã môn học:** IS403
- **Mã lớp:** IS403.R13
- **Năm học:** HK1 (2026 - 2027)
- **Giảng viên:** TS. Trần Văn Hải Triều

## THÀNH VIÊN NHÓM 7

| STT | MSSV     | Họ và Tên             | GitHub                          | Email                   |
| :-- | :------- | :-------------------- | :------------------------------ | :---------------------- |
| 1   | 24521696 | Lê Nguyễn Phương Uyên |                                 | 24521696@gm.uit.edu.vn  |
| 2   | 24521979 | Lê Nguyễn Thanh Vân   |                                 | 24521979@gm.uit.edu.vn  |
| 3   | 23520022 | Nguyễn Văn Ân         | https://github.com/Annv11022005 | 23520022@gm.uit.edu.vn  |
| 4   | 24520304 | Đào Trọng Định        |                                 | 24520304@gm.uit.edu.vn  |

## TỔNG QUAN ĐỀ TÀI

Đồ án tập trung phân tích các yếu tố ảnh hưởng đến giá thuê căn hộ tại TP.HCM (vị trí địa lý, diện tích, cấu trúc phòng, nội thất, tiện ích nội khu và các yếu tố kinh tế vĩ mô), đồng thời xây dựng và đánh giá các mô hình Học máy (Machine Learning) cùng Chuỗi thời gian (Time Series) nhằm phục vụ bài toán định giá và dự báo xu hướng thị trường.

## CẤU TRÚC THƯ MỤC

```text
IS403/
├── data/processed/        # Dữ liệu căn hộ và chuỗi thời gian đã xử lý
├── src/
│   ├── config.py          # Cấu hình chung của dự án
│   ├── analysis/          # Phân tích thống kê mô tả, trực quan hóa và kiểm định giả thuyết
│   └── model/             # Xây dựng và đánh giá các mô hình dự báo (ML & Time Series)
├── outputs/               # Bảng số liệu kết quả (.csv), biểu đồ (.png) và báo cáo tổng hợp (.txt)
├── requirements.txt       # Các thư viện Python cần thiết
└── README.md              # Thông tin tổng quan đồ án
```

## HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH

1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy phân tích dữ liệu và kiểm định thống kê:**
   ```bash
   python -m src.analysis.run_analysis
   ```

3. **Chạy huấn luyện và đánh giá các mô hình dự báo:**
   ```bash
   python -m src.model.run_models
   ```
