# Tài liệu dữ liệu

Tài liệu này mô tả những gì có thể kiểm tra từ các tệp đang có trong dự án. Chưa có bằng chứng đầy đủ về nguồn gốc và quy trình xử lý: không suy ra quan hệ nguồn–đích chỉ từ tên tệp hoặc các cột tương tự.

## Đầu vào được chương trình sử dụng

- `processed/hcm_apartment_rentals.csv`: 3.194 dòng tin căn hộ, 21 cột; thời gian trong `post_time` từ 21/09/2026 đến 02/11/2026. Các URL trong tệp có host `www.nhatot.com`. Giá và đơn giá được ghi lần lượt theo triệu đồng/tháng và triệu đồng/m²; đây là ý nghĩa theo tên cột và code, chưa đối chiếu được nguồn gốc từng giá trị. `project_name` có thể thiếu. Cột vị trí, nội thất và tiện ích có trong tệp nhưng cách thu thập, chuẩn hóa và tính toán chưa có tài liệu.
- `processed/hcm_rent_timeseries.csv`: 72 dòng theo tháng từ 2019-01 đến 2024-12. Giá thuê được code diễn giải là triệu đồng/tháng; các cột còn lại biểu thị nhu cầu tìm kiếm, CPI, lãi suất và nguồn cung. Chưa có nguồn, phương pháp tổng hợp, lịch công bố hoặc bằng chứng về tính xác thực cho từng chuỗi; chưa thể xác nhận dữ liệu là quan sát thực tế hay được ước lượng/tạo lập.

Những số lượng và khoảng thời gian trên mô tả phiên bản dữ liệu hiện có và có thể thay đổi khi thay tệp.

## Các tệp CSV khác đang lưu trong `data/`

- `data_public.csv`: 51.304 dòng, gồm thông tin tin bất động sản, giá, diện tích và thời gian thu thập.
- `chung cu chotot.csv`: 2.015 dòng, gồm tiêu đề, diện tích, giá và khu vực.
- `hcmc-rent-index.csv`: 147 dòng chỉ số theo khu vực/loại tài sản; tệp có cột thời điểm snapshot. Snapshot năm 2026 không tự chứng minh nguồn của chuỗi tháng 2019–2024.

Chưa tìm thấy mã hoặc tài liệu chứng minh ba tệp này được dùng để tạo các tệp trong `processed/`; không sử dụng chúng để tái tạo dữ liệu theo giả định.

## Quy trình hiện có và phần cần bổ sung

1. Lệnh phân tích/mô hình đọc trực tiếp hai tệp `processed/`; không có bước thu thập hoặc làm sạch dữ liệu gốc trong repository hiện tại.
2. `python -m src.data_validation` kiểm tra cấu trúc và một số điều kiện cơ bản, không ghi hoặc sửa dữ liệu. Hai lệnh chạy chính cũng kiểm tra trước khi tạo kết quả.
3. Cần bổ sung nguồn gốc có thể kiểm chứng (đường dẫn/tên đơn vị, thời gian lấy và quyền sử dụng), dữ liệu gốc theo từng nguồn, định nghĩa biến và đơn vị, công thức tổng hợp theo tháng, quy tắc làm sạch/lọc/trùng và mã tái lập có phiên bản. Với chỉ số vĩ mô, cần cả ngày công bố và quy tắc dùng giá trị tại thời điểm dự báo.
4. Cần đối chiếu các nhãn nguồn trong biểu đồ/báo cáo với tài liệu nguồn trước khi trích dẫn; các nhãn hiện có không tự chứng minh nguồn dữ liệu.
