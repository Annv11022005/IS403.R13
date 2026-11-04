# -*- coding: utf-8 -*-
"""
Entry-point chạy trọn bộ Giai đoạn 1 cho Dataset 1:
  - Bước 1.1: Thống kê mô tả đa tầng (src/dataset1/descriptive_stats.py)
  - Bước 1.2: Trực quan hóa 5 biểu đồ cốt lõi 300 DPI (src/dataset1/visualization.py)
  - Bước 1.3: Kiểm định giả thuyết thống kê 3 lớp (src/dataset1/hypothesis_testing.py)
  - Bước 1.4: Tổng hợp kết luận RQ1–RQ6 & định hướng kỹ thuật cho Giai đoạn 2 (ML Pipeline)

Cách chạy từ thư mục gốc d:\Code\IS403:
    python -m src.dataset1.run_phase1
"""

from pathlib import Path
import sys

# Đảm bảo import được package `src` dù chạy bằng `python -m` hay `python src/...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    DATASET1_CSV,
    PHASE1_FIGURES_DIR,
    PHASE1_REPORT_TXT,
    PHASE1_TABLES_DIR,
    ensure_phase1_dirs,
    setup_matplotlib_style,
)
from src.dataset1.descriptive_stats import run_descriptive_pipeline
from src.dataset1.hypothesis_testing import run_hypothesis_pipeline
from src.dataset1.visualization import run_visualization_pipeline


def build_phase1_report(
    df: pd.DataFrame,
    desc_res: dict[str, pd.DataFrame],
    pearson_df: pd.DataFrame,
    hyp_res: dict[str, pd.DataFrame],
) -> str:
    """Tạo báo cáo tổng hợp Giai đoạn 1 (Bước 1.1 -> 1.4) chuẩn bị cho báo cáo Word & Giai đoạn 2."""
    num_df = desc_res["numeric"].set_index("Biến")
    skew_raw = num_df.loc["price_million_vnd", "Skewness"]
    skew_log = num_df.loc["log_price_million_vnd", "Skewness"]

    # Tìm các cặp biến độc lập có tương quan nội tại lớn (|r| >= 0.65) cảnh báo đa cộng tuyến
    feat_cols = [
        "area_m2",
        "bedrooms",
        "bathrooms",
        "furniture_score",
        "distance_to_center_km",
        "has_pool",
        "has_gym",
        "is_commercial_project",
    ]
    high_corr_pairs = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            c1, c2 = feat_cols[i], feat_cols[j]
            r_ij = float(pearson_df.loc[c1, c2])
            if abs(r_ij) >= 0.60:
                high_corr_pairs.append(f"{c1} <-> {c2} (r = {r_ij:+.3f})")

    lines = [
        "=" * 86,
        "BÁO CÁO KẾT QUẢ GIAI ĐOẠN 1 (DATASET 1): EDA & KIỂM ĐỊNH THỐNG KÊ (RQ1 -> RQ6)",
        "=" * 86,
        f"Nguồn dữ liệu : {DATASET1_CSV.relative_to(PROJECT_ROOT)}",
        f"Quy mô mẫu    : {len(df):,} căn hộ cho thuê hợp lệ",
        "",
        "--------------------------------------------------------------------------------------",
        "1. [MỤC C.1] THỐNG KÊ MÔ TẢ THEO 3 QUẬN CHÍNH",
        "--------------------------------------------------------------------------------------",
        desc_res["district"][
            ["district_std", "Count", "Mean", "Median", "Std", "Skewness", "Kurtosis", "Mean_Đơn_giá_m2", "Tỷ_lệ_Chung_cư_TM_%"]
        ].to_string(index=False),
        "",
        "--------------------------------------------------------------------------------------",
        "2. [MỤC C.1 MỞ RỘNG] THỐNG KÊ THEO 5 TIỂU KHU VỰC (TÁCH QUẬN 2, QUẬN 9, THỦ ĐỨC CŨ)",
        "--------------------------------------------------------------------------------------",
        desc_res["subdistrict"][
            ["sub_district", "Count", "Mean", "Median", "Std", "Skewness", "Kurtosis", "Mean_Đơn_giá_m2", "Tỷ_lệ_Chung_cư_TM_%"]
        ].to_string(index=False),
        "",
        "--------------------------------------------------------------------------------------",
        "3. [NGHỊCH LÝ SIMPSON] THỐNG KÊ CHÉO TIỂU KHU VỰC x PHÂN KHÚC CĂN HỘ",
        "--------------------------------------------------------------------------------------",
        desc_res["crosstab"].to_string(index=False),
        "",
        "--------------------------------------------------------------------------------------",
        "4. [MỤC C.5 - LỚP 1] KIỂM TRA GIẢ ĐỊNH PHÂN PHỐI & ĐỒNG NHẤT PHƯƠNG SAI (LEVENE)",
        "--------------------------------------------------------------------------------------",
        hyp_res["assumptions"].to_string(index=False),
        "",
        "--------------------------------------------------------------------------------------",
        "5. [MỤC C.5 - LỚP 2A] KIỂM ĐỊNH PHÂN NHÓM (ANOVA, WELCH, KRUSKAL-WALLIS & ETA²)",
        "--------------------------------------------------------------------------------------",
        hyp_res["group_tests"][
            ["Câu_hỏi_NC", "Biến_phân_tích", "Kiểm_định_tham_số", "Kiểm_định_Welch", "Kiểm_định_phi_tham_số", "Độ_lớn_tác_động"]
        ].to_string(index=False),
        "",
        "--------------------------------------------------------------------------------------",
        "6. [MỤC C.5 - LỚP 2B] KIỂM ĐỊNH TƯƠNG QUAN BIẾN LIÊN TỤC (RQ2, RQ3, RQ4, RQ5)",
        "--------------------------------------------------------------------------------------",
        hyp_res["cont_tests"][
            ["Câu_hỏi_NC", "Biến", "Pearson_r", "P_Value_Pearson", "Spearman_rho", "Chiều_tác_động"]
        ].to_string(index=False),
        "",
        "--------------------------------------------------------------------------------------",
        "7. [MỤC C.5 - LỚP 3] HẬU ĐỊNH TUKEY HSD GIỮA 5 TIỂU KHU VỰC (CHỈ RÕ CẶP CHÊNH LỆCH)",
        "--------------------------------------------------------------------------------------",
        hyp_res["tukey_subdistrict"].to_string(index=False),
        "",
        "--------------------------------------------------------------------------------------",
        "8. [BƯỚC 1.4] ĐÚC KẾT QUYẾT ĐỊNH KỸ THUẬT CHUYỂN GIAO SANG GIAI ĐOẠN 2 (ML PIPELINE)",
        "--------------------------------------------------------------------------------------",
        f"  [1] Biến mục tiêu: Giá gốc lệch phải (Skewness = {skew_raw:+.2f}), khi lấy ln(giá) giảm còn {skew_log:+.2f}.",
        "      -> Ở Giai đoạn 2 nên đối chứng thêm biến đổi log-target để tối ưu sai số tỷ lệ MAPE (%).",
        "  [2] Biến không gian & phân khúc: Tách `sub_district` (5 khu vực) làm tăng mạnh Eta² so với",
        "      chỉ gộp 3 quận chính; đồng thời `is_commercial_project` tách bạch hoàn toàn 2 mặt bằng giá.",
        "      -> Bắt buộc đưa `sub_district`, `apartment_type`, `is_commercial_project` (One-Hot) cùng",
        "         `ward` và `project_name` (Target Encoding chống Data Leakage) vào Pipeline.",
        "  [3] Cảnh báo Đa cộng tuyến (|r| >= 0.60 giữa các biến đầu vào):",
    ]
    if high_corr_pairs:
        for pair in high_corr_pairs:
            lines.append(f"      - {pair}")
    else:
        lines.append("      - Không có cặp biến nào vượt ngưỡng |r| >= 0.60.")
    lines += [
        "      -> Cần kiểm tra hệ số VIF ở mô hình OLS và dùng điều chuẩn Ridge ở nhánh tuyến tính.",
        "",
        "DANH SÁCH FILE KẾT QUẢ ĐÃ XUẤT:",
        f"  - Thư mục bảng CSV : {PHASE1_TABLES_DIR.relative_to(PROJECT_ROOT)} (12 bảng chuẩn)",
        f"  - Thư mục biểu đồ  : {PHASE1_FIGURES_DIR.relative_to(PROJECT_ROOT)} (5 biểu đồ 300 DPI)",
        f"  - File tóm tắt     : {PHASE1_REPORT_TXT.relative_to(PROJECT_ROOT)}",
        "=" * 86,
    ]
    return "\n".join(lines)


def main() -> None:
    if not DATASET1_CSV.exists():
        print(f"[LỖI] Không tìm thấy file {DATASET1_CSV}. Hãy chạy `python clean_chotot.py` trước.")
        sys.exit(1)

    ensure_phase1_dirs()
    setup_matplotlib_style()

    print("=" * 72)
    print("ĐANG CHẠY GIAI ĐOẠN 1 (DATASET 1): THỐNG KÊ, TRỰC QUAN HÓA & KIỂM ĐỊNH")
    print("=" * 72)

    df = pd.read_csv(DATASET1_CSV, encoding="utf-8-sig")
    print(f"[1/4] Đã đọc {len(df):,} dòng dữ liệu sạch từ {DATASET1_CSV.relative_to(PROJECT_ROOT)}")

    print("[2/4] Đang tính toán Thống kê mô tả đa tầng (Bước 1.1)...")
    desc_res = run_descriptive_pipeline(df, PHASE1_TABLES_DIR)

    print("[3/4] Đang kết xuất 5 biểu đồ EDA độ phân giải 300 DPI (Bước 1.2)...")
    pearson_df, _ = run_visualization_pipeline(df, PHASE1_FIGURES_DIR, PHASE1_TABLES_DIR)

    print("[4/4] Đang chạy Kiểm định giả thuyết 3 lớp & Hậu định Tukey HSD (Bước 1.3)...")
    hyp_res = run_hypothesis_pipeline(df, PHASE1_TABLES_DIR)

    report_text = build_phase1_report(df, desc_res, pearson_df, hyp_res)
    PHASE1_REPORT_TXT.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)


if __name__ == "__main__":
    main()
