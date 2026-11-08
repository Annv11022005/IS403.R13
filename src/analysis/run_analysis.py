# -*- coding: utf-8 -*-
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis.descriptive_stats import run_descriptive_pipeline
from src.analysis.hypothesis_testing import run_hypothesis_pipeline
from src.analysis.ols_econometrics import run_ols_pipeline
from src.analysis.ts_stationarity_eda import (
    compute_ts_descriptive_stats,
    plot_ts_eda_figures,
    run_adf_stationarity_suite,
    run_macro_correlations_and_regimes,
)
from src.analysis.visualization import run_visualization_pipeline
from src.config import (
    DATASET1_CSV,
    DATASET2_CSV,
    MODELING_TABLES_DIR,
    PHASE1_FIGURES_DIR,
    PHASE1_REPORT_TXT,
    PHASE1_TABLES_DIR,
    TS_FIGURES_DIR,
    TS_TABLES_DIR,
    ensure_modeling_dirs,
    ensure_phase1_dirs,
    ensure_ts_dirs,
    setup_matplotlib_style,
)


def build_phase1_report(
    df: pd.DataFrame,
    desc_res: dict[str, pd.DataFrame],
    pearson_df: pd.DataFrame,
    hyp_res: dict[str, pd.DataFrame],
) -> str:
    # Tổng hợp báo cáo Giai đoạn 1 (Dataset 1)
    num_df = desc_res["numeric"].set_index("Biến")
    skew_raw = num_df.loc["price_million_vnd", "Skewness"]
    skew_log = num_df.loc["log_price_million_vnd", "Skewness"]

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
            ["district_std", "Count", "Mean", "Median", "Std", "Skewness", "Kurtosis", "Mean_Đơn_giá_m2", "Tỷ_lệ_Chung_cư_TM"]
        ].to_string(index=False),
        "",
        "--------------------------------------------------------------------------------------",
        "2. [MỤC C.1 MỞ RỘNG] THỐNG KÊ THEO 5 TIỂU KHU VỰC (TÁCH QUẬN 2, QUẬN 9, THỦ ĐỨC CŨ)",
        "--------------------------------------------------------------------------------------",
        desc_res["subdistrict"][
            ["sub_district", "Count", "Mean", "Median", "Std", "Skewness", "Kurtosis", "Mean_Đơn_giá_m2", "Tỷ_lệ_Chung_cư_TM"]
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
        "      -> Ở Giai đoạn 2 nên đối chứng thêm biến đổi log-target để tối ưu sai số tỷ lệ MAPE.",
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
    # Chạy toàn bộ nhánh Phân tích (Dataset 1 EDA/ANOVA/OLS & Dataset 2 ADF/Tương quan vĩ mô)
    ensure_phase1_dirs()
    ensure_modeling_dirs()
    ensure_ts_dirs()
    setup_matplotlib_style()

    df1 = pd.read_csv(DATASET1_CSV, encoding="utf-8-sig")
    desc_res = run_descriptive_pipeline(df1, PHASE1_TABLES_DIR)
    pearson_df, _ = run_visualization_pipeline(df1, PHASE1_FIGURES_DIR, PHASE1_TABLES_DIR)
    hyp_res = run_hypothesis_pipeline(df1, PHASE1_TABLES_DIR)
    run_ols_pipeline(df1, MODELING_TABLES_DIR)

    report_text = build_phase1_report(df1, desc_res, pearson_df, hyp_res)
    PHASE1_REPORT_TXT.write_text(report_text, encoding="utf-8")
    print(report_text)

    df2 = pd.read_csv(DATASET2_CSV)
    compute_ts_descriptive_stats(df2, TS_TABLES_DIR)
    adf_df = run_adf_stationarity_suite(df2, TS_TABLES_DIR)
    run_macro_correlations_and_regimes(df2, TS_TABLES_DIR)
    plot_ts_eda_figures(df2, adf_df, TS_FIGURES_DIR)


if __name__ == "__main__":
    main()

