# -*- coding: utf-8 -*-
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_validation import load_validated_data

from src.analysis.ols_econometrics import run_ols_pipeline
from src.analysis.ts_stationarity_eda import (
    compute_ts_descriptive_stats,
    plot_ts_eda_figures,
    run_adf_stationarity_suite,
    run_macro_correlations_and_regimes,
)
from src.config import (
    DATASET1_CSV,
    DATASET2_CSV,
    MODELING_FIGURES_DIR,
    MODELING_REPORT_TXT,
    MODELING_TABLES_DIR,
    TS_FIGURES_DIR,
    TS_REPORT_TXT,
    TS_TABLES_DIR,
    ensure_modeling_dirs,
    ensure_ts_dirs,
    setup_matplotlib_style,
)
from src.model.ml_models import HAS_XGBOOST, run_ml_pipeline
from src.model.ts_models import run_timeseries_modeling_suite


def build_modeling_report(ols_res: dict[str, object], ml_res: dict[str, object]) -> str:
    # Tổng hợp báo cáo mô hình hóa trên Dataset 1 (OLS + 5 mô hình ML)
    base_meta = ols_res["base_meta"]
    ext_meta = ols_res["ext_meta"]
    res_df: pd.DataFrame = ml_res["results_df"]
    imp_df: pd.DataFrame = ml_res["importance_df"]
    best_row = res_df.iloc[0]

    lines = [
        "=" * 90,
        "BÁO CÁO KẾT QUẢ GIAI ĐOẠN 2, 3 & 4 (DATASET 1): HỒI QUY OLS & MACHINE LEARNING",
        "=" * 90,
        f"Quy mô chia dữ liệu (Stratified 80/20): Train = {ml_res['train_size']:,} căn | Test = {ml_res['test_size']:,} căn",
        f"Tổng số đặc trưng sau Pipeline        : {ml_res['n_features']} biến (One-Hot + Out-of-Fold Target Encoding)",
        f"Trạng thái thư viện XGBoost           : {'Đã kích hoạt XGBRegressor chuẩn' if HAS_XGBOOST else 'Dùng HistGradientBoosting (cài thêm `pip install xgboost` nếu muốn)'}",
        "",
        "------------------------------------------------------------------------------------------",
        f"1. [MỤC C.6] HỒI QUY ĐA BIẾN OLS CƠ BẢN (R² = {base_meta['r2']:.4f}, Adj R² = {base_meta['adj_r2']:.4f}, F = {base_meta['f_stat']:.2f})",
        "------------------------------------------------------------------------------------------",
        ols_res["base_table"][
            ["Biến", "Mô_tả", "Coefficient", "Std_Error", "t_Stat", "P_Value", "VIF", "Kết_luận"]
        ].to_string(index=False),
        "",
        "------------------------------------------------------------------------------------------",
        f"2. [MỤC C.6 MỞ RỘNG] HỒI QUY OLS KIỂM SOÁT KHU VỰC & PHÂN KHÚC (R² = {ext_meta['r2']:.4f}, Adj R² = {ext_meta['adj_r2']:.4f})",
        "------------------------------------------------------------------------------------------",
        ols_res["ext_table"][
            ["Biến", "Mô_tả", "Coefficient", "Std_Error", "t_Stat", "P_Value", "VIF", "Kết_luận"]
        ].to_string(index=False),
        "",
        "------------------------------------------------------------------------------------------",
        "3. [MỤC D.11 & RQ10] BẢNG SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH MACHINE LEARNING THEO CV; CHỈ MÔ HÌNH ĐƯỢC CHỌN ĐÁNH GIÁ TRÊN TEST",
        "------------------------------------------------------------------------------------------",
        res_df[
            ["Mô_hình", "Nhánh_Pipeline", "CV_5Fold_R2", "CV_5Fold_RMSE", "Được_chọn_từ_CV", "Test_MAE", "Test_RMSE", "Test_MAPE", "Test_R2"]
        ].to_string(index=False),
        "",
        "------------------------------------------------------------------------------------------",
        "4. [TRẢ LỜI RQ7] TOP 10 ĐẶC TRƯNG QUAN TRỌNG TRONG MÔ HÌNH CÂY",
        "------------------------------------------------------------------------------------------",
        imp_df.head(10)[
            ["Biến", "Mô_tả", "Importance_RF", "Importance_XGB", "Importance_Tree", "Permutation_Importance"]
        ].to_string(index=False),
        "",
        "------------------------------------------------------------------------------------------",
        "5. KẾT LUẬN CHÍNH CHO BÁO CÁO (MỤC D.11, E.1, E.3 - DATASET 1):",
        "------------------------------------------------------------------------------------------",
        f"  - Mô hình được chọn theo CV trên Dataset 1 là `{best_row['Mô_hình']}` với kết quả trên tập Test:",
        f"      + MAE  = {best_row['Test_MAE']:.4f} triệu VNĐ/tháng",
        f"      + RMSE = {best_row['Test_RMSE']:.4f} triệu VNĐ/tháng",
        f"      + MAPE = {best_row['Test_MAPE']:.4f} (tỷ lệ thập phân; có thể lớn hơn 1)",
        f"      + R²   = {best_row['Test_R2']:.4f} (không có đơn vị; có thể âm)",
        "  - Mô hình và cấu hình được chọn theo RMSE 5-fold CV trên tập train; test chỉ đánh giá mô hình được chọn.",
        "  - CV dùng lại để chọn siêu tham số nên chỉ số CV có thể lạc quan.",
        "  - Tập test đã từng dùng chọn mô hình ở phiên bản trước; cần dữ liệu mới để kiểm định độc lập hơn.",
        "",
        "DANH SÁCH FILE KẾT QUẢ ĐÃ XUẤT:",
        f"  - Thư mục bảng CSV : {MODELING_TABLES_DIR.relative_to(PROJECT_ROOT)} (5 bảng chuẩn)",
        f"  - Thư mục biểu đồ  : {MODELING_FIGURES_DIR.relative_to(PROJECT_ROOT)} (4 biểu đồ 300 DPI: Hình 06 -> 09)",
        f"  - File báo cáo     : {MODELING_REPORT_TXT.relative_to(PROJECT_ROOT)}",
        "=" * 90,
    ]
    return "\n".join(lines)


def main() -> None:
    # Phân tích riêng dữ liệu căn hộ và dữ liệu chuỗi thời gian
    df1, df2 = load_validated_data()
    ensure_modeling_dirs()
    ensure_ts_dirs()
    setup_matplotlib_style()

    ols_res = run_ols_pipeline(df1, MODELING_TABLES_DIR)
    ml_res = run_ml_pipeline(df1, MODELING_TABLES_DIR, MODELING_FIGURES_DIR)
    modeling_report = build_modeling_report(ols_res, ml_res)
    MODELING_REPORT_TXT.write_text(modeling_report, encoding="utf-8")
    print(modeling_report)

    desc_df = compute_ts_descriptive_stats(df2, TS_TABLES_DIR)
    adf_df = run_adf_stationarity_suite(df2, TS_TABLES_DIR)
    corr_df, regime_df = run_macro_correlations_and_regimes(df2, TS_TABLES_DIR)
    plot_ts_eda_figures(df2, adf_df, TS_FIGURES_DIR)

    ts_res = run_timeseries_modeling_suite(df2, TS_TABLES_DIR, TS_FIGURES_DIR)
    comp_df = ts_res["comparison_df"]
    pred_df = ts_res["predictions_df"]
    future_df = ts_res["future_df"]

    ts_lines = [
        "=" * 88,
        "BÁO CÁO KẾT QUẢ PHÂN TÍCH & DỰ BÁO CHUỖI THỜI GIAN (DATASET 2: 2019-01 -> 2024-12)",
        "=" * 88,
        f"- Tổng số quan sát: {len(df2)} tháng (Train: 60 tháng 2019-2023 | Test: 12 tháng 2024)",
        "- Quy ước đơn vị đồng nhất: Sai số tuyệt đối (MAE, RMSE) tính bằng triệu VNĐ/tháng;",
        "  MAPE là tỷ lệ thập phân (0,12 tương đương 12%); MAPE có thể lớn hơn 1.",
        "  R² không có đơn vị và có thể âm; RMSE là sai số tính bằng triệu VNĐ/tháng.",
        "",
        "1. THỐNG KÊ MÔ TẢ CHUỖI THỜI GIAN (ts1_descriptive_stats.csv)",
        "-" * 88,
        desc_df.to_string(index=False),
        "",
        "2. KIỂM ĐỊNH NGHIỆM ĐƠN VỊ ADF - BẬC GỐC I(0) & SAI PHÂN BẬC 1 I(1) (ts2_adf_stationarity_tests.csv)",
        "-" * 88,
        adf_df.to_string(index=False),
        "",
        "3. DIỄN BIẾN THỊ TRƯỜNG QUA 4 GIAI ĐOẠN - TRẢ LỜI RQ8 (ts4_regime_summary_rq8.csv)",
        "-" * 88,
        regime_df.to_string(index=False),
        "",
        "4. TƯƠNG QUAN VỚI BIẾN VĨ MÔ & GOOGLE TRENDS - TRẢ LỜI RQ9 (ts3_macro_correlations_rq9.csv)",
        "-" * 88,
        corr_df.to_string(index=False),
        "",
        "5. SO SÁNH MÔ HÌNH DỰ BÁO CHUỖI THỜI GIAN TRÊN TẬP TEST 2024 - TRẢ LỜI RQ10 (ts5_model_comparison_rq10.csv)",
        "-" * 88,
        "Đánh giá cuốn chiếu một bước; chọn mô hình theo RMSE validation 2023, test là 2024.",
        "Ngoại sinh dùng dữ liệu tháng trước, giả định đã công bố; xem ts10_fit_warnings.csv.",
        "Mô hình có lần không hội tụ trên validation không được chọn để dự báo.",
        ts_res["rolling_metrics_df"].to_string(index=False),
        "Mô hình được chọn: " + ts_res["selected_model"],
        comp_df.to_string(index=False),
        "",
        "6. PHẠM VI SO SÁNH KẾT QUẢ",
        "-" * 88,
        "Dataset 1 dự báo giá từng căn hộ; Dataset 2 dự báo chuỗi giá trung bình theo tháng.",
        "Đơn vị MAE/RMSE giống nhau nhưng không dùng để xếp hạng mô hình giữa hai bài toán.",
        "Xem kết quả căn hộ trong outputs/dataset1_modeling/; kết quả chuỗi thời gian ở mục 5.",
        "Nguồn gốc dữ liệu chưa được xác minh; xem data/README.md.",
        "",
        "7. ĐỐI CHIẾU GIÁ THUÊ THỰC TẾ VÀ DỰ BÁO TỪNG THÁNG NĂM 2024 (ts6_test_2024_monthly_predictions.csv)",
        "-" * 88,
        pred_df.to_string(index=False),
        "",
        "8. DỰ BÁO XU HƯỚNG GIÁ THUÊ 12 THÁNG NĂM 2025 (ts7_future_forecast_2025.csv)",
        "-" * 88,
        "Dự báo bằng mô hình được chọn trên validation, dùng toàn bộ dữ liệu lịch sử.",
        "Nếu mô hình dùng ngoại sinh, giả định biến giữ nguyên giá trị tháng cuối; chưa ước lượng khoảng dự báo.",
        future_df.to_string(index=False),
        "",
        "9. DANH SÁCH 5 BIỂU ĐỒ CHUẨN IN ẤN (300 DPI) ĐÃ XUẤT",
        "-" * 88,
        f"- {TS_FIGURES_DIR / '10_timeseries_overview_rq8.png'}",
        f"- {TS_FIGURES_DIR / '11_adf_differencing_acf_pacf.png'}",
        f"- {TS_FIGURES_DIR / '12_macro_correlations_rq9.png'}",
        f"- {TS_FIGURES_DIR / '13_ts_model_comparison_rq10.png'}",
        f"- {TS_FIGURES_DIR / '14_ts_full_trajectory_and_2025_forecast.png'}",
        "=" * 88,
    ]
    ts_report = "\n".join(ts_lines)
    TS_REPORT_TXT.write_text(ts_report, encoding="utf-8")
    print("\n" + ts_report)


if __name__ == "__main__":
    main()

