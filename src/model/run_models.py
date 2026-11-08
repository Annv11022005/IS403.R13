# -*- coding: utf-8 -*-
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

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
        "3. [MỤC D.11 & RQ10] BẢNG SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH MACHINE LEARNING TRÊN TẬP TEST",
        "------------------------------------------------------------------------------------------",
        res_df[
            ["Mô_hình", "Nhánh_Pipeline", "CV_5Fold_R2", "Train_R2", "Test_MAE", "Test_RMSE", "Test_MAPE", "Test_R2"]
        ].to_string(index=False),
        "",
        "------------------------------------------------------------------------------------------",
        "4. [TRẢ LỜI RQ7] TOP 10 YẾU TỐ QUAN TRỌNG NHẤT TRONG DỰ BÁO GIÁ THUÊ CĂN HỘ",
        "------------------------------------------------------------------------------------------",
        imp_df.head(10)[
            ["Biến", "Mô_tả", "Importance_RF", "Importance_XGB", "Importance_Tree", "Permutation_Importance"]
        ].to_string(index=False),
        "",
        "------------------------------------------------------------------------------------------",
        "5. KẾT LUẬN CHÍNH CHO BÁO CÁO (MỤC D.11, E.1, E.3 - DATASET 1):",
        "------------------------------------------------------------------------------------------",
        f"  - Mô hình dự báo tốt nhất trên Dataset 1 là `{best_row['Mô_hình']}` với kết quả trên tập Test:",
        f"      + MAE  = {best_row['Test_MAE']:.4f} triệu VNĐ/tháng",
        f"      + RMSE = {best_row['Test_RMSE']:.4f} triệu VNĐ/tháng",
        f"      + MAPE = {best_row['Test_MAPE']:.4f} (thang thập phân 0-1)",
        f"      + R²   = {best_row['Test_R2']:.4f} (thang thập phân 0-1)",
        "  - Các mô hình Ensemble dựa trên cây (Random Forest, XGBoost) vượt trội hơn mô hình tuyến tính",
        "    nhờ nắm bắt được tương tác phi tuyến giữa Vị trí (Dự án/Phường) x Quy mô diện tích x Phân khúc.",
        "",
        "DANH SÁCH FILE KẾT QUẢ ĐÃ XUẤT:",
        f"  - Thư mục bảng CSV : {MODELING_TABLES_DIR.relative_to(PROJECT_ROOT)} (5 bảng chuẩn)",
        f"  - Thư mục biểu đồ  : {MODELING_FIGURES_DIR.relative_to(PROJECT_ROOT)} (4 biểu đồ 300 DPI: Hình 06 -> 09)",
        f"  - File báo cáo     : {MODELING_REPORT_TXT.relative_to(PROJECT_ROOT)}",
        "=" * 90,
    ]
    return "\n".join(lines)


def build_unified_rq10_table(comp_ts_df: pd.DataFrame) -> pd.DataFrame:
    # Tổng hợp bảng so sánh toàn bộ 10 mô hình (5 ML + 5 Time Series) cùng đơn vị chuẩn
    ml_csv = MODELING_TABLES_DIR / "d11_ml_model_comparison.csv"
    rows = []
    if ml_csv.exists():
        ml_df = pd.read_csv(ml_csv)
        for _, r in ml_df.iterrows():
            mape_val = r["Test_MAPE"] if "Test_MAPE" in r else float(r["Test_MAPE_%"]) / 100.0
            rows.append({
                "Nhóm_mô_hình": "Dataset 1 - Machine Learning (Cấp căn hộ)",
                "Mô_hình": r["Mô_hình"],
                "MAE": round(float(r["Test_MAE"]), 4),
                "RMSE": round(float(r["Test_RMSE"]), 4),
                "MAPE": round(float(mape_val), 4),
                "R2": round(float(r["Test_R2"]), 4),
            })

    for _, r in comp_ts_df.iterrows():
        rows.append({
            "Nhóm_mô_hình": "Dataset 2 - Chuỗi thời gian (Toàn thị trường)",
            "Mô_hình": r["Mô_hình"],
            "MAE": round(float(r["MAE"]), 4),
            "RMSE": round(float(r["RMSE"]), 4),
            "MAPE": round(float(r["MAPE"]), 4),
            "R2": round(float(r["R2"]), 4),
        })

    unified_df = pd.DataFrame(rows)
    unified_df.to_csv(TS_TABLES_DIR / "rq10_all_10_models_summary.csv", index=False, encoding="utf-8-sig")
    return unified_df


def main() -> None:
    # Huấn luyện và đánh giá toàn bộ 10 mô hình dự báo (5 ML trên Dataset 1 & 5 Time Series trên Dataset 2)
    ensure_modeling_dirs()
    ensure_ts_dirs()
    setup_matplotlib_style()

    df1 = pd.read_csv(DATASET1_CSV, encoding="utf-8-sig")
    ols_res = run_ols_pipeline(df1, MODELING_TABLES_DIR)
    ml_res = run_ml_pipeline(df1, MODELING_TABLES_DIR, MODELING_FIGURES_DIR)
    modeling_report = build_modeling_report(ols_res, ml_res)
    MODELING_REPORT_TXT.write_text(modeling_report, encoding="utf-8")
    print(modeling_report)

    df2 = pd.read_csv(DATASET2_CSV)
    desc_df = compute_ts_descriptive_stats(df2, TS_TABLES_DIR)
    adf_df = run_adf_stationarity_suite(df2, TS_TABLES_DIR)
    corr_df, regime_df = run_macro_correlations_and_regimes(df2, TS_TABLES_DIR)
    plot_ts_eda_figures(df2, adf_df, TS_FIGURES_DIR)

    ts_res = run_timeseries_modeling_suite(df2, TS_TABLES_DIR, TS_FIGURES_DIR)
    comp_df = ts_res["comparison_df"]
    pred_df = ts_res["predictions_df"]
    future_df = ts_res["future_df"]
    unified_rq10_df = build_unified_rq10_table(comp_df)

    ts_lines = [
        "=" * 88,
        "BÁO CÁO KẾT QUẢ PHÂN TÍCH & DỰ BÁO CHUỖI THỜI GIAN (DATASET 2: 2019-01 -> 2024-12)",
        "=" * 88,
        f"- Tổng số quan sát: {len(df2)} tháng (Train: 60 tháng 2019-2023 | Test: 12 tháng 2024)",
        "- Quy ước đơn vị đồng nhất: Sai số tuyệt đối (MAE, RMSE) tính bằng triệu VNĐ/tháng;",
        "  tất cả các chỉ số tỷ lệ (MAPE, R², Feature Importance, Eta², CV) tính trên thang thập phân [0, 1].",
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
        "5. SO SÁNH 5 MÔ HÌNH DỰ BÁO CHUỖI THỜI GIAN TRÊN TẬP TEST 2024 - TRẢ LỜI RQ10 (ts5_model_comparison_rq10.csv)",
        "-" * 88,
        comp_df.to_string(index=False),
        "",
        "6. BẢNG TỔNG HỢP ĐỒNG NHẤT CẢ 10 MÔ HÌNH (5 ML + 5 TIME SERIES) - TRẢ LỜI RQ10 (rq10_all_10_models_summary.csv)",
        "-" * 88,
        unified_rq10_df.to_string(index=False),
        "",
        "7. ĐỐI CHIẾU GIÁ THUÊ THỰC TẾ VÀ DỰ BÁO TỪNG THÁNG NĂM 2024 (ts6_test_2024_monthly_predictions.csv)",
        "-" * 88,
        pred_df.to_string(index=False),
        "",
        "8. DỰ BÁO XU HƯỚNG GIÁ THUÊ 12 THÁNG NĂM 2025 (ts7_future_forecast_2025.csv)",
        "-" * 88,
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

