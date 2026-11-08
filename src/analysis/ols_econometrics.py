# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def _fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def compute_vif_vector(X_no_intercept: np.ndarray) -> np.ndarray:
    # Tính hệ số đa cộng tuyến VIF
    corr = np.corrcoef(X_no_intercept, rowvar=False)
    corr_inv = np.linalg.pinv(corr)
    vif = np.diag(corr_inv)
    return np.clip(vif, 1.0, None)


def fit_ols_table(
    df: pd.DataFrame,
    feature_specs: list[tuple[str, str]],
    target_col: str = "price_million_vnd",
) -> tuple[pd.DataFrame, dict[str, float]]:
    # Ước lượng mô hình OLS và xuất bảng hệ số
    cols = [c for c, _ in feature_specs]
    labels = [lbl for _, lbl in feature_specs]

    work = df[cols + [target_col]].dropna().copy()
    n = len(work)
    k = len(cols)

    X_raw = work[cols].values.astype(float)
    y = work[target_col].values.astype(float)
    X = np.column_stack([np.ones(n), X_raw])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat

    df_resid = max(n - k - 1, 1)
    df_model = max(k, 1)

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid

    mse = ss_res / df_resid
    ms_model = (ss_tot - ss_res) / df_model
    f_stat = ms_model / max(mse, 1e-12)
    f_pval = float(stats.f.sf(f_stat, df_model, df_resid))

    cov_beta = mse * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov_beta), 1e-15, None))
    t_stats = beta / se
    p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=df_resid)
    t_crit = float(stats.t.ppf(0.975, df=df_resid))

    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se

    y_std = float(np.std(y, ddof=1))
    x_stds = np.std(X_raw, axis=0, ddof=1)
    std_betas = beta[1:] * (x_stds / max(y_std, 1e-12))

    vifs = compute_vif_vector(X_raw)

    rows = [{
        "Biến": "Intercept (Hệ số chặn)",
        "Mô_tả": "Giá cơ sở khi các biến bằng 0",
        "Coefficient": round(float(beta[0]), 4),
        "Std_Error": round(float(se[0]), 4),
        "t_Stat": round(float(t_stats[0]), 3),
        "P_Value": _fmt_p(float(p_vals[0])),
        "CI_95_Low": round(float(ci_low[0]), 3),
        "CI_95_High": round(float(ci_high[0]), 3),
        "Beta_Chuẩn_hóa": np.nan,
        "VIF": np.nan,
        "Kết_luận": "Có ý nghĩa thống kê" if p_vals[0] < 0.05 else "Không ý nghĩa",
    }]

    for idx, (col_name, label_vi) in enumerate(zip(cols, labels)):
        b_i = float(beta[idx + 1])
        se_i = float(se[idx + 1])
        t_i = float(t_stats[idx + 1])
        p_i = float(p_vals[idx + 1])
        vif_i = float(vifs[idx])

        if p_i < 0.05:
            direction = "Tăng giá (+)" if b_i > 0 else "Giảm giá (-)"
            concl = f"Có ý nghĩa ({direction})"
        else:
            concl = "Không có ý nghĩa thống kê (p >= 0.05)"

        rows.append({
            "Biến": col_name,
            "Mô_tả": label_vi,
            "Coefficient": round(b_i, 4),
            "Std_Error": round(se_i, 4),
            "t_Stat": round(t_i, 3),
            "P_Value": _fmt_p(p_i),
            "CI_95_Low": round(float(ci_low[idx + 1]), 3),
            "CI_95_High": round(float(ci_high[idx + 1]), 3),
            "Beta_Chuẩn_hóa": round(float(std_betas[idx]), 4),
            "VIF": round(vif_i, 2),
            "Kết_luận": concl,
        })

    summary_stats = {
        "n_obs": float(n),
        "r2": round(r2, 4),
        "adj_r2": round(adj_r2, 4),
        "f_stat": round(f_stat, 2),
        "f_pval": f_pval,
        "rmse": round(float(np.sqrt(np.mean(residuals ** 2))), 3),
    }
    return pd.DataFrame(rows), summary_stats


def run_ols_pipeline(df: pd.DataFrame, table_dir: Path) -> dict[str, object]:
    work = df.copy()

    base_specs = [
        ("area_m2", "Diện tích (m²)"),
        ("bedrooms", "Số phòng ngủ"),
        ("bathrooms", "Số phòng vệ sinh (WC)"),
        ("furniture_score", "Điểm nội thất (0–3)"),
        ("distance_to_center_km", "Khoảng cách tới Q1 (km)"),
        ("has_pool", "Tiện ích Hồ bơi (0/1)"),
        ("has_gym", "Tiện ích Phòng Gym (0/1)"),
    ]
    ols_base_df, ols_base_meta = fit_ols_table(work, base_specs)

    work["is_quan_2"] = (work["sub_district"] == "Quận 2 (Thủ Đức)").astype(float)
    work["is_binh_thanh"] = (work["sub_district"] == "Bình Thạnh").astype(float)
    work["is_quan_7"] = (work["sub_district"] == "Quận 7").astype(float)

    ext_specs = [
        ("area_m2", "Diện tích (m²)"),
        ("bedrooms", "Số phòng ngủ"),
        ("bathrooms", "Số phòng vệ sinh (WC)"),
        ("furniture_score", "Điểm nội thất (0–3)"),
        ("distance_to_center_km", "Khoảng cách tới Q1 (km)"),
        ("has_pool", "Tiện ích Hồ bơi (0/1)"),
        ("has_gym", "Tiện ích Phòng Gym (0/1)"),
        ("is_commercial_project", "Chung cư thương mại (0/1)"),
        ("is_quan_2", "Khu vực: Quận 2 (so với Q9/TĐ cũ)"),
        ("is_binh_thanh", "Khu vực: Bình Thạnh (so với Q9/TĐ cũ)"),
        ("is_quan_7", "Khu vực: Quận 7 (so với Q9/TĐ cũ)"),
    ]
    ols_ext_df, ols_ext_meta = fit_ols_table(work, ext_specs)

    ols_base_df.to_csv(table_dir / "c6_ols_baseline_table.csv", index=False, encoding="utf-8-sig")
    ols_ext_df.to_csv(table_dir / "c6_ols_extended_spatial_table.csv", index=False, encoding="utf-8-sig")

    return {
        "base_table": ols_base_df,
        "base_meta": ols_base_meta,
        "ext_table": ols_ext_df,
        "ext_meta": ols_ext_meta,
    }

