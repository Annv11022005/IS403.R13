# -*- coding: utf-8 -*-
"""
Bước 1.3: Kiểm định giả thuyết thống kê 3 lớp cho Dataset 1 (Mục C.5 - RQ1 đến RQ6)
  - Lớp 1: Kiểm tra giả định phân phối chuẩn (D'Agostino-Pearson) & đồng nhất phương sai (Levene)
  - Lớp 2: Kiểm định chính (One-Way ANOVA / Welch t-test) + Đối chứng bền vững (Welch ANOVA,
           Kruskal-Wallis, Mann-Whitney U) + Độ lớn tác động thực tế (Eta-squared η², Cohen's d)
  - Lớp 3: Kiểm định hậu định cặp Tukey HSD (chỉ đích danh cặp khu vực / nội thất chênh lệch)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from src.config import FURNITURE_ORDER


def _fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def _eta_squared(groups: list[np.ndarray]) -> float:
    """Tính Eta-squared (η² = SS_between / SS_total) đo % phương sai giá giải thích bởi biến phân nhóm."""
    all_vals = np.concatenate(groups)
    grand_mean = float(np.mean(all_vals))
    ss_total = float(np.sum((all_vals - grand_mean) ** 2))
    if ss_total <= 0:
        return 0.0
    ss_between = sum(len(g) * float((np.mean(g) - grand_mean) ** 2) for g in groups)
    return float(ss_between / ss_total)


def _interpret_eta2(eta2: float) -> str:
    """Phân loại độ lớn tác động theo chuẩn Cohen (1988)."""
    if eta2 >= 0.14:
        return f"Lớn ({eta2 * 100:.1f}%)"
    if eta2 >= 0.06:
        return f"Trung bình ({eta2 * 100:.1f}%)"
    if eta2 >= 0.01:
        return f"Nhỏ ({eta2 * 100:.1f}%)"
    return f"Rất nhỏ ({eta2 * 100:.2f}%)"


def _welch_anova(groups: list[np.ndarray]) -> tuple[float, float]:
    """
    Tính chính xác kiểm định Welch's One-Way ANOVA (Welch, 1951)
    cho trường hợp phương sai giữa các nhóm không đồng nhất (Levene p < 0.05).
    """
    valid = [g for g in groups if len(g) >= 2 and np.var(g, ddof=1) > 1e-12]
    k = len(valid)
    if k < 2:
        return np.nan, np.nan

    ns = np.array([len(g) for g in valid], dtype=float)
    means = np.array([np.mean(g) for g in valid], dtype=float)
    vars_ = np.array([np.var(g, ddof=1) for g in valid], dtype=float)

    weights = ns / vars_
    w_sum = float(np.sum(weights))
    y_tilde = float(np.sum(weights * means) / w_sum)

    term = float(np.sum(((1.0 - weights / w_sum) ** 2) / (ns - 1.0)))
    denom = 1.0 + (2.0 * (k - 2.0) / (k ** 2 - 1.0)) * term
    f_welch = float((np.sum(weights * ((means - y_tilde) ** 2)) / (k - 1.0)) / denom)

    df1 = k - 1.0
    df2 = (k ** 2 - 1.0) / (3.0 * term) if term > 0 else np.inf
    p_welch = float(stats.f.sf(f_welch, df1, df2))
    return f_welch, p_welch


def check_statistical_assumptions(df: pd.DataFrame) -> pd.DataFrame:
    """Lớp 1: Kiểm tra phân phối chuẩn của biến mục tiêu và tính đồng nhất phương sai Levene."""
    rows = []

    # 1. Kiểm tra phân phối chuẩn của giá gốc và ln(giá)
    prices = df["price_million_vnd"].dropna().values
    log_p = np.log(np.clip(prices, 0.1, None))

    k2_raw, p_norm_raw = stats.normaltest(prices)
    k2_log, p_norm_log = stats.normaltest(log_p)

    rows.append({
        "Loại_kiểm_tra": "Phân phối chuẩn (D'Agostino K²)",
        "Đối_tượng": "price_million_vnd (Giá gốc)",
        "Thống_kê": round(float(k2_raw), 2),
        "P_Value": _fmt_p(float(p_norm_raw)),
        "Đánh_giá": f"Lệch phải (Skew={pd.Series(prices).skew():.2f}) -> Cần kèm Kruskal-Wallis / Mann-Whitney",
    })
    rows.append({
        "Loại_kiểm_tra": "Phân phối chuẩn (D'Agostino K²)",
        "Đối_tượng": "ln(price_million_vnd) (Log giá)",
        "Thống_kê": round(float(k2_log), 2),
        "P_Value": _fmt_p(float(p_norm_log)),
        "Đánh_giá": f"Cân đối hơn rõ rệt (Skew={pd.Series(log_p).skew():.2f})",
    })

    # 2. Kiểm định Levene (Brown-Forsythe center='median') về tính đồng nhất phương sai
    levene_targets = [
        ("district_std", "3 Quận chính (RQ1)"),
        ("sub_district", "5 Tiểu khu vực (RQ1 chi tiết)"),
        ("furniture", "4 Cấp nội thất (RQ4)"),
        ("has_pool", "Hồ bơi Có/Không (RQ6)"),
        ("has_gym", "Phòng Gym Có/Không (RQ6)"),
    ]
    for col, desc in levene_targets:
        groups = [
            grp["price_million_vnd"].dropna().values
            for _, grp in df.groupby(col)
            if len(grp["price_million_vnd"].dropna()) >= 2
        ]
        if len(groups) >= 2:
            w_stat, p_lev = stats.levene(*groups, center="median")
            equal_var = p_lev >= 0.05
            rows.append({
                "Loại_kiểm_tra": "Đồng nhất phương sai (Levene)",
                "Đối_tượng": desc,
                "Thống_kê": round(float(w_stat), 2),
                "P_Value": _fmt_p(float(p_lev)),
                "Đánh_giá": (
                    "Phương sai đồng nhất (p >= 0.05)"
                    if equal_var
                    else "Phương sai khác biệt (p < 0.05) -> Bổ sung Welch ANOVA / Welch t-test"
                ),
            })

    return pd.DataFrame(rows)


def run_group_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Lớp 2A: Kiểm định cho các biến phân nhóm (RQ1, RQ4, RQ6 và Phân khúc căn hộ)."""
    rows = []

    # A. Biến >= 3 nhóm: One-Way ANOVA + Welch ANOVA + Kruskal-Wallis + Eta²
    multi_specs = [
        ("RQ1", "district_std", "Giá thuê giữa 3 quận chính (Bình Thạnh, Q7, Thủ Đức)", None),
        ("RQ1 (Chi tiết)", "sub_district", "Giá thuê giữa 5 tiểu khu vực (tách Q2, Q9, Thủ Đức cũ)", None),
        ("RQ4", "furniture", "Giá thuê theo 4 tình trạng nội thất", FURNITURE_ORDER),
    ]
    for rq, col, desc, custom_order in multi_specs:
        if custom_order:
            names = [x for x in custom_order if x in set(df[col].dropna())]
        else:
            names = df.groupby(col)["price_million_vnd"].mean().sort_values(ascending=False).index.tolist()

        groups = [df.loc[df[col] == n, "price_million_vnd"].dropna().values for n in names]
        groups = [g for g in groups if len(g) >= 2]

        f_stat, p_anova = stats.f_oneway(*groups)
        f_welch, p_welch = _welch_anova(groups)
        h_kw, p_kw = stats.kruskal(*groups)
        eta2 = _eta_squared(groups)

        rows.append({
            "Câu_hỏi_NC": rq,
            "Biến_phân_tích": col,
            "Nội_dung_kiểm_định": desc,
            "Kiểm_định_tham_số": f"ANOVA F = {f_stat:.2f} (p {_fmt_p(p_anova)})",
            "Kiểm_định_Welch": f"Welch F = {f_welch:.2f} (p {_fmt_p(p_welch)})",
            "Kiểm_định_phi_tham_số": f"Kruskal H = {h_kw:.2f} (p {_fmt_p(p_kw)})",
            "Eta_Squared": round(eta2, 4),
            "Độ_lớn_tác_động": _interpret_eta2(eta2),
            "Kết_luận": "Bác bỏ H0 (Khác biệt có ý nghĩa thống kê)" if p_welch < 0.05 else "Không bác bỏ H0",
        })

    # B. Biến nhị phân 2 nhóm: Welch t-test + Mann-Whitney U + Chênh lệch Mean + Eta²
    binary_specs = [
        ("RQ6", "has_pool", "Chênh lệch giá thuê khi có Hồ bơi"),
        ("RQ6", "has_gym", "Chênh lệch giá thuê khi có Phòng Gym"),
        ("Phân khúc", "is_commercial_project", "Chênh lệch Chung cư thương mại vs CHDV/Mini"),
    ]
    for rq, col, desc in binary_specs:
        g1 = df.loc[df[col] == 1, "price_million_vnd"].dropna().values
        g0 = df.loc[df[col] == 0, "price_million_vnd"].dropna().values
        diff = float(np.mean(g1) - np.mean(g0))

        t_val, p_t = stats.ttest_ind(g1, g0, equal_var=False)
        u_val, p_mw = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        eta2 = _eta_squared([g0, g1])

        rows.append({
            "Câu_hỏi_NC": rq,
            "Biến_phân_tích": col,
            "Nội_dung_kiểm_định": f"{desc} (+{diff:.2f} triệu/tháng)",
            "Kiểm_định_tham_số": f"Welch t = {t_val:.2f} (p {_fmt_p(p_t)})",
            "Kiểm_định_Welch": f"ΔMean = +{diff:.2f} tr",
            "Kiểm_định_phi_tham_số": f"Mann-Whitney p {_fmt_p(p_mw)}",
            "Eta_Squared": round(eta2, 4),
            "Độ_lớn_tác_động": _interpret_eta2(eta2),
            "Kết_luận": "Bác bỏ H0 (Tăng giá có ý nghĩa thống kê)" if p_t < 0.05 else "Không bác bỏ H0",
        })

    return pd.DataFrame(rows)


def run_continuous_correlation_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Lớp 2B: Kiểm định ý nghĩa thống kê hệ số tương quan cho RQ2, RQ3, RQ4, RQ5."""
    specs = [
        ("RQ2", "area_m2", "Diện tích căn hộ (m²)"),
        ("RQ3", "bedrooms", "Số phòng ngủ"),
        ("RQ3", "bathrooms", "Số phòng vệ sinh (WC)"),
        ("RQ4", "furniture_score", "Điểm chất lượng nội thất (0–3)"),
        ("RQ5", "distance_to_center_km", "Khoảng cách tới Trung tâm Q1 (km)"),
    ]
    y = df["price_million_vnd"].values
    rows = []

    for rq, col, desc in specs:
        x = df[col].values
        r_val, p_r = stats.pearsonr(x, y)
        rho_val, p_rho = stats.spearmanr(x, y)

        direction = "Thuận chiều (+)" if r_val > 0 else "Ngược chiều (-)"
        rows.append({
            "Câu_hỏi_NC": rq,
            "Biến": col,
            "Mô_tả": desc,
            "Pearson_r": round(float(r_val), 4),
            "P_Value_Pearson": _fmt_p(float(p_r)),
            "Spearman_rho": round(float(rho_val), 4),
            "P_Value_Spearman": _fmt_p(float(p_rho)),
            "Chiều_tác_động": direction,
            "Kết_luận": (
                f"Có tương quan {direction.lower()} ý nghĩa thống kê"
                if p_r < 0.05 else "Không có tương quan tuyến tính rõ rệt"
            ),
        })
    return pd.DataFrame(rows)


def run_tukey_posthoc(df: pd.DataFrame, group_col: str, custom_order: list[str] | None = None) -> pd.DataFrame:
    """Lớp 3: Kiểm định hậu định cặp Tukey HSD giữa từng cặp nhóm."""
    if custom_order:
        names = [x for x in custom_order if x in set(df[group_col].dropna())]
    else:
        names = df.groupby(group_col)["price_million_vnd"].mean().sort_values(ascending=False).index.tolist()

    samples = [df.loc[df[group_col] == n, "price_million_vnd"].dropna().values for n in names]
    rows = []

    if hasattr(stats, "tukey_hsd"):
        tuk = stats.tukey_hsd(*samples)
        ci = tuk.confidence_interval(confidence_level=0.95)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = float(np.mean(samples[i]) - np.mean(samples[j]))
                p_ij = float(tuk.pvalue[i, j])
                rows.append({
                    "Nhóm_A": names[i],
                    "Nhóm_B": names[j],
                    "Mean_A": round(float(np.mean(samples[i])), 2),
                    "Mean_B": round(float(np.mean(samples[j])), 2),
                    "Chênh_lệch_(A-B)": round(diff, 2),
                    "CI_95_Low": round(float(ci.low[i, j]), 2),
                    "CI_95_High": round(float(ci.high[i, j]), 2),
                    "P_Value": _fmt_p(p_ij),
                    "Khác_biệt_ý_nghĩa_(p<0.05)": "Có" if p_ij < 0.05 else "Không",
                })
    else:
        n_pairs = max(len(names) * (len(names) - 1) / 2.0, 1.0)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = float(np.mean(samples[i]) - np.mean(samples[j]))
                _, p_raw = stats.ttest_ind(samples[i], samples[j], equal_var=False)
                p_adj = min(1.0, float(p_raw) * n_pairs)
                rows.append({
                    "Nhóm_A": names[i],
                    "Nhóm_B": names[j],
                    "Mean_A": round(float(np.mean(samples[i])), 2),
                    "Mean_B": round(float(np.mean(samples[j])), 2),
                    "Chênh_lệch_(A-B)": round(diff, 2),
                    "P_Value": _fmt_p(p_adj),
                    "Khác_biệt_ý_nghĩa_(p<0.05)": "Có" if p_adj < 0.05 else "Không",
                })

    return pd.DataFrame(rows)


def run_hypothesis_pipeline(df: pd.DataFrame, table_dir: Path) -> dict[str, pd.DataFrame]:
    """Thực thi toàn bộ Bước 1.3 và xuất các bảng CSV kết quả kiểm định."""
    assumptions_df = check_statistical_assumptions(df)
    group_tests_df = run_group_hypothesis_tests(df)
    cont_tests_df = run_continuous_correlation_tests(df)

    tukey_dist_df = run_tukey_posthoc(df, "district_std")
    tukey_sub_df = run_tukey_posthoc(df, "sub_district")
    tukey_furn_df = run_tukey_posthoc(df, "furniture", custom_order=FURNITURE_ORDER)

    assumptions_df.to_csv(table_dir / "c5_assumption_checks.csv", index=False, encoding="utf-8-sig")
    group_tests_df.to_csv(table_dir / "c5_group_tests_anova_ttest.csv", index=False, encoding="utf-8-sig")
    cont_tests_df.to_csv(table_dir / "c5_continuous_tests_rq2_rq3_rq5.csv", index=False, encoding="utf-8-sig")
    tukey_dist_df.to_csv(table_dir / "c5_tukey_posthoc_district.csv", index=False, encoding="utf-8-sig")
    tukey_sub_df.to_csv(table_dir / "c5_tukey_posthoc_subdistrict.csv", index=False, encoding="utf-8-sig")
    tukey_furn_df.to_csv(table_dir / "c5_tukey_posthoc_furniture.csv", index=False, encoding="utf-8-sig")

    return {
        "assumptions": assumptions_df,
        "group_tests": group_tests_df,
        "cont_tests": cont_tests_df,
        "tukey_district": tukey_dist_df,
        "tukey_subdistrict": tukey_sub_df,
        "tukey_furniture": tukey_furn_df,
    }
