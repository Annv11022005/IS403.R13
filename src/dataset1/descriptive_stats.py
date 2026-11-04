# -*- coding: utf-8 -*-
"""
Bước 1.1: Thống kê mô tả đa tầng cho Dataset 1 (Mục C.1)
- Bảng 1: Tổng quan toàn bộ biến định lượng (kèm đối chiếu log(giá) giảm độ lệch phải)
- Bảng 2A & 2B: Thống kê giá thuê & đơn giá/m² theo 3 Quận chính và 5 Tiểu khu vực
- Bảng 3: Thống kê chéo Tiểu khu vực x Phân khúc căn hộ (giải trình Nghịch lý Simpson)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import NUMERIC_FEATURES_META


def compute_numeric_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Tính thống kê mô tả đầy đủ cho tất cả biến định lượng và biến log(price)."""
    work = df.copy()
    work["log_price_million_vnd"] = np.log(work["price_million_vnd"].clip(lower=0.1))

    features = list(NUMERIC_FEATURES_META) + [
        ("log_price_million_vnd", "Log tự nhiên giá thuê ln(giá)"),
    ]

    rows = []
    for col, desc in features:
        if col not in work.columns:
            continue
        s = work[col].dropna().astype(float)
        if s.empty:
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        rows.append({
            "Biến": col,
            "Mô tả": desc,
            "Count": int(len(s)),
            "Mean": round(float(s.mean()), 3),
            "Median": round(float(s.median()), 3),
            "Std": round(float(s.std(ddof=1)), 3),
            "IQR": round(q3 - q1, 3),
            "Skewness": round(float(s.skew()), 3),
            "Kurtosis": round(float(s.kurt()), 3),
            "Min": round(float(s.min()), 3),
            "Q1_25%": round(q1, 3),
            "Q3_75%": round(q3, 3),
            "Max": round(float(s.max()), 3),
        })
    return pd.DataFrame(rows)


def compute_spatial_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Tính thống kê mô tả giá thuê (price_million_vnd) kèm đơn giá/m² và tỷ lệ chung cư TM
    theo cột phân nhóm không gian (`district_std` hoặc `sub_district`).
    """
    total_n = len(df)
    rows = []

    for grp_name, grp in df.groupby(group_col):
        p = grp["price_million_vnd"].dropna().astype(float)
        pm2 = grp["price_per_m2"].dropna().astype(float)
        comm_ratio = float(grp["is_commercial_project"].mean() * 100.0)
        q1, q3 = float(p.quantile(0.25)), float(p.quantile(0.75))

        rows.append({
            group_col: grp_name,
            "Count": int(len(p)),
            "Tỷ_trọng_%": round(len(p) * 100.0 / max(total_n, 1), 1),
            "Mean": round(float(p.mean()), 2),
            "Median": round(float(p.median()), 2),
            "Std": round(float(p.std(ddof=1)), 2),
            "IQR": round(q3 - q1, 2),
            "Skewness": round(float(p.skew()), 2),
            "Kurtosis": round(float(p.kurt()), 2),
            "Min": round(float(p.min()), 2),
            "Max": round(float(p.max()), 2),
            "Mean_Đơn_giá_m2": round(float(pm2.mean()), 3),
            "Median_Đơn_giá_m2": round(float(pm2.median()), 3),
            "Tỷ_lệ_Chung_cư_TM_%": round(comm_ratio, 1),
        })

    res = pd.DataFrame(rows).sort_values("Mean", ascending=False).reset_index(drop=True)

    # Bổ sung dòng Tổng thể ở cuối bảng
    p_all = df["price_million_vnd"].dropna().astype(float)
    pm2_all = df["price_per_m2"].dropna().astype(float)
    overall = pd.DataFrame([{
        group_col: "Toàn bộ mẫu",
        "Count": int(len(p_all)),
        "Tỷ_trọng_%": 100.0,
        "Mean": round(float(p_all.mean()), 2),
        "Median": round(float(p_all.median()), 2),
        "Std": round(float(p_all.std(ddof=1)), 2),
        "IQR": round(float(p_all.quantile(0.75) - p_all.quantile(0.25)), 2),
        "Skewness": round(float(p_all.skew()), 2),
        "Kurtosis": round(float(p_all.kurt()), 2),
        "Min": round(float(p_all.min()), 2),
        "Max": round(float(p_all.max()), 2),
        "Mean_Đơn_giá_m2": round(float(pm2_all.mean()), 3),
        "Median_Đơn_giá_m2": round(float(pm2_all.median()), 3),
        "Tỷ_lệ_Chung_cư_TM_%": round(float(df["is_commercial_project"].mean() * 100.0), 1),
    }])
    return pd.concat([res, overall], ignore_index=True)


def compute_segment_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thống kê chéo Tiểu khu vực x Phân khúc (Chung cư thương mại vs CHDV/Mini)
    nhằm bóc tách mặt bằng giá thực và kiểm chứng Nghịch lý Simpson.
    """
    work = df.copy()
    work["Phân_khúc"] = np.where(
        work["is_commercial_project"] == 1,
        "Chung cư thương mại",
        "CHDV / Mini / Khác",
    )

    sub_order = (
        work.groupby("sub_district")["price_million_vnd"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    rows = []
    for sub_name in sub_order:
        sub_df = work[work["sub_district"] == sub_name]
        sub_total = len(sub_df)
        for seg_name in ["Chung cư thương mại", "CHDV / Mini / Khác"]:
            grp = sub_df[sub_df["Phân_khúc"] == seg_name]
            if grp.empty:
                continue
            p = grp["price_million_vnd"].dropna()
            rows.append({
                "Tiểu_khu_vực": sub_name,
                "Phân_khúc": seg_name,
                "Count": int(len(grp)),
                "Tỷ_lệ_trong_khu_vực_%": round(len(grp) * 100.0 / max(sub_total, 1), 1),
                "Mean_Giá_thuê": round(float(p.mean()), 2),
                "Median_Giá_thuê": round(float(p.median()), 2),
                "Std_Giá_thuê": round(float(p.std(ddof=1)), 2) if len(p) > 1 else 0.0,
                "Mean_Diện_tích_m2": round(float(grp["area_m2"].mean()), 1),
                "Mean_Đơn_giá_m2": round(float(grp["price_per_m2"].mean()), 3),
            })

    return pd.DataFrame(rows)


def run_descriptive_pipeline(df: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Chạy toàn bộ Bước 1.1 và lưu 4 bảng CSV vào thư mục tables."""
    numeric_df = compute_numeric_overview(df)
    district_df = compute_spatial_stats(df, "district_std")
    subdistrict_df = compute_spatial_stats(df, "sub_district")
    crosstab_df = compute_segment_crosstab(df)

    numeric_df.to_csv(out_dir / "c1_numeric_summary.csv", index=False, encoding="utf-8-sig")
    district_df.to_csv(out_dir / "c1_price_by_district.csv", index=False, encoding="utf-8-sig")
    subdistrict_df.to_csv(out_dir / "c1_price_by_subdistrict.csv", index=False, encoding="utf-8-sig")
    crosstab_df.to_csv(out_dir / "c1_subdistrict_x_segment.csv", index=False, encoding="utf-8-sig")

    return {
        "numeric": numeric_df,
        "district": district_df,
        "subdistrict": subdistrict_df,
        "crosstab": crosstab_df,
    }
