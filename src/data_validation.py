# -*- coding: utf-8 -*-
"""Kiểm tra dữ liệu đầu vào; không chỉnh sửa hay tạo dữ liệu."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATASET1_CSV, DATASET2_CSV

APARTMENT_COLUMNS = {
    "listing_id", "district_std", "sub_district", "ward", "project_name", "apartment_type",
    "is_commercial_project", "price_million_vnd", "price_per_m2", "area_m2", "bedrooms",
    "bathrooms", "furniture", "furniture_score", "distance_to_center_km", "has_pool", "has_gym",
    "lat", "lon", "post_time", "url",
}
TS_COLUMNS = {"month", "rent_price_million_vnd", "search_demand", "search_can_ho_hcm",
              "search_chung_cu_hcm", "cpi_yoy_pct", "interest_rate_pct", "policy_rate_pct", "supply_units"}


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: thiếu cột {sorted(missing)}")


def _numeric(df: pd.DataFrame, columns: list[str], name: str) -> None:
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: cột {col} có giá trị thiếu hoặc không phải số hữu hạn")


def validate_apartments(df: pd.DataFrame) -> None:
    _require_columns(df, APARTMENT_COLUMNS, "Căn hộ")
    if df.empty:
        raise ValueError("Căn hộ: dữ liệu rỗng")
    required = APARTMENT_COLUMNS - {"project_name"}
    if df[list(required)].isna().any().any():
        raise ValueError("Căn hộ: có giá trị thiếu ở cột bắt buộc")
    if df["listing_id"].duplicated().any():
        raise ValueError("Căn hộ: listing_id bị trùng")
    numeric = ["price_million_vnd", "price_per_m2", "area_m2", "bedrooms", "bathrooms", "furniture_score",
               "distance_to_center_km", "is_commercial_project", "has_pool", "has_gym", "lat", "lon"]
    _numeric(df, numeric, "Căn hộ")
    for col in ("price_million_vnd", "price_per_m2", "area_m2"):
        if (pd.to_numeric(df[col]) <= 0).any():
            raise ValueError(f"Căn hộ: {col} phải lớn hơn 0")
    for col in ("is_commercial_project", "has_pool", "has_gym"):
        if not pd.to_numeric(df[col]).isin([0, 1]).all():
            raise ValueError(f"Căn hộ: {col} phải là 0 hoặc 1")
    if pd.to_datetime(df["post_time"], errors="coerce").isna().any():
        raise ValueError("Căn hộ: post_time không hợp lệ")


def validate_timeseries(df: pd.DataFrame) -> None:
    _require_columns(df, TS_COLUMNS, "Chuỗi thời gian")
    if df.empty or df[list(TS_COLUMNS)].isna().any().any():
        raise ValueError("Chuỗi thời gian: dữ liệu rỗng hoặc có giá trị thiếu")
    months = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    if months.isna().any() or not df["month"].astype(str).str.fullmatch(r"\d{4}-\d{2}").all():
        raise ValueError("Chuỗi thời gian: tháng phải có dạng YYYY-MM")
    periods = pd.PeriodIndex(months, freq="M")
    if not periods.equals(pd.period_range(periods[0], periods[-1], freq="M")):
        raise ValueError("Chuỗi thời gian: tháng phải liên tục, không trùng và tăng dần")
    _numeric(df, sorted(TS_COLUMNS - {"month"}), "Chuỗi thời gian")
    if (pd.to_numeric(df["rent_price_million_vnd"]) <= 0).any() or (pd.to_numeric(df["supply_units"]) < 0).any():
        raise ValueError("Chuỗi thời gian: giá thuê phải dương, nguồn cung không âm")


def load_validated_data(apartments_path: Path = DATASET1_CSV, timeseries_path: Path = DATASET2_CSV):
    apartments = pd.read_csv(apartments_path, encoding="utf-8-sig")
    timeseries = pd.read_csv(timeseries_path, encoding="utf-8-sig")
    validate_apartments(apartments)
    validate_timeseries(timeseries)
    return apartments, timeseries


if __name__ == "__main__":
    apartments, timeseries = load_validated_data()
    print(f"Căn hộ: {len(apartments)} dòng; chuỗi thời gian: {len(timeseries)} tháng; kiểm tra thành công.")
