# -*- coding: utf-8 -*-
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET1_CSV = PROCESSED_DATA_DIR / "hcm_apartment_rentals.csv"
DATASET2_CSV = PROCESSED_DATA_DIR / "hcm_rent_timeseries.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PHASE1_OUT_DIR = OUTPUTS_DIR / "dataset1_phase1"
PHASE1_TABLES_DIR = PHASE1_OUT_DIR / "tables"
PHASE1_FIGURES_DIR = PHASE1_OUT_DIR / "figures"
PHASE1_REPORT_TXT = PHASE1_OUT_DIR / "phase1_report.txt"

MODELING_OUT_DIR = OUTPUTS_DIR / "dataset1_modeling"
MODELING_TABLES_DIR = MODELING_OUT_DIR / "tables"
MODELING_FIGURES_DIR = MODELING_OUT_DIR / "figures"
MODELING_REPORT_TXT = MODELING_OUT_DIR / "modeling_report.txt"

TS_OUT_DIR = OUTPUTS_DIR / "dataset2_timeseries"
TS_TABLES_DIR = TS_OUT_DIR / "tables"
TS_FIGURES_DIR = TS_OUT_DIR / "figures"
TS_REPORT_TXT = TS_OUT_DIR / "timeseries_report.txt"

FURNITURE_ORDER = [
    "Bàn giao thô",
    "Hoàn thiện cơ bản",
    "Nội thất đầy đủ",
    "Nội thất cao cấp",
]

NUMERIC_FEATURES_META = [
    ("price_million_vnd", "Giá thuê (triệu VNĐ/tháng)"),
    ("price_per_m2", "Đơn giá thuê (triệu VNĐ/m²)"),
    ("area_m2", "Diện tích (m²)"),
    ("bedrooms", "Số phòng ngủ"),
    ("bathrooms", "Số phòng vệ sinh (WC)"),
    ("furniture_score", "Điểm nội thất (0–3)"),
    ("distance_to_center_km", "Khoảng cách tới Q1 (km)"),
    ("has_pool", "Có hồ bơi (0/1)"),
    ("has_gym", "Có phòng Gym (0/1)"),
    ("is_commercial_project", "Chung cư thương mại (0/1)"),
]

PALETTE_MAIN = ["#2563EB", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
COLOR_COMMERCIAL = "#2563EB"
COLOR_CHDV = "#F59E0B"


def ensure_phase1_dirs() -> None:
    PHASE1_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PHASE1_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_modeling_dirs() -> None:
    MODELING_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELING_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_ts_dirs() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def setup_matplotlib_style() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        print("[LỖI] Chưa cài thư viện matplotlib. Vui lòng chạy: pip install matplotlib scipy")
        sys.exit(1)

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "--"


