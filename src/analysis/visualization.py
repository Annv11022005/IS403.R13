# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.config import (
    COLOR_CHDV,
    COLOR_COMMERCIAL,
    FURNITURE_ORDER,
    PALETTE_MAIN,
)


def _styled_boxplot(
    ax,
    groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    title: str,
    ylabel: str = "Giá thuê (triệu VNĐ/tháng)",
    rotate: int = 0,
) -> None:
    bp = ax.boxplot(
        groups,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#111827",
            "markersize": 5.5,
        },
        medianprops={"color": "#111827", "linewidth": 1.8},
        flierprops={
            "marker": "o",
            "markersize": 2.8,
            "alpha": 0.32,
            "markeredgecolor": "#6B7280",
        },
        widths=0.5,
    )
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate >= 15 else "center")
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.74)
        patch.set_edgecolor("#1F2937")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)


def plot_target_distribution(df: pd.DataFrame, fig_dir: Path) -> None:
    # Hình 1: Phân bố giá thuê gốc, ln(giá) và theo phân khúc
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    prices = df["price_million_vnd"].dropna().values
    skew_raw = float(pd.Series(prices).skew())

    ax0 = axes[0]
    ax0.hist(prices, bins=34, density=True, color="#3B82F6", alpha=0.55, edgecolor="white")
    xg0 = np.linspace(prices.min(), prices.max(), 300)
    ax0.plot(xg0, stats.gaussian_kde(prices)(xg0), color="#1D4ED8", lw=2.1, label="KDE")
    ax0.axvline(prices.mean(), color="#DC2626", ls="--", lw=1.7, label=f"Mean = {prices.mean():.2f} tr")
    ax0.axvline(np.median(prices), color="#059669", ls="-.", lw=1.7, label=f"Median = {np.median(prices):.2f} tr")
    ax0.set_title(f"(a) Giá thuê gốc (Skew = {skew_raw:+.2f})", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Giá thuê (triệu VNĐ/tháng)")
    ax0.set_ylabel("Mật độ")
    ax0.legend(fontsize=8.5, frameon=True)

    log_p = np.log(np.clip(prices, 0.1, None))
    skew_log = float(pd.Series(log_p).skew())
    ax1 = axes[1]
    ax1.hist(log_p, bins=30, density=True, color="#10B981", alpha=0.55, edgecolor="white")
    xg1 = np.linspace(log_p.min(), log_p.max(), 300)
    ax1.plot(xg1, stats.gaussian_kde(log_p)(xg1), color="#047857", lw=2.1, label="KDE ln(giá)")
    ax1.axvline(log_p.mean(), color="#DC2626", ls="--", lw=1.7, label=f"Mean = {log_p.mean():.2f}")
    ax1.axvline(np.median(log_p), color="#111827", ls="-.", lw=1.7, label=f"Median = {np.median(log_p):.2f}")
    ax1.set_title(f"(b) Sau biến đổi ln(Giá) (Skew = {skew_log:+.2f})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("ln(Giá thuê)")
    ax1.set_ylabel("Mật độ")
    ax1.legend(fontsize=8.5, frameon=True)

    ax2 = axes[2]
    comm = df.loc[df["is_commercial_project"] == 1, "price_million_vnd"].dropna().values
    chdv = df.loc[df["is_commercial_project"] == 0, "price_million_vnd"].dropna().values
    bins = np.linspace(prices.min(), min(prices.max(), 45.0), 32)
    ax2.hist(chdv, bins=bins, density=True, alpha=0.5, color=COLOR_CHDV, edgecolor="white",
             label=f"CHDV/Mini (n={len(chdv)}, TB={chdv.mean():.1f} tr)")
    ax2.hist(comm, bins=bins, density=True, alpha=0.5, color=COLOR_COMMERCIAL, edgecolor="white",
             label=f"Chung cư TM (n={len(comm)}, TB={comm.mean():.1f} tr)")
    xg2 = np.linspace(bins[0], bins[-1], 250)
    if len(chdv) > 5:
        ax2.plot(xg2, stats.gaussian_kde(chdv)(xg2), color="#D97706", lw=2.0)
    if len(comm) > 5:
        ax2.plot(xg2, stats.gaussian_kde(comm)(xg2), color="#1D4ED8", lw=2.0)
    ax2.set_title("(c) Cấu trúc 2 phân khúc chồng lấp", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Giá thuê (triệu VNĐ/tháng)")
    ax2.set_ylabel("Mật độ")
    ax2.legend(fontsize=8.5, frameon=True)

    fig.suptitle("C.2. Phân tích phân bố biến mục tiêu và hiệu ứng phân khúc căn hộ", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_target_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_analysis(df: pd.DataFrame, fig_dir: Path) -> None:
    # Hình 2: So sánh giá thuê theo khu vực & phân khúc (RQ1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.2))

    dist_order = df.groupby("district_std")["price_million_vnd"].mean().sort_values(ascending=False).index.tolist()
    dist_groups = [df.loc[df["district_std"] == d, "price_million_vnd"].dropna().values for d in dist_order]
    dist_labels = [f"{d}\n(n={len(g)}, TB={g.mean():.1f})" for d, g in zip(dist_order, dist_groups)]
    _styled_boxplot(axes[0, 0], dist_groups, dist_labels, PALETTE_MAIN[:len(dist_order)],
                    "(a) Giá thuê theo 3 khu vực hành chính")

    sub_order = df.groupby("sub_district")["price_million_vnd"].mean().sort_values(ascending=False).index.tolist()
    sub_short = [s.replace(" (Thủ Đức)", "") for s in sub_order]
    sub_groups = [df.loc[df["sub_district"] == s, "price_million_vnd"].dropna().values for s in sub_order]
    sub_labels = [f"{sh}\n(n={len(g)}, TB={g.mean():.1f})" for sh, g in zip(sub_short, sub_groups)]
    _styled_boxplot(axes[0, 1], sub_groups, sub_labels, PALETTE_MAIN[:len(sub_order)],
                    "(b) Giá thuê theo 5 Tiểu khu vực (tách Q2, Q9, Thủ Đức cũ)")

    pm2_groups = [df.loc[df["sub_district"] == s, "price_per_m2"].dropna().values for s in sub_order]
    pm2_labels = [f"{sh}\n(TB={g.mean():.2f})" for sh, g in zip(sub_short, pm2_groups)]
    _styled_boxplot(axes[1, 0], pm2_groups, pm2_labels, PALETTE_MAIN[:len(sub_order)],
                    "(c) Đơn giá thuê trên mỗi m² theo 5 Tiểu khu vực",
                    ylabel="Đơn giá (triệu VNĐ/m²/tháng)")

    ax3 = axes[1, 1]
    x = np.arange(len(sub_order))
    width = 0.36
    comm_means = [
        df.loc[(df["sub_district"] == s) & (df["is_commercial_project"] == 1), "price_million_vnd"].mean()
        for s in sub_order
    ]
    chdv_means = [
        df.loc[(df["sub_district"] == s) & (df["is_commercial_project"] == 0), "price_million_vnd"].mean()
        for s in sub_order
    ]
    b1 = ax3.bar(x - width / 2, comm_means, width, color=COLOR_COMMERCIAL, alpha=0.85, label="Chung cư thương mại")
    b2 = ax3.bar(x + width / 2, chdv_means, width, color=COLOR_CHDV, alpha=0.85, label="CHDV / Mini / Khác")
    ax3.set_xticks(x)
    ax3.set_xticklabels(sub_short)
    ax3.set_ylabel("Giá thuê trung bình (triệu VNĐ/tháng)")
    ax3.set_title("(d) Giá trung bình tách theo Phân khúc tại từng Tiểu khu vực", fontsize=11, fontweight="bold")
    ax3.legend(frameon=True, fontsize=9)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.1f}",
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.suptitle("C.3. Phân tích sự khác biệt không gian về giá thuê căn hộ (Trả lời RQ1)", fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_spatial_boxplots_rq1.png", bbox_inches="tight")
    plt.close(fig)


def plot_furniture_and_amenities(df: pd.DataFrame, fig_dir: Path) -> None:
    # Hình 3: Boxplot theo Nội thất (RQ4) và Hồ bơi/Gym (RQ6)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    present_furn = [f for f in FURNITURE_ORDER if f in set(df["furniture"].dropna())]
    furn_groups = [df.loc[df["furniture"] == f, "price_million_vnd"].dropna().values for f in present_furn]
    furn_labels = [f"{f}\n(n={len(g)}, TB={g.mean():.1f})" for f, g in zip(present_furn, furn_groups)]
    _styled_boxplot(
        axes[0],
        furn_groups,
        furn_labels,
        ["#94A3B8", "#38BDF8", "#3B82F6", "#1D4ED8"][:len(present_furn)],
        "(a) Theo Tình trạng nội thất (RQ4)",
        rotate=16,
    )

    pool_g = [
        df.loc[df["has_pool"] == 0, "price_million_vnd"].dropna().values,
        df.loc[df["has_pool"] == 1, "price_million_vnd"].dropna().values,
    ]
    pool_lbl = [f"Không hồ bơi\n(n={len(pool_g[0])}, TB={pool_g[0].mean():.1f})",
                f"Có hồ bơi\n(n={len(pool_g[1])}, TB={pool_g[1].mean():.1f})"]
    _styled_boxplot(axes[1], pool_g, pool_lbl, ["#94A3B8", "#0EA5E9"], "(b) Theo Tiện ích Hồ bơi (RQ6)")

    gym_g = [
        df.loc[df["has_gym"] == 0, "price_million_vnd"].dropna().values,
        df.loc[df["has_gym"] == 1, "price_million_vnd"].dropna().values,
    ]
    gym_lbl = [f"Không Gym\n(n={len(gym_g[0])}, TB={gym_g[0].mean():.1f})",
               f"Có Gym\n(n={len(gym_g[1])}, TB={gym_g[1].mean():.1f})"]
    _styled_boxplot(axes[2], gym_g, gym_lbl, ["#94A3B8", "#10B981"], "(c) Theo Tiện ích Phòng Gym (RQ6)")

    fig.suptitle("Tác động của Nội thất và Tiện ích nội khu đến Giá thuê (RQ4 & RQ6)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_furniture_amenities_rq4_rq6.png", bbox_inches="tight")
    plt.close(fig)


def plot_area_and_distance_relations(df: pd.DataFrame, fig_dir: Path) -> None:
    # Hình 4: Scatter plot Diện tích (RQ2) và Khoảng cách (RQ5)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

    y = df["price_million_vnd"].values

    ax0 = axes[0]
    for flag, label, color in [
        (0, "CHDV / Mini / Khác", COLOR_CHDV),
        (1, "Chung cư thương mại", COLOR_COMMERCIAL),
    ]:
        sub = df[df["is_commercial_project"] == flag]
        ax0.scatter(sub["area_m2"], sub["price_million_vnd"], alpha=0.38, s=18, color=color, label=label)

    x_area = df["area_m2"].values
    sl1, ic1, r1, _, _ = stats.linregress(x_area, y)
    xg1 = np.linspace(x_area.min(), x_area.max(), 200)
    ax0.plot(xg1, ic1 + sl1 * xg1, color="#DC2626", lw=2.2,
             label=f"Tuyến tính: y = {sl1:.2f}x + {ic1:.2f} (r = {r1:.2f})")
    ax0.set_title("(a) Diện tích căn hộ vs Giá thuê (RQ2)", fontsize=11.5, fontweight="bold")
    ax0.set_xlabel("Diện tích (m²)")
    ax0.set_ylabel("Giá thuê (triệu VNĐ/tháng)")
    ax0.legend(frameon=True, fontsize=8.8)

    ax1 = axes[1]
    sub_order = df.groupby("sub_district")["price_million_vnd"].mean().sort_values(ascending=False).index.tolist()
    for s_name, color in zip(sub_order, PALETTE_MAIN):
        sub = df[df["sub_district"] == s_name]
        ax1.scatter(sub["distance_to_center_km"], sub["price_million_vnd"], alpha=0.4, s=18,
                    color=color, label=s_name.replace(" (Thủ Đức)", ""))

    x_dist = df["distance_to_center_km"].values
    sl2, ic2, r2, _, _ = stats.linregress(x_dist, y)
    xg2 = np.linspace(x_dist.min(), x_dist.max(), 200)
    ax1.plot(xg2, ic2 + sl2 * xg2, color="#DC2626", ls="--", lw=2.0,
             label=f"Tuyến tính (r = {r2:.2f})")

    poly_coef = np.polyfit(x_dist, y, deg=2)
    ax1.plot(xg2, np.polyval(poly_coef, xg2), color="#111827", lw=2.2,
             label="Xu hướng phi tuyến bậc 2")

    ax1.set_title("(b) Khoảng cách tới Trung tâm Q1 vs Giá thuê (RQ5)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Khoảng cách chim bay đến chợ Bến Thành (km)")
    ax1.set_ylabel("Giá thuê (triệu VNĐ/tháng)")
    ax1.legend(frameon=True, fontsize=8.2, ncol=2)

    fig.suptitle("C.4. Quan hệ giữa Quy mô diện tích, Khoảng cách vị trí và Giá thuê", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_area_distance_scatter_rq2_rq5.png", bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrices(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Hình 5: Ma trận tương quan Pearson và Spearman
    cols = [
        "price_million_vnd",
        "price_per_m2",
        "area_m2",
        "bedrooms",
        "bathrooms",
        "furniture_score",
        "distance_to_center_km",
        "has_pool",
        "has_gym",
        "is_commercial_project",
    ]
    labels = [
        "Giá thuê (triệu)",
        "Đơn giá/m²",
        "Diện tích (m²)",
        "Số phòng ngủ",
        "Số phòng WC",
        "Điểm nội thất",
        "Khoảng cách Q1",
        "Có hồ bơi",
        "Có phòng Gym",
        "Chung cư TM",
    ]

    pearson_df = df[cols].corr(method="pearson")
    spearman_df = df[cols].corr(method="spearman")

    pearson_df.round(4).to_csv(table_dir / "c4_correlation_pearson.csv", encoding="utf-8-sig")
    spearman_df.round(4).to_csv(table_dir / "c4_correlation_spearman.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2))
    for ax, mat, title in [
        (axes[0], pearson_df.values, "(a) Ma trận tương quan tuyến tính Pearson (r)"),
        (axes[1], spearman_df.values, "(b) Ma trận tương quan hạng Spearman (ρ)"),
    ]:
        ax.grid(False)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.2)
        ax.set_yticklabels(labels, fontsize=9.2)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = mat[i, j]
                color = "white" if abs(val) >= 0.55 else "#111827"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8.4, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8.5)

    fig.suptitle("C.4. Phân tích tương quan đa biến và phát hiện sớm đa cộng tuyến", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "05_correlation_matrices_c4.png", bbox_inches="tight")
    plt.close(fig)

    return pearson_df, spearman_df


def run_visualization_pipeline(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_target_distribution(df, fig_dir)
    plot_spatial_analysis(df, fig_dir)
    plot_furniture_and_amenities(df, fig_dir)
    plot_area_and_distance_relations(df, fig_dir)
    return plot_correlation_matrices(df, fig_dir, table_dir)

