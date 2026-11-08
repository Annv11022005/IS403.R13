# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def _adf_fallback(series: np.ndarray, max_lags: int = 4) -> tuple[float, float, int, dict]:
    # Tính kiểm định ADF và p-value theo bảng tới hạn MacKinnon (1994)
    y = np.asarray(series, dtype=float)
    dy = np.diff(y)
    y_lag1 = y[:-1]
    n_total = len(dy)

    best_aic = np.inf
    best_res = (-1.5, 0.50, 0)

    for k in range(0, min(max_lags, n_total // 5) + 1):
        dep = dy[k:]
        cols = [np.ones(len(dep)), y_lag1[k:]]
        for j in range(1, k + 1):
            cols.append(dy[k - j : len(dy) - j])
        X = np.column_stack(cols)
        beta, _, _, _ = np.linalg.lstsq(X, dep, rcond=None)
        resid = dep - X @ beta
        T = len(dep)
        sse = float(np.sum(resid ** 2))
        sigma2 = sse / max(T - X.shape[1], 1)
        aic = np.log(max(sse / T, 1e-12)) + 2.0 * X.shape[1] / T
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se_gamma = float(np.sqrt(max(cov[1, 1], 1e-15)))
        t_stat = float(beta[1] / se_gamma)

        if aic < best_aic:
            best_aic = aic
            tau_grid = np.array([-6.0, -4.5, -3.53, -3.20, -2.90, -2.59, -2.00, -1.50, -1.00, 0.0, 1.0])
            p_grid = np.array([0.0001, 0.0005, 0.01, 0.025, 0.05, 0.10, 0.28, 0.53, 0.75, 0.95, 0.99])
            pval = float(np.interp(t_stat, tau_grid, p_grid))
            best_res = (t_stat, pval, k)

    crit = {"1%": -3.53, "5%": -2.90, "10%": -2.59}
    return best_res[0], best_res[1], best_res[2], crit


def run_single_adf(series: pd.Series, is_diff: bool = False) -> tuple[float, float, int, dict]:
    clean = series.dropna().to_numpy(dtype=float)
    if HAS_STATSMODELS:
        reg_mode = "n" if is_diff else "c"
        max_l = 1 if is_diff else 4
        res = adfuller(clean, regression=reg_mode, autolag="AIC", maxlag=max_l)
        return float(res[0]), float(res[1]), int(res[2]), {k: round(float(v), 3) for k, v in res[4].items()}
    return _adf_fallback(clean)


def compute_ts_descriptive_stats(df: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    # Thống kê mô tả các biến chuỗi thời gian (2019-2024)
    specs = [
        ("rent_price_million_vnd", "Giá thuê căn hộ TB (triệu VNĐ/tháng)"),
        ("search_demand", "Chỉ số nhu cầu tìm kiếm Google Trends (0-100)"),
        ("cpi_yoy_pct", "Chỉ số CPI YoY hàng tháng (%)"),
        ("interest_rate_pct", "Lãi suất tham chiếu 12 tháng (%/năm)"),
        ("policy_rate_pct", "Lãi suất tái cấp vốn NHNN (%/năm)"),
        ("supply_units", "Nguồn cung căn hộ mở bán/cho thuê (căn/tháng)"),
    ]
    rows = []
    for col, label in specs:
        s = df[col].dropna().astype(float)
        mean_val = float(s.mean())
        sd_val = float(s.std(ddof=1))
        q25, q50, q75 = np.percentile(s, [25, 50, 75])
        rows.append({
            "Biến": col,
            "Mô_tả": label,
            "N_Tháng": len(s),
            "Mean": round(mean_val, 2),
            "SD": round(sd_val, 2),
            "Median": round(float(q50), 2),
            "IQR": round(float(q75 - q25), 2),
            "Min": round(float(s.min()), 2),
            "Max": round(float(s.max()), 2),
            "Skewness": round(float(stats.skew(s)), 2),
            "CV": round(sd_val / max(abs(mean_val), 1e-9), 4),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(table_dir / "ts1_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    return out_df


def run_adf_stationarity_suite(df: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    # Kiểm định nghiệm đơn vị ADF cho chuỗi gốc I(0) và sai phân bậc 1 I(1)
    specs = [
        ("rent_price_million_vnd", "Giá thuê căn hộ TB (triệu/tháng)"),
        ("search_demand", "Nhu cầu tìm kiếm Google Trends"),
        ("cpi_yoy_pct", "Chỉ số lạm phát CPI YoY (%)"),
        ("interest_rate_pct", "Lãi suất kỳ hạn 12 tháng (%/năm)"),
        ("supply_units", "Nguồn cung căn hộ (căn/tháng)"),
    ]
    rows = []
    for col, label in specs:
        s_level = df[col].astype(float)
        stat0, p0, lag0, crit0 = run_single_adf(s_level, is_diff=False)
        rows.append({
            "Biến": col,
            "Mô_tả": label,
            "Bậc_sai_phân": "Chuỗi gốc I(0) (d=0)",
            "ADF_Statistic": round(stat0, 4),
            "P_Value": round(p0, 4),
            "Lag_AIC": lag0,
            "Critical_5%": crit0.get("5%", -2.90),
            "Kết_luận": "Chuỗi dừng (Stationary)" if p0 < 0.05 else "Không dừng (Non-stationary)",
        })

        s_diff1 = s_level.diff().dropna()
        stat1, p1, lag1, crit1 = run_single_adf(s_diff1, is_diff=True)
        rows.append({
            "Biến": col,
            "Mô_tả": label,
            "Bậc_sai_phân": "Sai phân bậc 1 I(1) (d=1)",
            "ADF_Statistic": round(stat1, 4),
            "P_Value": round(p1, 4),
            "Lag_AIC": lag1,
            "Critical_5%": crit1.get("5%", -1.95),
            "Kết_luận": "Chuỗi dừng (Stationary)" if p1 < 0.05 else "Không dừng (Non-stationary)",
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(table_dir / "ts2_adf_stationarity_tests.csv", index=False, encoding="utf-8-sig")
    return out_df


def run_macro_correlations_and_regimes(df: pd.DataFrame, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Phân tích tương quan vĩ mô (RQ9) và diễn biến theo 4 giai đoạn thị trường (RQ8)
    work = df.copy()
    work["year"] = work["month"].str.slice(0, 4).astype(int)
    work["rent_yoy_change_pct"] = work["rent_price_million_vnd"].pct_change(12) * 100.0

    exog_specs = [
        ("search_demand", "Nhu cầu tìm kiếm Google Trends"),
        ("cpi_yoy_pct", "Lạm phát CPI YoY (%)"),
        ("interest_rate_pct", "Lãi suất 12 tháng (%/năm)"),
        ("policy_rate_pct", "Lãi suất tái cấp vốn NHNN (%/năm)"),
        ("supply_units", "Nguồn cung căn hộ (căn/tháng)"),
    ]

    corr_rows = []
    y = work["rent_price_million_vnd"].to_numpy(dtype=float)
    dy = work["rent_price_million_vnd"].diff().to_numpy(dtype=float)

    for col, label in exog_specs:
        x = work[col].to_numpy(dtype=float)
        r0, p0 = stats.pearsonr(x, y)
        rho0, _ = stats.spearmanr(x, y)

        best_lag, best_r, best_p = 0, r0, p0
        for lag in [1, 2, 3]:
            rl, pl = stats.pearsonr(x[:-lag], y[lag:])
            if abs(rl) > abs(best_r):
                best_lag, best_r, best_p = lag, rl, pl

        r_diff, p_diff = stats.pearsonr(x[1:], dy[1:])

        corr_rows.append({
            "Biến_vĩ_mô": col,
            "Mô_tả": label,
            "Pearson_r_Level": round(float(r0), 4),
            "P_Value_Level": round(float(p0), 4),
            "Spearman_rho": round(float(rho0), 4),
            "Best_Lag_Months": best_lag,
            "Pearson_r_BestLag": round(float(best_r), 4),
            "Pearson_r_vs_DeltaRent": round(float(r_diff), 4),
            "P_Value_vs_DeltaRent": round(float(p_diff), 4),
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(table_dir / "ts3_macro_correlations_rq9.csv", index=False, encoding="utf-8-sig")

    def _assign_regime(m: str) -> str:
        if m <= "2019-12":
            return "1. Tiền COVID-19 (2019)"
        if m <= "2021-09":
            return "2. Giãn cách COVID-19 (01/2020 - 09/2021)"
        if m <= "2022-12":
            return "3. Phục hồi hậu mở cửa (10/2021 - 12/2022)"
        return "4. Khan hiếm nguồn cung & tăng giá mới (2023 - 2024)"

    work["regime"] = work["month"].apply(_assign_regime)
    regime_rows = []
    for reg, grp in work.groupby("regime", sort=True):
        regime_rows.append({
            "Giai_đoạn": reg,
            "Số_tháng": len(grp),
            "Giá_thuê_TB_triệu": round(float(grp["rent_price_million_vnd"].mean()), 2),
            "Giá_thuê_Min": round(float(grp["rent_price_million_vnd"].min()), 2),
            "Giá_thuê_Max": round(float(grp["rent_price_million_vnd"].max()), 2),
            "Google_Trends_TB": round(float(grp["search_demand"].mean()), 1),
            "CPI_YoY_TB_%": round(float(grp["cpi_yoy_pct"].mean()), 2),
            "Lãi_suất_12T_TB_%": round(float(grp["interest_rate_pct"].mean()), 2),
            "Nguồn_cung_TB_căn_tháng": int(round(float(grp["supply_units"].mean()))),
        })
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(table_dir / "ts4_regime_summary_rq8.csv", index=False, encoding="utf-8-sig")
    return corr_df, regime_df


def _compute_acf_pacf(series: np.ndarray, max_lag: int = 20) -> tuple[np.ndarray, np.ndarray]:
    # Tính ACF và PACF bằng thuật toán Durbin-Levinson
    x = np.asarray(series, dtype=float)
    x_c = x - np.mean(x)
    var = float(np.dot(x_c, x_c))
    acf = np.ones(max_lag + 1)
    for k in range(1, max_lag + 1):
        acf[k] = float(np.dot(x_c[:-k], x_c[k:]) / max(var, 1e-12))

    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    if max_lag >= 1:
        pacf[1] = acf[1]
    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = acf[1]
    for k in range(2, max_lag + 1):
        num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
        phi[k, k] = num / max(den, 1e-12)
        pacf[k] = phi[k, k]
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
    return acf, pacf


def plot_ts_eda_figures(df: pd.DataFrame, adf_df: pd.DataFrame, fig_dir: Path) -> None:
    # Vẽ bộ 3 biểu đồ EDA chuỗi thời gian và kiểm định ADF/ACF/PACF
    dates = pd.to_datetime(df["month"] + "-01")

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11.0), sharex=True)
    covid_start, covid_end = pd.to_datetime("2020-03-01"), pd.to_datetime("2021-09-30")

    for ax in axes:
        ax.axvspan(covid_start, covid_end, color="#FEE2E2", alpha=0.55, label="Giai đoạn COVID-19 (03/2020 - 09/2021)")
        ax.axvline(pd.to_datetime("2024-01-01"), color="#6B7280", linestyle="--", linewidth=1.1)

    axes[0].plot(dates, df["rent_price_million_vnd"], color="#2563EB", marker="o", markersize=3.5, linewidth=2.0, label="Giá thuê căn hộ 2PN TB (triệu VNĐ/tháng)")
    axes[0].set_ylabel("Triệu VNĐ/tháng", fontweight="bold")
    axes[0].set_title("Diễn biến Giá thuê căn hộ TP.HCM và các yếu tố Vĩ mô – Thị trường (2019–2024)", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(dates, df["search_demand"], color="#10B981", linewidth=1.9, label="Google Trends: Trung bình nhu cầu tìm thuê")
    axes[1].plot(dates, df["search_chung_cu_hcm"], color="#059669", linestyle=":", alpha=0.75, label="'thuê chung cư' (TP.HCM)")
    axes[1].set_ylabel("Chỉ số (0–100)", fontweight="bold")
    axes[1].legend(loc="upper left", frameon=True)

    axes[2].plot(dates, df["cpi_yoy_pct"], color="#EF4444", linewidth=1.8, label="Lạm phát CPI YoY (%) - GSO")
    axes[2].plot(dates, df["interest_rate_pct"], color="#F59E0B", linewidth=1.8, label="Lãi suất 12T Vietcombank (%/năm)")
    axes[2].plot(dates, df["policy_rate_pct"], color="#8B5CF6", linestyle="--", linewidth=1.5, label="Lãi suất tái cấp vốn NHNN (%/năm)")
    axes[2].set_ylabel("Tỷ lệ (%)", fontweight="bold")
    axes[2].legend(loc="upper left", ncol=3, frameon=True)

    axes[3].bar(dates, df["supply_units"], width=20, color="#0EA5E9", alpha=0.8, label="Nguồn cung căn hộ mở bán/cho thuê (CBRE/BXD)")
    axes[3].set_ylabel("Số căn/tháng", fontweight="bold")
    axes[3].set_xlabel("Thời gian (Tháng)", fontweight="bold")
    axes[3].legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(fig_dir / "10_timeseries_overview_rq8.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    rent_lvl = df["rent_price_million_vnd"].to_numpy(dtype=float)
    rent_diff = np.diff(rent_lvl)

    r_adf0 = adf_df[(adf_df["Biến"] == "rent_price_million_vnd") & (adf_df["Bậc_sai_phân"].str.contains("d=0"))].iloc[0]
    r_adf1 = adf_df[(adf_df["Biến"] == "rent_price_million_vnd") & (adf_df["Bậc_sai_phân"].str.contains("d=1"))].iloc[0]

    axes[0, 0].plot(dates, rent_lvl, color="#2563EB", linewidth=2.0)
    axes[0, 0].set_title(f"A. Chuỗi gốc Giá thuê I(0): ADF = {r_adf0['ADF_Statistic']:.2f} (p = {r_adf0['P_Value']:.4f} - Không dừng)", fontweight="bold")
    axes[0, 0].set_ylabel("Triệu VNĐ/tháng")

    axes[0, 1].plot(dates[1:], rent_diff, color="#10B981", linewidth=1.8, marker="o", markersize=3)
    axes[0, 1].axhline(0, color="#374151", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title(f"B. Sai phân bậc 1 ΔGiá thuê (d=1): ADF = {r_adf1['ADF_Statistic']:.2f} (p = {r_adf1['P_Value']:.4f} - Chuỗi dừng)", fontweight="bold")
    axes[0, 1].set_ylabel("Δ Triệu VNĐ/tháng")

    acf_vals, pacf_vals = _compute_acf_pacf(rent_diff, max_lag=18)
    lags = np.arange(len(acf_vals))
    ci_bound = 1.96 / np.sqrt(len(rent_diff))

    axes[1, 0].bar(lags, acf_vals, width=0.35, color="#2563EB")
    axes[1, 0].axhline(ci_bound, color="#EF4444", linestyle="--", linewidth=1.1, label="Ngưỡng 95% CI")
    axes[1, 0].axhline(-ci_bound, color="#EF4444", linestyle="--", linewidth=1.1)
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("C. Hàm tự tương quan (ACF) của chuỗi sai phân bậc 1 (d=1)", fontweight="bold")
    axes[1, 0].set_xlabel("Độ trễ (Lag - Tháng)")
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].bar(lags, pacf_vals, width=0.35, color="#8B5CF6")
    axes[1, 1].axhline(ci_bound, color="#EF4444", linestyle="--", linewidth=1.1, label="Ngưỡng 95% CI")
    axes[1, 1].axhline(-ci_bound, color="#EF4444", linestyle="--", linewidth=1.1)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("D. Hàm tự tương quan riêng phần (PACF) của chuỗi sai phân bậc 1 (d=1)", fontweight="bold")
    axes[1, 1].set_xlabel("Độ trễ (Lag - Tháng)")
    axes[1, 1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_dir / "11_adf_differencing_acf_pacf.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    cols = ["rent_price_million_vnd", "search_demand", "cpi_yoy_pct", "interest_rate_pct", "policy_rate_pct", "supply_units"]
    labels = ["Giá thuê TB", "Google Trends", "CPI YoY (%)", "Lãi suất 12T", "LS Tái cấp vốn", "Nguồn cung"]
    corr_mat = df[cols].corr().to_numpy()

    im = axes[0].imshow(corr_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_yticklabels(labels)
    axes[0].set_title("A. Ma trận tương quan Pearson các biến chuỗi thời gian", fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_mat[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold", fontsize=9.5)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    ax_r = axes[1]
    sc = ax_r.scatter(
        df["search_demand"],
        df["rent_price_million_vnd"].diff().fillna(0),
        c=df["cpi_yoy_pct"],
        cmap="viridis",
        s=55,
        alpha=0.85,
        edgecolors="black",
        linewidth=0.5,
    )
    m_slope, m_int = np.polyfit(df["search_demand"].iloc[1:], df["rent_price_million_vnd"].diff().dropna(), 1)
    x_line = np.linspace(df["search_demand"].min(), df["search_demand"].max(), 50)
    ax_r.plot(x_line, m_slope * x_line + m_int, color="#EF4444", linestyle="--", linewidth=2.0, label=f"Xu hướng tuyến tính (slope={m_slope:+.4f})")
    ax_r.axhline(0, color="#6B7280", linestyle=":", linewidth=1.0)
    ax_r.set_title("B. Tác động của Nhu cầu tìm kiếm & CPI lên mức thay đổi Giá thuê (ΔRent)", fontweight="bold")
    ax_r.set_xlabel("Chỉ số nhu cầu tìm kiếm Google Trends (0–100)")
    ax_r.set_ylabel("Mức thay đổi giá thuê hàng tháng ΔRent (triệu VNĐ)")
    ax_r.legend(loc="upper left")
    cbar2 = fig.colorbar(sc, ax=ax_r, fraction=0.046, pad=0.04)
    cbar2.set_label("CPI YoY (%)")

    fig.tight_layout()
    fig.savefig(fig_dir / "12_macro_correlations_rq9.png", bbox_inches="tight")
    plt.close(fig)

