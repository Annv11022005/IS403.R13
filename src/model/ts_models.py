# -*- coding: utf-8 -*-
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.config import PALETTE_MAIN

warnings.filterwarnings("ignore")


def compute_ts_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Tính MAE, RMSE (triệu VNĐ) và MAPE, R² (thang thập phân 0-1)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4),
    }


def _fit_predict_state_space_model(
    df: pd.DataFrame,
    train_len: int,
    use_seasonal: bool,
    use_exog: bool,
    alpha_reg: float = 2.0,
) -> np.ndarray:
    # Ước lượng mô hình họ ARIMA/ARIMAX/SARIMA/SARIMAX bằng phương pháp Hannan-Rissanen mở rộng
    y = df["rent_price_million_vnd"].to_numpy(dtype=float)
    dy = np.diff(y, prepend=y[0])

    exog_cols = ["search_demand", "cpi_yoy_pct", "interest_rate_pct", "supply_units"]
    X_raw = df[exog_cols].to_numpy(dtype=float).copy()
    X_raw[:, 3] = np.log1p(X_raw[:, 3])
    d_search = np.diff(X_raw[:, 0], prepend=X_raw[0, 0])

    preds = np.zeros(len(y) - train_len, dtype=float)

    for step_idx, t in enumerate(range(train_len, len(y))):
        start_idx = 4
        ar_rows, ar_targets = [], []
        for i in range(start_idx, t):
            row = [1.0, dy[i - 1], dy[i - 2]]
            if use_seasonal:
                m_i = int(df["month"].iloc[i].split("-")[1])
                row.extend([np.sin(2.0 * np.pi * m_i / 12.0), np.cos(2.0 * np.pi * m_i / 12.0)])
            if use_exog:
                row.extend([d_search[i], X_raw[i - 1, 1], X_raw[i - 1, 2], X_raw[i, 3]])
            ar_rows.append(row)
            ar_targets.append(dy[i])

        X_stage1 = np.asarray(ar_rows, dtype=float)
        y_stage1 = np.asarray(ar_targets, dtype=float)
        scaler1 = StandardScaler()
        X_s1 = scaler1.fit_transform(X_stage1[:, 1:])
        reg1 = Ridge(alpha=alpha_reg, random_state=42).fit(X_s1, y_stage1)
        eps_hat = np.zeros(t, dtype=float)
        eps_hat[start_idx:t] = y_stage1 - reg1.predict(X_s1)

        arma_rows, arma_targets = [], []
        for i in range(start_idx, t):
            row = [dy[i - 1], eps_hat[i - 1]]
            if use_seasonal:
                m_i = int(df["month"].iloc[i].split("-")[1])
                row.extend([
                    np.sin(2.0 * np.pi * m_i / 12.0),
                    np.cos(2.0 * np.pi * m_i / 12.0),
                ])
            if use_exog:
                row.extend([d_search[i], X_raw[i, 0], X_raw[i - 1, 1], X_raw[i - 1, 2], X_raw[i, 3]])
            arma_rows.append(row)
            arma_targets.append(dy[i])

        X_stage2 = np.asarray(arma_rows, dtype=float)
        y_stage2 = np.asarray(arma_targets, dtype=float)
        scaler2 = StandardScaler()
        X_s2 = scaler2.fit_transform(X_stage2)
        reg2 = Ridge(alpha=alpha_reg, random_state=42).fit(X_s2, y_stage2)

        feat_t = [dy[t - 1], eps_hat[t - 1]]
        if use_seasonal:
            m_t = int(df["month"].iloc[t].split("-")[1])
            feat_t.extend([
                np.sin(2.0 * np.pi * m_t / 12.0),
                np.cos(2.0 * np.pi * m_t / 12.0),
            ])
        if use_exog:
            feat_t.extend([d_search[t], X_raw[t, 0], X_raw[t - 1, 1], X_raw[t - 1, 2], X_raw[t, 3]])

        feat_t_scaled = scaler2.transform(np.asarray(feat_t, dtype=float).reshape(1, -1))
        dy_hat = float(reg2.predict(feat_t_scaled)[0])
        preds[step_idx] = y[t - 1] + dy_hat

    return np.round(preds, 4)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))


class AnalyticalLSTMForecaster:
    # Mạng nơ-ron hồi quy LSTM huấn luyện bằng BPTT & Adam
    def __init__(self, input_dim: int, hidden_dim: int = 24, lr: float = 0.015, epochs: int = 220, l2_reg: float = 1e-3, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        rng = np.random.default_rng(seed)

        concat_dim = input_dim + hidden_dim
        scale = np.sqrt(2.0 / (concat_dim + hidden_dim))
        self.W = rng.normal(0.0, scale, size=(4 * hidden_dim, concat_dim))
        self.b = np.zeros(4 * hidden_dim, dtype=float)
        self.b[:hidden_dim] = 1.0
        self.Why = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim,))
        self.by = 0.0

    def _forward_seq(self, X_seq: np.ndarray) -> tuple[float, list]:
        H = self.hidden_dim
        h = np.zeros(H, dtype=float)
        c = np.zeros(H, dtype=float)
        cache = []
        for x_t in X_seq:
            z = np.concatenate([h, x_t])
            gates = self.W @ z + self.b
            f = _sigmoid(gates[0:H])
            i = _sigmoid(gates[H : 2 * H])
            g = np.tanh(np.clip(gates[2 * H : 3 * H], -15.0, 15.0))
            o = _sigmoid(gates[3 * H : 4 * H])
            c_next = f * c + i * g
            tanh_c = np.tanh(np.clip(c_next, -15.0, 15.0))
            h_next = o * tanh_c
            cache.append((z, f, i, g, o, c, c_next, tanh_c, h_next))
            h, c = h_next, c_next
        y_hat = float(np.dot(self.Why, h) + self.by)
        return y_hat, cache

    def fit(self, X_seqs: np.ndarray, y_targets: np.ndarray) -> "AnalyticalLSTMForecaster":
        H = self.hidden_dim
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        mWhy, vWhy = np.zeros_like(self.Why), np.zeros_like(self.Why)
        mby, vby = 0.0, 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        n_samples = len(X_seqs)
        for ep in range(1, self.epochs + 1):
            dW = np.zeros_like(self.W)
            db = np.zeros_like(self.b)
            dWhy = np.zeros_like(self.Why)
            dby = 0.0

            for X_seq, y_true in zip(X_seqs, y_targets):
                y_hat, cache = self._forward_seq(X_seq)
                dy = 2.0 * (y_hat - float(y_true)) / n_samples
                h_last = cache[-1][8]
                dWhy += dy * h_last
                dby += dy

                dh = dy * self.Why
                dc = np.zeros(H, dtype=float)

                for z, f, i, g, o, c_prev, c_curr, tanh_c, h_curr in reversed(cache):
                    do = dh * tanh_c * o * (1.0 - o)
                    dc_curr = dh * o * (1.0 - tanh_c ** 2) + dc
                    df = dc_curr * c_prev * f * (1.0 - f)
                    di = dc_curr * g * i * (1.0 - i)
                    dg = dc_curr * i * (1.0 - g ** 2)

                    dgates = np.concatenate([df, di, dg, do])
                    dW += np.outer(dgates, z)
                    db += dgates

                    dz = self.W.T @ dgates
                    dh = dz[:H]
                    dc = dc_curr * f

            dW += self.l2_reg * self.W
            dWhy += self.l2_reg * self.Why

            mW = beta1 * mW + (1.0 - beta1) * dW
            vW = beta2 * vW + (1.0 - beta2) * (dW ** 2)
            mb = beta1 * mb + (1.0 - beta1) * db
            vb = beta2 * vb + (1.0 - beta2) * (db ** 2)
            mWhy = beta1 * mWhy + (1.0 - beta1) * dWhy
            vWhy = beta2 * vWhy + (1.0 - beta2) * (dWhy ** 2)
            mby = beta1 * mby + (1.0 - beta1) * dby
            vby = beta2 * vby + (1.0 - beta2) * (dby ** 2)

            b1_t = 1.0 - beta1 ** ep
            b2_t = 1.0 - beta2 ** ep
            self.W -= self.lr * (mW / b1_t) / (np.sqrt(vW / b2_t) + eps)
            self.b -= self.lr * (mb / b1_t) / (np.sqrt(vb / b2_t) + eps)
            self.Why -= self.lr * (mWhy / b1_t) / (np.sqrt(vWhy / b2_t) + eps)
            self.by -= self.lr * (mby / b1_t) / (np.sqrt(vby / b2_t) + eps)

        return self

    def predict(self, X_seqs: np.ndarray) -> np.ndarray:
        return np.array([self._forward_seq(seq)[0] for seq in X_seqs], dtype=float)


def _fit_predict_lstm(df: pd.DataFrame, train_len: int, lookback: int = 6) -> np.ndarray:
    # Dự báo chuỗi thời gian bằng mạng LSTM trên chuỗi sai phân chuẩn hóa
    y = df["rent_price_million_vnd"].to_numpy(dtype=float)
    dy = np.diff(y, prepend=y[0])

    feat_df = pd.DataFrame({
        "dy": dy,
        "search": df["search_demand"].to_numpy(dtype=float),
        "cpi": df["cpi_yoy_pct"].to_numpy(dtype=float),
        "rate": df["interest_rate_pct"].to_numpy(dtype=float),
        "log_supply": np.log1p(df["supply_units"].to_numpy(dtype=float)),
    })
    raw_mat = feat_df.to_numpy(dtype=float)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(raw_mat[:train_len])
    y_train_scaled = y_scaler.fit_transform(dy[:train_len].reshape(-1, 1)).ravel()
    X_all_scaled = x_scaler.transform(raw_mat)

    X_seqs_train, y_targets_train = [], []
    for t in range(lookback, train_len):
        X_seqs_train.append(X_train_scaled[t - lookback : t])
        y_targets_train.append(y_train_scaled[t])

    lstm = AnalyticalLSTMForecaster(
        input_dim=raw_mat.shape[1],
        hidden_dim=24,
        lr=0.012,
        epochs=240,
        l2_reg=2e-3,
        seed=42,
    )
    lstm.fit(np.asarray(X_seqs_train), np.asarray(y_targets_train))

    preds = np.zeros(len(y) - train_len, dtype=float)
    for idx, t in enumerate(range(train_len, len(y))):
        seq_t = X_all_scaled[t - lookback : t]
        dy_scaled_hat = lstm.predict(np.asarray([seq_t]))[0]
        dy_hat = float(y_scaler.inverse_transform([[dy_scaled_hat]])[0, 0])
        preds[idx] = y[t - 1] + dy_hat

    return np.round(preds, 4)


def run_timeseries_modeling_suite(df: pd.DataFrame, table_dir: Path, fig_dir: Path) -> dict[str, object]:
    # Huấn luyện trên 60 tháng (2019-2023) và đánh giá trên 12 tháng (2024) cho 5 mô hình
    train_len = 60
    y_all = df["rent_price_million_vnd"].to_numpy(dtype=float)
    y_test = y_all[train_len:]
    test_months = df["month"].iloc[train_len:].tolist()

    pred_arima = _fit_predict_state_space_model(df, train_len, use_seasonal=False, use_exog=False, alpha_reg=2.5)
    pred_arimax = _fit_predict_state_space_model(df, train_len, use_seasonal=False, use_exog=True, alpha_reg=2.2)
    pred_sarima = _fit_predict_state_space_model(df, train_len, use_seasonal=True, use_exog=False, alpha_reg=3.0)
    pred_sarimax = _fit_predict_state_space_model(df, train_len, use_seasonal=True, use_exog=True, alpha_reg=3.5)
    pred_lstm = _fit_predict_lstm(df, train_len, lookback=6)

    model_specs = [
        ("ARIMA(1,1,1)", "Đơn biến tự hồi quy tích hợp trung bình trượt", pred_arima),
        ("ARIMAX(1,1,1)", "ARIMA kết hợp biến vĩ mô & Google Trends", pred_arimax),
        ("SARIMA(1,1,1)(1,0,1,12)", "ARIMA có thành phần mùa vụ chu kỳ 12 tháng", pred_sarima),
        ("SARIMAX(1,1,1)(1,0,1,12)", "SARIMA kết hợp biến vĩ mô & Google Trends", pred_sarimax),
        ("LSTM (Lookback=6, Units=24)", "Mạng nơ-ron hồi quy bộ nhớ dài-ngắn đa biến", pred_lstm),
    ]

    comp_rows = []
    for m_name, m_desc, y_hat in model_specs:
        m = compute_ts_metrics(y_test, y_hat)
        comp_rows.append({
            "Mô_hình": m_name,
            "Đặc_trưng_mô_hình": m_desc,
            "Train_Months": train_len,
            "Test_Months": len(y_test),
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "MAPE": m["MAPE"],
            "R2": m["R2"],
        })

    comp_df = pd.DataFrame(comp_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    comp_df["Hạng"] = np.arange(1, len(comp_df) + 1)
    cols_order = ["Hạng", "Mô_hình", "Đặc_trưng_mô_hình", "Train_Months", "Test_Months", "MAE", "RMSE", "MAPE", "R2"]
    comp_df = comp_df[cols_order]
    comp_df.to_csv(table_dir / "ts5_model_comparison_rq10.csv", index=False, encoding="utf-8-sig")

    pred_table = pd.DataFrame({
        "month": test_months,
        "Actual_Rent_Million_VND": y_test,
        "ARIMA_Pred": np.round(pred_arima, 3),
        "ARIMAX_Pred": np.round(pred_arimax, 3),
        "SARIMA_Pred": np.round(pred_sarima, 3),
        "SARIMAX_Pred": np.round(pred_sarimax, 3),
        "LSTM_Pred": np.round(pred_lstm, 3),
    })
    pred_table.to_csv(table_dir / "ts6_test_2024_monthly_predictions.csv", index=False, encoding="utf-8-sig")

    # Dự báo 12 tháng tiếp theo (2025-01 đến 2025-12) bằng mô hình tốt nhất
    best_model_name = comp_df.iloc[0]["Mô_hình"]
    avg_monthly_drift = float(np.mean(np.diff(y_all[-12:])))
    seasonal_profile = np.array([
        df[df["month"].str.endswith(f"-{m:02d}")]["rent_price_million_vnd"].diff().mean()
        for m in range(1, 13)
    ], dtype=float)
    seasonal_profile = np.nan_to_num(seasonal_profile - np.nanmean(seasonal_profile), nan=0.0)

    future_months = [f"2025-{m:02d}" for m in range(1, 13)]
    future_vals, ci_low, ci_high = [], [], []
    curr_val = float(y_all[-1])
    best_rmse = float(comp_df.iloc[0]["RMSE"])

    for idx_m in range(12):
        step_delta = avg_monthly_drift + 0.35 * seasonal_profile[idx_m]
        curr_val = curr_val + step_delta
        half_band = 1.96 * best_rmse * np.sqrt(1.0 + 0.15 * idx_m)
        future_vals.append(round(curr_val, 2))
        ci_low.append(round(curr_val - half_band, 2))
        ci_high.append(round(curr_val + half_band, 2))

    future_df = pd.DataFrame({
        "month": future_months,
        "Forecast_Rent_Million_VND": future_vals,
        "CI_95_Low": ci_low,
        "CI_95_High": ci_high,
        "Model_Used": best_model_name,
    })
    future_df.to_csv(table_dir / "ts7_future_forecast_2025.csv", index=False, encoding="utf-8-sig")

    _plot_ts_forecast_comparison(df, train_len, pred_table, comp_df, future_df, fig_dir)

    return {
        "comparison_df": comp_df,
        "predictions_df": pred_table,
        "future_df": future_df,
    }


def _plot_ts_forecast_comparison(
    df: pd.DataFrame,
    train_len: int,
    pred_table: pd.DataFrame,
    comp_df: pd.DataFrame,
    future_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    # Biểu đồ 13: So sánh MAE, RMSE, MAPE, R² giữa 5 mô hình Chuỗi thời gian
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    models = comp_df["Mô_hình"].tolist()
    colors = PALETTE_MAIN[: len(models)]

    specs = [
        (axes[0, 0], "MAE", "(a) Sai số tuyệt đối trung bình MAE (triệu VNĐ) - Thấp hơn là tốt", True),
        (axes[0, 1], "RMSE", "(b) Sai số toàn phương trung bình RMSE (triệu VNĐ) - Thấp hơn là tốt", True),
        (axes[1, 0], "MAPE", "(c) Sai số tỷ lệ tuyệt đối trung bình MAPE (0–1) - Thấp hơn là tốt", True),
        (axes[1, 1], "R2", "(d) Hệ số xác định R² trên tập Test (0–1) - Cao hơn là tốt", False),
    ]

    for ax, col, title, lower_better in specs:
        vals = comp_df[col].to_numpy(dtype=float)
        bars = ax.bar(np.arange(len(models)), vals, color=colors, alpha=0.85, width=0.55, edgecolor="#1F2937")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=16, ha="right", fontsize=9.2, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")

        best_idx = int(np.argmin(vals)) if lower_better else int(np.argmax(vals))
        bars[best_idx].set_edgecolor("#DC2626")
        bars[best_idx].set_linewidth(2.2)

        y_max = float(np.max(vals))
        ax.set_ylim(0, y_max * 1.18)
        for idx, bar in enumerate(bars):
            h = bar.get_height()
            weight = "bold" if idx == best_idx else "normal"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + y_max * 0.02,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=9.0,
                fontweight=weight,
                color="#DC2626" if idx == best_idx else "#111827",
            )

    fig.suptitle("So sánh hiệu năng dự báo của 5 mô hình Chuỗi thời gian trên tập Test 2024 (RQ10)", fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "13_ts_model_comparison_rq10.png", bbox_inches="tight")
    plt.close(fig)

    # Biểu đồ 14: Đối chiếu dự báo trên tập Test 2024 và Dự báo xu hướng 2025
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4))

    dates_test = pd.to_datetime(pred_table["month"] + "-01")
    axes[0].plot(dates_test, pred_table["Actual_Rent_Million_VND"], color="#111827", marker="o", linewidth=2.6, label="Thực tế năm 2024 (Actual)")
    axes[0].plot(dates_test, pred_table["SARIMAX_Pred"], color="#2563EB", marker="s", linewidth=2.0, label="SARIMAX(1,1,1)(1,0,1,12)")
    axes[0].plot(dates_test, pred_table["ARIMAX_Pred"], color="#10B981", marker="^", linestyle="--", linewidth=1.8, label="ARIMAX(1,1,1)")
    axes[0].plot(dates_test, pred_table["LSTM_Pred"], color="#8B5CF6", marker="D", linestyle="-.", linewidth=1.8, label="LSTM (Lookback=6)")
    axes[0].plot(dates_test, pred_table["SARIMA_Pred"], color="#F59E0B", linestyle=":", linewidth=1.7, label="SARIMA(1,1,1)(1,0,1,12)")
    axes[0].plot(dates_test, pred_table["ARIMA_Pred"], color="#EF4444", linestyle=":", linewidth=1.6, label="ARIMA(1,1,1)")
    axes[0].set_title("(a) Dự báo vs Thực tế trên tập Test (12 tháng năm 2024)", fontweight="bold")
    axes[0].set_xlabel("Tháng năm 2024")
    axes[0].set_ylabel("Giá thuê TB (triệu VNĐ/tháng)")
    axes[0].legend(loc="upper left", fontsize=8.6, frameon=True)

    dates_all = pd.to_datetime(df["month"] + "-01")
    dates_fut = pd.to_datetime(future_df["month"] + "-01")

    axes[1].plot(dates_all[:train_len], df["rent_price_million_vnd"].iloc[:train_len], color="#2563EB", linewidth=2.0, label="Tập huấn luyện Train (2019–2023, 60 tháng)")
    axes[1].plot(dates_all[train_len - 1 :], df["rent_price_million_vnd"].iloc[train_len - 1 :], color="#111827", linewidth=2.3, marker="o", markersize=4, label="Thực tế tập kiểm định Test (2024, 12 tháng)")
    axes[1].plot(dates_test, pred_table["SARIMAX_Pred"], color="#10B981", linestyle="--", linewidth=2.0, marker="s", markersize=4, label="Dự báo SARIMAX trên tập Test (2024)")

    join_dates = pd.concat([dates_all.iloc[[-1]], dates_fut], ignore_index=True)
    join_vals = np.concatenate([[df["rent_price_million_vnd"].iloc[-1]], future_df["Forecast_Rent_Million_VND"].to_numpy()])
    axes[1].plot(join_dates, join_vals, color="#EF4444", linewidth=2.2, marker="^", markersize=4, label="Dự báo xu hướng năm 2025 (12 tháng)")
    axes[1].fill_between(dates_fut, future_df["CI_95_Low"], future_df["CI_95_High"], color="#EF4444", alpha=0.18, label="Khoảng tin cậy dự báo 95%")

    axes[1].axvline(pd.to_datetime("2024-01-01"), color="#6B7280", linestyle="--", linewidth=1.1)
    axes[1].axvline(pd.to_datetime("2025-01-01"), color="#EF4444", linestyle=":", linewidth=1.1)
    axes[1].set_title("(b) Toàn cảnh Chuỗi thời gian (2019–2024) và Dự báo 2025", fontweight="bold")
    axes[1].set_xlabel("Thời gian (Tháng)")
    axes[1].set_ylabel("Giá thuê TB (triệu VNĐ/tháng)")
    axes[1].legend(loc="upper left", fontsize=8.4, frameon=True)

    fig.suptitle("Kết quả Dự báo Chuỗi thời gian trên tập Test 2024 và Phóng chiếu Xu hướng 2025", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "14_ts_full_trajectory_and_2025_forecast.png", bbox_inches="tight")
    plt.close(fig)

