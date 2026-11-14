# -*- coding: utf-8 -*-
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.config import PALETTE_MAIN



def compute_ts_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # MAE/RMSE: triệu VNĐ/tháng; MAPE: tỷ lệ; R² có thể âm.
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


def _fit_predict_lstm(df: pd.DataFrame, train_len: int, lookback: int = 6, recursive: bool = False) -> np.ndarray:
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
        if recursive:
            y[t] = preds[idx]
            raw_mat[t, 0] = dy_hat
            X_all_scaled[t] = x_scaler.transform(raw_mat[t:t + 1])[0]

    return np.round(preds, 4)


# Các biến ngoại sinh được trễ một tháng: giả định số liệu tháng trước đã công bố
# trước khi dự báo tháng hiện tại; cần kiểm tra lịch công bố thực tế của từng nguồn.
EXOG_COLS = ["search_demand", "cpi_yoy_pct", "interest_rate_pct", "supply_units"]
FIT_WARNINGS = []
TS_MODELS = {
    "ARIMA(1,1,1)": (False, False),
    "ARIMAX(1,1,1)": (False, True),
    "SARIMA(1,1,1)(1,0,1,12)": (True, False),
    "SARIMAX(1,1,1)(1,0,1,12)": (True, True),
}


def _sarimax_forecast(history: pd.DataFrame, steps: int, seasonal: bool, exogenous: bool) -> np.ndarray:
    # Hàng t sử dụng biến ngoại sinh t-1. Với dự báo nhiều bước, giữ nguyên
    # các biến từ tháng cuối lịch sử, một kịch bản chứ không phải số liệu thực tế.
    exog = history[EXOG_COLS].shift(1).iloc[1:].astype(float) if exogenous else None
    y = history["rent_price_million_vnd"].iloc[1:] if exogenous else history["rent_price_million_vnd"]
    future_exog = np.repeat(history[EXOG_COLS].iloc[[-1]].to_numpy(dtype=float), steps, axis=0) if exogenous else None
    if exogenous:
        scaler = StandardScaler().fit(exog)
        future_exog = scaler.transform(future_exog)
        exog = scaler.transform(exog)
    model = SARIMAX(y.to_numpy(dtype=float), exog=exog,
                    order=(1, 1, 1), seasonal_order=(1, 0, 1, 12) if seasonal else (0, 0, 0, 0),
                    enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("always", ConvergenceWarning)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            fit = model.fit(disp=False, maxiter=300)
        if any(issubclass(w.category, ConvergenceWarning) for w in captured) or not fit.mle_retvals.get("converged", False):
            FIT_WARNINGS.append({"history_end": str(history.month.iloc[-1]), "seasonal": seasonal, "exogenous": exogenous})
    return np.asarray(fit.forecast(steps=steps, exog=future_exog), dtype=float)


def run_timeseries_modeling_suite(df: pd.DataFrame, table_dir: Path, fig_dir: Path) -> dict[str, object]:
    FIT_WARNINGS.clear()
    months = pd.PeriodIndex(df["month"], freq="M")
    if len(df) != 72 or not months.equals(pd.period_range("2019-01", "2024-12", freq="M")):
        raise ValueError("Cần chuỗi hàng tháng liên tục từ 2019-01 đến 2024-12")
    y = df["rent_price_million_vnd"].to_numpy(dtype=float)
    rows = []
    # 2023 dùng chọn mô hình; 2024 chỉ dùng đánh giá sau khi chọn.
    for t in range(48, 72):
        history = df.iloc[:t].copy()
        values = {"Naive (tháng trước)": y[t-1], "Seasonal naive (cùng tháng năm trước)": y[t-12]}
        for name, (seasonal, exogenous) in TS_MODELS.items():
            values[name] = _sarimax_forecast(history, 1, seasonal, exogenous)[0]
        # Mô hình LSTM chỉ dùng dữ liệu tới t-1; huấn luyện lại tại từng mốc.
        values["LSTM (Lookback=6, Units=24)"] = _fit_predict_lstm(df.iloc[:t+1].copy(), t)[0]
        for name, prediction in values.items():
            rows.append({"month": str(months[t]), "period": "validation" if t < 60 else "test",
                         "model": name, "actual": y[t], "prediction": prediction})
    rolling = pd.DataFrame(rows)
    rolling.to_csv(table_dir / "ts8_rolling_predictions_2023_2024.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for (period, name), group in rolling.groupby(["period", "model"], sort=False):
        summaries.append({"period": period, "Mô_hình": name, "N": len(group),
                          **compute_ts_metrics(group.actual.to_numpy(), group.prediction.to_numpy())})
    metrics = pd.DataFrame(summaries)
    metrics.to_csv(table_dir / "ts9_rolling_metrics_2023_2024.csv", index=False, encoding="utf-8-sig")
    warning_df = pd.DataFrame(FIT_WARNINGS, columns=["history_end", "seasonal", "exogenous"])
    warning_df.to_csv(table_dir / "ts10_fit_warnings.csv", index=False, encoding="utf-8-sig")
    failed_specs = {(r["seasonal"], r["exogenous"]) for r in FIT_WARNINGS if r["history_end"] < "2024-01"}
    eligible = [name for name in metrics["Mô_hình"].unique() if name not in TS_MODELS or TS_MODELS[name] not in failed_specs]
    validation = metrics[(metrics.period == "validation") & metrics["Mô_hình"].isin(eligible)].sort_values("RMSE")
    selected = validation.iloc[0]["Mô_hình"]
    comp_df = metrics[metrics.period == "test"].sort_values("RMSE").reset_index(drop=True)
    comp_df.insert(0, "Hạng", np.arange(1, len(comp_df)+1))
    comp_df["Được_chọn_từ_validation"] = comp_df["Mô_hình"] == selected
    comp_df.to_csv(table_dir / "ts5_model_comparison_rq10.csv", index=False, encoding="utf-8-sig")
    pred_table = rolling[rolling.period == "test"].pivot(index="month", columns="model", values="prediction").reset_index()
    pred_table.insert(1, "Actual_Rent_Million_VND", y[60:])
    pred_table.to_csv(table_dir / "ts6_test_2024_monthly_predictions.csv", index=False, encoding="utf-8-sig")

    future_months = pd.period_range("2025-01", periods=12, freq="M").astype(str)
    if selected in TS_MODELS:
        future = _sarimax_forecast(df, 12, *TS_MODELS[selected])
        assumption = "Biến ngoại sinh giữ mức tháng 12/2024" if TS_MODELS[selected][1] else "Không sử dụng biến ngoại sinh"
    elif selected == "Naive (tháng trước)":
        future = np.repeat(y[-1], 12)
        assumption = "Không sử dụng biến ngoại sinh"
    elif selected == "Seasonal naive (cùng tháng năm trước)":
        future = y[-12:].copy()
        assumption = "Không sử dụng biến ngoại sinh"
    else:
        future_input = pd.concat([df.copy(), pd.DataFrame([df.iloc[-1].copy() for _ in future_months])], ignore_index=True)
        future_input.loc[len(df):, "month"] = future_months
        future_input.loc[len(df):, "rent_price_million_vnd"] = np.nan
        future = _fit_predict_lstm(future_input, len(df), recursive=True)
        assumption = "Biến ngoại sinh giữ mức tháng 12/2024"
    future_df = pd.DataFrame({"month": future_months, "Forecast_Rent_Million_VND": np.round(future, 4),
                              "Model_Used": selected, "Exogenous_Assumption": assumption})
    future_df.to_csv(table_dir / "ts7_future_forecast_2025.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(comp_df["Mô_hình"], comp_df["RMSE"], color="#2563EB")
    ax.invert_yaxis(); ax.set_xlabel("RMSE (triệu VNĐ/tháng)")
    ax.set_title("Đánh giá cuốn chiếu một tháng: năm 2024")
    fig.tight_layout(); fig.savefig(fig_dir / "13_ts_model_comparison_rq10.png", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6))
    dates = pd.to_datetime(df.month + "-01")
    ax.plot(dates, y, label="Thực tế 2019–2024")
    selected_test = rolling[(rolling.period == "test") & (rolling.model == selected)]
    ax.plot(pd.to_datetime(selected_test.month + "-01"), selected_test.prediction, label="Dự báo một bước trên test")
    ax.plot(pd.to_datetime(future_df.month + "-01"), future_df.Forecast_Rent_Million_VND, label="Dự báo 2025")
    ax.set_ylabel("Giá thuê TB (triệu VNĐ/tháng)"); ax.legend(); ax.set_title(selected)
    fig.tight_layout(); fig.savefig(fig_dir / "14_ts_full_trajectory_and_2025_forecast.png", bbox_inches="tight"); plt.close(fig)
    return {"comparison_df": comp_df, "predictions_df": pred_table, "future_df": future_df,
            "rolling_metrics_df": metrics, "selected_model": selected}
