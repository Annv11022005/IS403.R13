# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor

from src.config import PALETTE_MAIN
from src.model.feature_pipeline import LeakFreeFeatureBuilder, create_stratified_split

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # MAE/RMSE: triệu VNĐ/tháng; MAPE: tỷ lệ; R² có thể âm.
    y_pred_clean = np.clip(y_pred, 1.5, None)
    mae = float(mean_absolute_error(y_true, y_pred_clean))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clean)))
    mape = float(np.mean(np.abs((y_true - y_pred_clean) / np.clip(y_true, 0.5, None))))
    r2 = float(r2_score(y_true, y_pred_clean))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4),
    }


def _fit_with_best_target_mode(estimator, folds, y_train, param_distributions, branch, allow_log_target):
    # Mỗi fold có bộ mã hóa và chuẩn hóa học riêng từ phần train của fold.
    candidates = list(ParameterSampler(param_distributions, n_iter=12, random_state=42)) if param_distributions else [{}]
    best = None
    for params in candidates:
        for log_target in ([False, True] if allow_log_target else [False]):
            oof = np.empty(len(y_train), dtype=float)
            for tr_idx, val_idx, x_train, x_val in folds[branch]:
                model = clone(estimator).set_params(**params)
                target = np.log(y_train[tr_idx]) if log_target else y_train[tr_idx]
                model.fit(x_train, target)
                pred = model.predict(x_val)
                oof[val_idx] = np.exp(pred) if log_target else pred
            metrics = compute_metrics(y_train, oof)
            score = metrics["RMSE"]
            if best is None or score < best["rmse"]:
                best = {"params": params, "log": log_target, "rmse": score, "r2": metrics["R2"]}
    return best


def plot_06_model_comparison(results_df: pd.DataFrame, fig_dir: Path) -> None:
    # Hình 6: So sánh MAE, RMSE, MAPE và R² giữa các mô hình
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    models = results_df["Mô_hình"].tolist()
    colors = PALETTE_MAIN[: len(models)]

    specs = [
        (axes[0, 0], "Test_MAE", "(a) Sai số tuyệt đối trung bình MAE (triệu VNĐ) - Thấp hơn là tốt", True),
        (axes[0, 1], "Test_RMSE", "(b) Sai số toàn phương trung bình RMSE (triệu VNĐ) - Thấp hơn là tốt", True),
        (axes[1, 0], "Test_MAPE", "(c) Sai số tỷ lệ tuyệt đối trung bình MAPE (tỷ lệ thập phân) - Thấp hơn là tốt", True),
        (axes[1, 1], "Test_R2", "(d) Hệ số xác định R² trên tập Test - Cao hơn là tốt", False),
    ]

    for ax, col, title, lower_better in specs:
        vals = results_df[col].values
        bars = ax.bar(np.arange(len(models)), vals, color=colors, alpha=0.85, width=0.55, edgecolor="#1F2937")
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=16, ha="right", fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")

        best_idx = int(np.argmin(vals)) if lower_better else int(np.argmax(vals))
        bars[best_idx].set_edgecolor("#DC2626")
        bars[best_idx].set_linewidth(2.2)

        y_max = float(np.max(vals))
        ax.set_ylim(0, y_max * 1.18)
        for idx, bar in enumerate(bars):
            h = bar.get_height()
            fmt = f"{h:.4f}"
            weight = "bold" if idx == best_idx else "normal"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + y_max * 0.02,
                fmt,
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight=weight,
                color="#DC2626" if idx == best_idx else "#111827",
            )

    fig.suptitle("D.11. So sánh hiệu năng dự báo giá thuê của các mô hình Machine Learning (RQ10)", fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_model_comparison_d11.png", bbox_inches="tight")
    plt.close(fig)


def plot_07_actual_vs_predicted(
    y_test: np.ndarray,
    preds_dict: dict[str, np.ndarray],
    results_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    # Hình 7: Giá thực tế vs Giá dự báo trên tập Test
    selected = ["Linear Regression", "Random Forest", "XGBoost"]
    selected = [m for m in selected if m in preds_dict]

    fig, axes = plt.subplots(1, len(selected), figsize=(15.5, 4.9))
    if len(selected) == 1:
        axes = [axes]

    min_v = float(min(y_test.min(), 2.0))
    max_v = float(max(y_test.max(), 38.0))

    for ax, model_name, color in zip(axes, selected, ["#64748B", "#10B981", "#2563EB"]):
        y_pred = np.clip(preds_dict[model_name], 1.5, None)
        row = results_df[results_df["Mô_hình"] == model_name].iloc[0]

        ax.scatter(y_test, y_pred, alpha=0.42, s=20, color=color, edgecolors="none", label="Căn hộ tập Test")
        ax.plot([min_v, max_v], [min_v, max_v], color="#DC2626", ls="--", lw=2.0, label="Đường lý tưởng (y = ŷ)")
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.set_xlabel("Giá thuê thực tế (triệu VNĐ/tháng)")
        ax.set_ylabel("Giá thuê dự báo (triệu VNĐ/tháng)")
        ax.set_title(
            f"{model_name}\n(R² = {row['Test_R2']:.4f} | RMSE = {row['Test_RMSE']:.4f} | MAPE = {row['Test_MAPE']:.4f})",
            fontsize=10.2,
            fontweight="bold",
        )
        ax.legend(frameon=True, fontsize=8.8, loc="upper left")

    fig.suptitle("Đối chiếu Giá thuê thực tế và Giá dự báo trên tập Kiểm tra (Test Set)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "07_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)


def plot_08_residuals_diagnostics(
    y_test: np.ndarray,
    best_model_name: str,
    best_pred: np.ndarray,
    fig_dir: Path,
) -> None:
    # Hình 8: Phân tích phần dư của mô hình tốt nhất
    y_pred = np.clip(best_pred, 1.5, None)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))

    ax0 = axes[0]
    ax0.scatter(y_pred, residuals, alpha=0.42, s=20, color="#2563EB")
    ax0.axhline(0.0, color="#DC2626", ls="--", lw=2.0, label="Sai số = 0")
    ax0.set_title(f"(a) Phần dư theo Giá dự báo ({best_model_name})", fontsize=11.5, fontweight="bold")
    ax0.set_xlabel("Giá thuê dự báo ŷ (triệu VNĐ/tháng)")
    ax0.set_ylabel("Phần dư e = y - ŷ (triệu VNĐ)")
    ax0.legend(frameon=True, fontsize=9)

    ax1 = axes[1]
    ax1.hist(residuals, bins=32, density=True, color="#10B981", alpha=0.6, edgecolor="white")
    ax1.axvline(0.0, color="#DC2626", ls="--", lw=2.0, label="Tâm sai số = 0")
    ax1.axvline(np.mean(residuals), color="#111827", ls="-.", lw=1.6, label=f"Mean sai số = {np.mean(residuals):+.2f} tr")
    within_2m_ratio = float(np.mean(np.abs(residuals) <= 2.0))
    ax1.set_title(f"(b) Phân bố sai số (Tỷ lệ sai lệch trong ±2 triệu = {within_2m_ratio:.4f})", fontsize=11.2, fontweight="bold")
    ax1.set_xlabel("Sai số dự báo e = y - ŷ (triệu VNĐ)")
    ax1.set_ylabel("Mật độ")
    ax1.legend(frameon=True, fontsize=9)

    fig.suptitle(f"Phân tích sai số dự báo của mô hình {best_model_name} trên tập Test", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "08_residuals_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def plot_09_feature_importance(imp_df: pd.DataFrame, fig_dir: Path) -> None:
    # Hình 9: Xếp hạng Feature Importance (RQ7) trên thang chuẩn 0-1
    top_df = imp_df.head(12).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))

    ax0 = axes[0]
    ax0.barh(top_df["Mô_tả"], top_df["Importance_Tree"], color="#2563EB", alpha=0.85, edgecolor="#1E3A8A")
    ax0.set_xlabel("Tỷ trọng đóng góp trong mô hình cây (0–1)")
    ax0.set_title("(a) Độ quan trọng đặc trưng nội tại (Tree Importance)", fontsize=11.5, fontweight="bold")
    for idx, val in enumerate(top_df["Importance_Tree"]):
        ax0.text(val + 0.005, idx, f"{val:.4f}", va="center", fontsize=8.8, fontweight="bold")
    ax0.set_xlim(0, float(top_df["Importance_Tree"].max()) * 1.18)

    top_perm = imp_df.sort_values("Permutation_Importance", ascending=False).head(12).iloc[::-1]
    ax1 = axes[1]
    ax1.barh(top_perm["Mô_tả"], top_perm["Permutation_Importance"], color="#10B981", alpha=0.85, edgecolor="#065F46")
    ax1.set_xlabel("Mức độ sụt giảm hiệu năng chuẩn hóa (0–1)")
    ax1.set_title("(b) Độ quan trọng hoán vị trên tập Train (Permutation Importance)", fontsize=11.5, fontweight="bold")
    for idx, val in enumerate(top_perm["Permutation_Importance"]):
        ax1.text(val + 0.005, idx, f"{val:.4f}", va="center", fontsize=8.8, fontweight="bold")
    ax1.set_xlim(0, float(top_perm["Permutation_Importance"].max()) * 1.18)

    fig.suptitle("Trả lời RQ7: Xếp hạng độ quan trọng đặc trưng trong mô hình", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "09_feature_importance_rq7.png", bbox_inches="tight")
    plt.close(fig)


def plot_cv_comparison(results_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(results_df["Mô_hình"], results_df["CV_5Fold_RMSE"], color="#2563EB")
    ax.invert_yaxis(); ax.set_xlabel("RMSE cross-validation (triệu VNĐ/tháng)")
    ax.set_title("Chọn mô hình bằng 5-fold cross-validation trên tập train")
    fig.tight_layout(); fig.savefig(fig_dir / "06_model_comparison_d11.png", bbox_inches="tight"); plt.close(fig)


def plot_selected_predictions(y_test: np.ndarray, pred: np.ndarray, name: str, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, pred, alpha=0.4, s=20)
    maximum = max(float(y_test.max()), float(pred.max()))
    ax.plot([0, maximum], [0, maximum], "r--")
    ax.set_xlabel("Giá thực tế (triệu VNĐ/tháng)"); ax.set_ylabel("Giá dự báo (triệu VNĐ/tháng)")
    ax.set_title("Tập test: " + name)
    fig.tight_layout(); fig.savefig(fig_dir / "07_actual_vs_predicted.png", bbox_inches="tight"); plt.close(fig)


def run_ml_pipeline(df: pd.DataFrame, table_dir: Path, fig_dir: Path) -> dict[str, object]:
    train_df, test_df = create_stratified_split(df, test_size=0.20, random_state=42)

    builder = LeakFreeFeatureBuilder(smoothing=12.0, n_splits=5, random_state=42)
    X_tr_lin, X_tr_tree, feat_names = builder.fit_transform(train_df)
    X_te_lin, X_te_tree = builder.transform(test_df)

    y_tr = train_df["price_million_vnd"].values.astype(float)
    y_te = test_df["price_million_vnd"].values.astype(float)

    model_specs = [
        ("Linear Regression", "linear", LinearRegression(), None),
        ("Ridge Regression", "linear", Ridge(random_state=42), {"alpha": [0.1, 1.0, 5.0, 15.0, 50.0, 120.0]}),
        (
            "Decision Tree",
            "tree",
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [5, 7, 9, 12], "min_samples_split": [8, 15, 25], "min_samples_leaf": [4, 8, 12]},
        ),
        (
            "Random Forest",
            "tree",
            RandomForestRegressor(random_state=42, n_jobs=1),
            {
                "n_estimators": [200, 350],
                "max_depth": [12, 16, 22, None],
                "min_samples_split": [4, 8, 12],
                "min_samples_leaf": [2, 4, 6],
                "max_features": [0.6, 0.8, "sqrt"],
            },
        ),
    ]

    if HAS_XGBOOST:
        model_specs.append((
            "XGBoost",
            "tree",
            XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=1, verbosity=0),
            {
                "n_estimators": [250, 400, 550],
                "max_depth": [4, 5, 6, 7],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.75, 0.85, 0.95],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [1.0, 3.0, 8.0],
            },
        ))
    else:
        model_specs.append((
            "XGBoost (HistGB)",
            "tree",
            HistGradientBoostingRegressor(random_state=42),
            {
                "max_iter": [200, 350, 500],
                "max_depth": [5, 7, 10],
                "learning_rate": [0.03, 0.05, 0.08],
                "min_samples_leaf": [10, 20, 30],
                "l2_regularization": [0.1, 1.0, 5.0],
            },
        ))

    # Chuẩn bị đặc trưng độc lập cho từng fold trước khi thử các cấu hình.
    folds = {"linear": [], "tree": []}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in cv.split(train_df):
        fold_builder = LeakFreeFeatureBuilder(smoothing=12.0, n_splits=5, random_state=42)
        tr_lin, tr_tree, _ = fold_builder.fit_transform(train_df.iloc[tr_idx].reset_index(drop=True))
        val_lin, val_tree = fold_builder.transform(train_df.iloc[val_idx].reset_index(drop=True))
        folds["linear"].append((tr_idx, val_idx, tr_lin, val_lin))
        folds["tree"].append((tr_idx, val_idx, tr_tree, val_tree))

    rows, selections, fitted_models, preds_dict = [], {}, {}, {}
    for name, branch, estimator, param_grid in model_specs:
        selected = _fit_with_best_target_mode(estimator, folds, y_tr, param_grid, branch, branch == "tree")
        selections[name] = (branch, estimator, selected)
        cfg = ", ".join(f"{k}={v}" for k, v in selected["params"].items()) or "Default"
        rows.append({"Mô_hình": name, "Nhánh_Pipeline": "preprocessor_" + branch,
                     "CV_5Fold_R2": selected["r2"], "CV_5Fold_RMSE": selected["rmse"],
                     "Cấu_hình_tối_ưu": ("ln(y)" if selected["log"] else "direct") + " | " + cfg})
    results_df = pd.DataFrame(rows).sort_values("CV_5Fold_RMSE").reset_index(drop=True)
    best_model_name = str(results_df.iloc[0]["Mô_hình"])
    results_df["Được_chọn_từ_CV"] = results_df["Mô_hình"] == best_model_name

    # Huấn luyện lại trên toàn bộ train; chỉ mô hình được chọn được chấm trên test.
    for name in ("Random Forest", "XGBoost" if HAS_XGBOOST else "XGBoost (HistGB)", best_model_name):
        if name in fitted_models:
            continue
        branch, estimator, selection = selections[name]
        x = X_tr_lin if branch == "linear" else X_tr_tree
        target = np.log(y_tr) if selection["log"] else y_tr
        model = clone(estimator).set_params(**selection["params"]).fit(x, target)
        fitted_models[name] = model
    branch, _, selected = selections[best_model_name]
    chosen_model = fitted_models[best_model_name]
    x_test = X_te_lin if branch == "linear" else X_te_tree
    best_pred = chosen_model.predict(x_test)
    if selected["log"]:
        best_pred = np.exp(best_pred)
    best_pred = np.clip(best_pred, 1.5, None)
    preds_dict[best_model_name] = best_pred
    te_metrics = compute_metrics(y_te, best_pred)
    for key, value in te_metrics.items():
        results_df["Test_" + key] = np.nan
        results_df.loc[results_df["Được_chọn_từ_CV"], "Test_" + key] = value
    results_df.to_csv(table_dir / "d11_ml_model_comparison.csv", index=False, encoding="utf-8-sig")

    rf_model = fitted_models["Random Forest"]
    xgb_key = "XGBoost" if "XGBoost" in fitted_models else "XGBoost (HistGB)"
    xgb_model = fitted_models[xgb_key]

    rf_imp = getattr(rf_model, "feature_importances_", np.zeros(len(feat_names)))
    xgb_imp = getattr(xgb_model, "feature_importances_", rf_imp)

    rf_imp_norm = rf_imp / max(float(np.sum(rf_imp)), 1e-12)
    xgb_imp_norm = xgb_imp / max(float(np.sum(xgb_imp)), 1e-12)
    tree_imp_norm = 0.5 * (rf_imp_norm + xgb_imp_norm)

    perm = permutation_importance(
        rf_model, X_tr_tree, np.log(y_tr) if selections["Random Forest"][2]["log"] else y_tr,
        n_repeats=8, random_state=42, n_jobs=1
    )
    perm_raw = np.clip(perm.importances_mean, 0.0, None)
    perm_norm = perm_raw / max(float(np.sum(perm_raw)), 1e-12)

    imp_df = pd.DataFrame({
        "Biến": feat_names,
        "Mô_tả": [builder.feature_labels_vi_.get(f, f) for f in feat_names],
        "Importance_RF": np.round(rf_imp_norm, 4),
        "Importance_XGB": np.round(xgb_imp_norm, 4),
        "Importance_Tree": np.round(tree_imp_norm, 4),
        "Permutation_Importance": np.round(perm_norm, 4),
    }).sort_values("Importance_Tree", ascending=False).reset_index(drop=True)

    imp_df.to_csv(table_dir / "rq7_feature_importance.csv", index=False, encoding="utf-8-sig")

    pred_sample_df = test_df[["listing_id", "sub_district", "ward", "project_name", "apartment_type", "area_m2", "bedrooms", "price_million_vnd"]].copy()
    pred_sample_df["Predicted_Price_Best"] = np.round(best_pred, 2)
    pred_sample_df["Residual_Error"] = np.round(pred_sample_df["price_million_vnd"] - pred_sample_df["Predicted_Price_Best"], 2)
    pred_sample_df["APE"] = np.round(
        np.abs(pred_sample_df["Residual_Error"]) / pred_sample_df["price_million_vnd"].clip(lower=0.5), 4
    )
    pred_sample_df.to_csv(table_dir / "d11_test_set_predictions.csv", index=False, encoding="utf-8-sig")

    plot_cv_comparison(results_df, fig_dir)
    plot_selected_predictions(y_te, best_pred, best_model_name, fig_dir)
    plot_08_residuals_diagnostics(y_te, best_model_name, best_pred, fig_dir)
    plot_09_feature_importance(imp_df, fig_dir)

    return {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "n_features": len(feat_names),
        "results_df": results_df,
        "importance_df": imp_df,
        "best_model_name": best_model_name,
    }

