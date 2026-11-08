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
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
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
    # Tính MAE, RMSE (triệu VNĐ) và MAPE, R² (thang thập phân 0-1)
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


def _fit_with_best_target_mode(
    estimator,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    param_distributions: dict | None = None,
    n_iter: int = 12,
    allow_log_target: bool = True,
    random_state: int = 42,
) -> tuple[object, np.ndarray, np.ndarray, float, str]:
    # Tối ưu siêu tham số 5-Fold CV trên tập Train
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    def tune_and_cv(target_vec: np.ndarray, is_log: bool):
        if param_distributions:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_distributions,
                n_iter=min(n_iter, int(np.prod([len(v) for v in param_distributions.values()]))),
                scoring="neg_root_mean_squared_error",
                cv=kf,
                random_state=random_state,
                n_jobs=1,
            )
            search.fit(X_tr, target_vec)
            best_est = search.best_estimator_
            best_params = search.best_params_
        else:
            from sklearn.base import clone
            best_est = clone(estimator)
            best_est.fit(X_tr, target_vec)
            best_params = {}

        oof_pred_raw = cross_val_predict(best_est, X_tr, target_vec, cv=kf, n_jobs=1)
        tr_pred_raw = best_est.predict(X_tr)
        te_pred_raw = best_est.predict(X_te)

        if is_log:
            resid_tr = target_vec - oof_pred_raw
            smearing = float(np.median(np.exp(resid_tr)))
            oof_pred = np.exp(oof_pred_raw) * smearing
            tr_pred = np.exp(tr_pred_raw) * smearing
            te_pred = np.exp(te_pred_raw) * smearing
        else:
            oof_pred = oof_pred_raw
            tr_pred = tr_pred_raw
            te_pred = te_pred_raw

        cv_r2 = float(r2_score(y_tr, np.clip(oof_pred, 1.5, None)))
        cv_rmse = float(np.sqrt(mean_squared_error(y_tr, np.clip(oof_pred, 1.5, None))))
        return best_est, tr_pred, te_pred, cv_r2, cv_rmse, best_params

    est_dir, tr_dir, te_dir, cv_r2_dir, cv_rmse_dir, params_dir = tune_and_cv(y_tr, is_log=False)

    if allow_log_target:
        est_log, tr_log, te_log, cv_r2_log, cv_rmse_log, params_log = tune_and_cv(np.log(y_tr), is_log=True)
        if cv_rmse_log < cv_rmse_dir:
            param_str = ", ".join(f"{k}={v}" for k, v in params_log.items()) if params_log else "Default"
            return est_log, tr_log, te_log, cv_r2_log, f"ln(y) | {param_str}"

    param_str = ", ".join(f"{k}={v}" for k, v in params_dir.items()) if params_dir else "Default"
    return est_dir, tr_dir, te_dir, cv_r2_dir, f"direct | {param_str}"


def plot_06_model_comparison(results_df: pd.DataFrame, fig_dir: Path) -> None:
    # Hình 6: So sánh MAE, RMSE, MAPE và R² giữa các mô hình
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    models = results_df["Mô_hình"].tolist()
    colors = PALETTE_MAIN[: len(models)]

    specs = [
        (axes[0, 0], "Test_MAE", "(a) Sai số tuyệt đối trung bình MAE (triệu VNĐ) - Thấp hơn là tốt", True),
        (axes[0, 1], "Test_RMSE", "(b) Sai số toàn phương trung bình RMSE (triệu VNĐ) - Thấp hơn là tốt", True),
        (axes[1, 0], "Test_MAPE", "(c) Sai số tỷ lệ tuyệt đối trung bình MAPE (0–1) - Thấp hơn là tốt", True),
        (axes[1, 1], "Test_R2", "(d) Hệ số xác định R² trên tập Test (0–1) - Cao hơn là tốt", False),
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
    ax1.set_title("(b) Độ quan trọng hoán vị trên tập Test (Permutation Importance)", fontsize=11.5, fontweight="bold")
    for idx, val in enumerate(top_perm["Permutation_Importance"]):
        ax1.text(val + 0.005, idx, f"{val:.4f}", va="center", fontsize=8.8, fontweight="bold")
    ax1.set_xlim(0, float(top_perm["Permutation_Importance"].max()) * 1.18)

    fig.suptitle("Trả lời RQ7: Xếp hạng các yếu tố quan trọng nhất quyết định giá thuê căn hộ", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "09_feature_importance_rq7.png", bbox_inches="tight")
    plt.close(fig)


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

    rows = []
    preds_dict: dict[str, np.ndarray] = {}
    fitted_models: dict[str, object] = {}

    for name, branch, est, param_grid in model_specs:
        X_tr_use = X_tr_lin if branch == "linear" else X_tr_tree
        X_te_use = X_te_lin if branch == "linear" else X_te_tree

        best_est, tr_pred, te_pred, cv_r2, cfg_desc = _fit_with_best_target_mode(
            estimator=est,
            X_tr=X_tr_use,
            y_tr=y_tr,
            X_te=X_te_use,
            param_distributions=param_grid,
            n_iter=12,
            allow_log_target=(branch == "tree"),
            random_state=42,
        )

        tr_metrics = compute_metrics(y_tr, tr_pred)
        te_metrics = compute_metrics(y_te, te_pred)

        preds_dict[name] = te_pred
        fitted_models[name] = best_est

        rows.append({
            "Mô_hình": name,
            "Nhánh_Pipeline": "preprocessor_linear" if branch == "linear" else "preprocessor_tree",
            "CV_5Fold_R2": round(cv_r2, 4),
            "Train_R2": tr_metrics["R2"],
            "Test_MAE": te_metrics["MAE"],
            "Test_RMSE": te_metrics["RMSE"],
            "Test_MAPE": te_metrics["MAPE"],
            "Test_R2": te_metrics["R2"],
            "Cấu_hình_tối_ưu": cfg_desc,
        })

    results_df = pd.DataFrame(rows).sort_values("Test_RMSE", ascending=True).reset_index(drop=True)
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
        rf_model, X_te_tree, np.log(y_te) if np.max(rf_model.predict(X_te_tree)) < 6.0 else y_te,
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

    best_model_name = str(results_df.iloc[0]["Mô_hình"])
    best_pred = np.clip(preds_dict[best_model_name], 1.5, None)
    pred_sample_df = test_df[["listing_id", "sub_district", "ward", "project_name", "apartment_type", "area_m2", "bedrooms", "price_million_vnd"]].copy()
    pred_sample_df["Predicted_Price_Best"] = np.round(best_pred, 2)
    pred_sample_df["Residual_Error"] = np.round(pred_sample_df["price_million_vnd"] - pred_sample_df["Predicted_Price_Best"], 2)
    pred_sample_df["APE"] = np.round(
        np.abs(pred_sample_df["Residual_Error"]) / pred_sample_df["price_million_vnd"].clip(lower=0.5), 4
    )
    pred_sample_df.to_csv(table_dir / "d11_test_set_predictions.csv", index=False, encoding="utf-8-sig")

    plot_06_model_comparison(results_df, fig_dir)
    plot_07_actual_vs_predicted(y_te, preds_dict, results_df, fig_dir)
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

