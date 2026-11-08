# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Chia Train (80%) / Test (20%) có phân tầng
    work = df.copy().reset_index(drop=True)
    price_bin = pd.qcut(work["price_million_vnd"], q=4, labels=False, duplicates="drop")
    strat_key = (
        work["sub_district"].astype(str)
        + "_"
        + work["is_commercial_project"].astype(str)
        + "_"
        + price_bin.astype(str)
    )
    counts = strat_key.value_counts()
    rare_keys = set(counts[counts < 5].index)
    if rare_keys:
        strat_key = strat_key.where(~strat_key.isin(rare_keys), other="rare_stratum")

    train_df, test_df = train_test_split(
        work,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class LeakFreeFeatureBuilder:
    # Pipeline mã hóa One-Hot & Target Encoding học trên tập Train
    def __init__(self, smoothing: float = 12.0, n_splits: int = 5, random_state: int = 42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

        self.global_price_mean_: float = 0.0
        self.global_pm2_mean_: float = 0.0
        self.global_area_mean_: float = 0.0

        self.proj_price_map_: dict[str, float] = {}
        self.proj_pm2_map_: dict[str, float] = {}
        self.proj_area_map_: dict[str, float] = {}

        self.ward_price_map_: dict[str, float] = {}
        self.ward_pm2_map_: dict[str, float] = {}

        self.sub_pm2_map_: dict[str, float] = {}

        self.ohe_: OneHotEncoder | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.feature_labels_vi_: dict[str, str] = {}

    @staticmethod
    def _make_keys(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        proj_clean = out["project_name"].fillna("").astype(str).str.strip()
        fallback_proj = (
            "NoProj_"
            + out["ward"].fillna("Unknown").astype(str)
            + "_"
            + out["apartment_type"].fillna("Unknown").astype(str)
        )
        out["project_key"] = np.where(proj_clean != "", proj_clean, fallback_proj)
        out["ward_key"] = (
            out["sub_district"].fillna("").astype(str)
            + "__"
            + out["ward"].fillna("").astype(str)
            + "__"
            + out["is_commercial_project"].astype(str)
        )
        return out

    def _compute_smoothed_map(
        self,
        keys: pd.Series,
        target: pd.Series,
        prior_map: dict[str, float] | None = None,
        global_prior: float = 0.0,
        parent_keys: pd.Series | None = None,
    ) -> dict[str, float]:
        stats_df = pd.DataFrame({"key": keys, "target": target})
        if parent_keys is not None:
            stats_df["parent"] = parent_keys
            grp = stats_df.groupby("key").agg(
                count=("target", "count"),
                mean=("target", "mean"),
                parent=("parent", "first"),
            )
        else:
            grp = stats_df.groupby("key").agg(
                count=("target", "count"),
                mean=("target", "mean"),
            )

        result = {}
        for k, row in grp.iterrows():
            n = float(row["count"])
            m = float(row["mean"])
            prior = prior_map.get(row["parent"], global_prior) if (prior_map and "parent" in row) else global_prior
            smoothed = (n * m + self.smoothing * prior) / (n + self.smoothing)
            result[str(k)] = float(smoothed)
        return result

    def fit_transform(self, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        work = self._make_keys(train_df)
        y_price = work["price_million_vnd"].astype(float)
        y_pm2 = y_price / work["area_m2"].clip(lower=10.0)

        self.global_price_mean_ = float(y_price.mean())
        self.global_pm2_mean_ = float(y_pm2.mean())
        self.global_area_mean_ = float(work["area_m2"].mean())

        self.sub_pm2_map_ = self._compute_smoothed_map(
            work["sub_district"], y_pm2, global_prior=self.global_pm2_mean_
        )
        self.ward_price_map_ = self._compute_smoothed_map(
            work["ward_key"], y_price, global_prior=self.global_price_mean_
        )
        self.ward_pm2_map_ = self._compute_smoothed_map(
            work["ward_key"],
            y_pm2,
            prior_map=self.sub_pm2_map_,
            global_prior=self.global_pm2_mean_,
            parent_keys=work["sub_district"],
        )
        self.proj_price_map_ = self._compute_smoothed_map(
            work["project_key"],
            y_price,
            prior_map=self.ward_price_map_,
            global_prior=self.global_price_mean_,
            parent_keys=work["ward_key"],
        )
        self.proj_pm2_map_ = self._compute_smoothed_map(
            work["project_key"],
            y_pm2,
            prior_map=self.ward_pm2_map_,
            global_prior=self.global_pm2_mean_,
            parent_keys=work["ward_key"],
        )
        self.proj_area_map_ = self._compute_smoothed_map(
            work["project_key"], work["area_m2"], global_prior=self.global_area_mean_
        )

        oof_proj_price = np.full(len(work), self.global_price_mean_, dtype=float)
        oof_proj_pm2 = np.full(len(work), self.global_pm2_mean_, dtype=float)
        oof_ward_price = np.full(len(work), self.global_price_mean_, dtype=float)
        oof_ward_pm2 = np.full(len(work), self.global_pm2_mean_, dtype=float)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr_idx, val_idx in kf.split(work):
            fold_tr = work.iloc[tr_idx]
            fold_val = work.iloc[val_idx]
            f_price = y_price.iloc[tr_idx]
            f_pm2 = y_pm2.iloc[tr_idx]

            f_sub_pm2 = self._compute_smoothed_map(fold_tr["sub_district"], f_pm2, global_prior=float(f_pm2.mean()))
            f_ward_price = self._compute_smoothed_map(fold_tr["ward_key"], f_price, global_prior=float(f_price.mean()))
            f_ward_pm2 = self._compute_smoothed_map(
                fold_tr["ward_key"], f_pm2, prior_map=f_sub_pm2, global_prior=float(f_pm2.mean()), parent_keys=fold_tr["sub_district"]
            )
            f_proj_price = self._compute_smoothed_map(
                fold_tr["project_key"], f_price, prior_map=f_ward_price, global_prior=float(f_price.mean()), parent_keys=fold_tr["ward_key"]
            )
            f_proj_pm2 = self._compute_smoothed_map(
                fold_tr["project_key"], f_pm2, prior_map=f_ward_pm2, global_prior=float(f_pm2.mean()), parent_keys=fold_tr["ward_key"]
            )

            for pos, row_i in enumerate(val_idx):
                w_k = str(fold_val["ward_key"].iat[pos])
                p_k = str(fold_val["project_key"].iat[pos])
                s_k = str(fold_val["sub_district"].iat[pos])

                w_pr = f_ward_price.get(w_k, float(f_price.mean()))
                w_pm = f_ward_pm2.get(w_k, f_sub_pm2.get(s_k, float(f_pm2.mean())))
                oof_ward_price[row_i] = w_pr
                oof_ward_pm2[row_i] = w_pm
                oof_proj_price[row_i] = f_proj_price.get(p_k, w_pr)
                oof_proj_pm2[row_i] = f_proj_pm2.get(p_k, w_pm)

        num_df = self._build_numeric_frame(work, oof_proj_price, oof_proj_pm2, oof_ward_price, oof_ward_pm2)

        cat_cols = ["sub_district", "apartment_type", "furniture"]
        try:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse=False)

        ohe_mat = self.ohe_.fit_transform(work[cat_cols].fillna("Unknown").astype(str))
        ohe_names = list(self.ohe_.get_feature_names_out(cat_cols))

        X_tree = np.hstack([num_df.values, ohe_mat])
        self.feature_names_ = list(num_df.columns) + ohe_names
        self._populate_vi_labels()

        self.scaler_ = StandardScaler()
        num_scaled = self.scaler_.fit_transform(num_df.values)
        X_linear = np.hstack([num_scaled, ohe_mat])

        return X_linear, X_tree, self.feature_names_

    def transform(self, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        work = self._make_keys(test_df)

        ward_price = np.array([
            self.ward_price_map_.get(str(wk), self.global_price_mean_)
            for wk in work["ward_key"]
        ], dtype=float)
        ward_pm2 = np.array([
            self.ward_pm2_map_.get(str(wk), self.sub_pm2_map_.get(str(sk), self.global_pm2_mean_))
            for wk, sk in zip(work["ward_key"], work["sub_district"])
        ], dtype=float)
        proj_price = np.array([
            self.proj_price_map_.get(str(pk), wp)
            for pk, wp in zip(work["project_key"], ward_price)
        ], dtype=float)
        proj_pm2 = np.array([
            self.proj_pm2_map_.get(str(pk), wm)
            for pk, wm in zip(work["project_key"], ward_pm2)
        ], dtype=float)

        num_df = self._build_numeric_frame(work, proj_price, proj_pm2, ward_price, ward_pm2)

        cat_cols = ["sub_district", "apartment_type", "furniture"]
        ohe_mat = self.ohe_.transform(work[cat_cols].fillna("Unknown").astype(str))

        X_tree = np.hstack([num_df.values, ohe_mat])
        num_scaled = self.scaler_.transform(num_df.values)
        X_linear = np.hstack([num_scaled, ohe_mat])
        return X_linear, X_tree

    def _build_numeric_frame(
        self,
        work: pd.DataFrame,
        proj_price: np.ndarray,
        proj_pm2: np.ndarray,
        ward_price: np.ndarray,
        ward_pm2: np.ndarray,
    ) -> pd.DataFrame:
        area = work["area_m2"].astype(float).values
        beds = work["bedrooms"].astype(float).clip(lower=1.0).values
        baths = work["bathrooms"].astype(float).clip(lower=1.0).values
        furn = work["furniture_score"].astype(float).values
        dist = work["distance_to_center_km"].astype(float).values
        pool = work["has_pool"].astype(float).values
        gym = work["has_gym"].astype(float).values
        comm = work["is_commercial_project"].astype(float).values

        return pd.DataFrame({
            "area_m2": area,
            "bedrooms": beds,
            "bathrooms": baths,
            "furniture_score": furn,
            "distance_to_center_km": dist,
            "has_pool": pool,
            "has_gym": gym,
            "is_commercial_project": comm,
            "te_ward_pm2": ward_pm2,
            "te_project_pm2": proj_pm2,
        })

    def _populate_vi_labels(self) -> None:
        base_map = {
            "area_m2": "Diện tích căn hộ (m²)",
            "bedrooms": "Số phòng ngủ",
            "bathrooms": "Số phòng vệ sinh (WC)",
            "furniture_score": "Điểm chất lượng nội thất (0–3)",
            "distance_to_center_km": "Khoảng cách đến Q1 (km)",
            "has_pool": "Tiện ích Hồ bơi",
            "has_gym": "Tiện ích Phòng Gym",
            "is_commercial_project": "Phân khúc Chung cư thương mại",
            "te_ward_pm2": "Vị trí Phường/Xã (Target Encoded)",
            "te_project_pm2": "Đặc trưng Dự án/Tòa nhà (Target Encoded)",
        }
        self.feature_labels_vi_ = {}
        for f in self.feature_names_:
            if f in base_map:
                self.feature_labels_vi_[f] = base_map[f]
            elif f.startswith("sub_district_"):
                self.feature_labels_vi_[f] = "Khu vực: " + f.replace("sub_district_", "")
            elif f.startswith("apartment_type_"):
                self.feature_labels_vi_[f] = "Loại hình: " + f.replace("apartment_type_", "")
            elif f.startswith("furniture_"):
                self.feature_labels_vi_[f] = "Nội thất: " + f.replace("furniture_", "")
            else:
                self.feature_labels_vi_[f] = f

