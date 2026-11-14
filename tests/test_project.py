import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import config
from src.data_validation import load_validated_data, validate_apartments, validate_timeseries
from src.model.feature_pipeline import LeakFreeFeatureBuilder, create_stratified_split
from src.model.ml_models import compute_metrics
from src.model.ts_models import compute_ts_metrics, run_timeseries_modeling_suite


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.apartments, cls.timeseries = load_validated_data()

    def test_validation_rejects_invalid_inputs(self):
        bad = self.apartments.copy()
        bad.loc[0, "area_m2"] = 0
        with self.assertRaises(ValueError):
            validate_apartments(bad)
        bad = self.timeseries.drop(index=10).reset_index(drop=True)
        with self.assertRaises(ValueError):
            validate_timeseries(bad)

    def test_split_and_target_encoding_isolation(self):
        train, test = create_stratified_split(self.apartments)
        self.assertTrue(set(train.listing_id).isdisjoint(test.listing_id))
        for fit_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(train):
            fit = train.iloc[fit_idx].reset_index(drop=True)
            validation = train.iloc[val_idx].reset_index(drop=True)
            builder = LeakFreeFeatureBuilder()
            builder.fit_transform(fit)
            original = builder.transform(validation)[1]
            changed = validation.copy()
            changed["price_million_vnd"] *= 100
            changed["price_per_m2"] *= 100
            np.testing.assert_array_equal(original, builder.transform(changed)[1])

    def test_directory_setup_preserves_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep = root / "raw" / "keep.txt"
            keep.parent.mkdir()
            keep.write_text("keep", encoding="utf-8")
            dirs = {name: root / name for name in ["PHASE1_TABLES_DIR", "PHASE1_FIGURES_DIR",
                    "MODELING_TABLES_DIR", "MODELING_FIGURES_DIR", "PROCESSED_DATA_DIR",
                    "TS_TABLES_DIR", "TS_FIGURES_DIR"]}
            with patch.multiple(config, DATA_DIR=root, **dirs):
                config.ensure_phase1_dirs()
                config.ensure_modeling_dirs()
                config.ensure_ts_dirs()
            self.assertEqual(keep.read_text(encoding="utf-8"), "keep")
            self.assertTrue(all(path.is_dir() for path in dirs.values()))


class MetricsAndForecastTests(unittest.TestCase):
    def test_metrics_units_and_negative_r2(self):
        actual = np.array([2.0, 4.0])
        pred = np.array([4.0, 2.0])
        for metric in (compute_metrics, compute_ts_metrics):
            result = metric(actual, pred)
            self.assertEqual(result["MAE"], 2.0)
            self.assertEqual(result["RMSE"], 2.0)
            self.assertEqual(result["MAPE"], 0.75)
            self.assertEqual(result["R2"], -3.0)

    def test_rolling_forecast_only_uses_past_and_selects_validation(self):
        _, df = load_validated_data()
        calls = []
        def fake_sarimax(history, steps, seasonal, exogenous):
            calls.append((str(history.month.iloc[-1]), steps))
            return np.repeat(history.rent_price_million_vnd.iloc[-1] + 10, steps)
        def fake_lstm(frame, train_len, **kwargs):
            self.assertEqual(len(frame), train_len + 1)
            return np.array([frame.rent_price_million_vnd.iloc[train_len - 1] + 10])
        with tempfile.TemporaryDirectory() as tmp, \
             patch("src.model.ts_models._sarimax_forecast", side_effect=fake_sarimax), \
             patch("src.model.ts_models._fit_predict_lstm", side_effect=fake_lstm):
            result = run_timeseries_modeling_suite(df, Path(tmp), Path(tmp))
            selected = result["selected_model"]
            metrics = result["rolling_metrics_df"]
            validation = metrics[metrics.period == "validation"].sort_values("RMSE")
            self.assertEqual(selected, validation.iloc[0]["Mô_hình"])
            self.assertEqual(result["future_df"].Model_Used.unique().tolist(), [selected])
            self.assertEqual(len(result["future_df"]), 12)
            self.assertEqual(calls[0], ("2022-12", 1))
            self.assertEqual(calls[-1], ("2024-11", 1))
            rolling = pd.read_csv(Path(tmp) / "ts8_rolling_predictions_2023_2024.csv")
            naive = rolling[(rolling.model == "Naive (tháng trước)") & (rolling.month == "2023-01")]
            self.assertAlmostEqual(naive.prediction.iloc[0], df.rent_price_million_vnd.iloc[47])


if __name__ == "__main__":
    unittest.main()
