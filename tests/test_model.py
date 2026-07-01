import math

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.model import (
    FEATURE_COLS,
    HORIZON_SAFE_FEATURE_COLS,
    HORIZON_UNSAFE_COLS,
    TARGET_COL,
    mape,
    rmse,
    split_train_val,
    train_model,
)
from src.train import _load_features

CATEGORICAL_COLS = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]

# Fast params for tests: tiny model, single-threaded, deterministic.
SMALL_PARAMS = {
    "n_estimators": 15,
    "num_leaves": 7,
    "min_child_samples": 5,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": -1,
}


def make_df(n_days=60):
    dates = pd.date_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {"date": dates, "sales": range(n_days), "store_id": "S1", "item_id": "I1"}
    )


def make_feature_df(n_items=6, n_days=80, store_id="CA_1", seed=0):
    """
    Build a synthetic feature matrix with every column in ``FEATURE_COLS``,
    the 5 ID columns as ``category`` dtype, and some NaN lag values (mirroring
    the first-28-days behaviour of the real pipeline).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days)
    items = [f"ITEM_{i}" for i in range(n_items)]
    df = pd.DataFrame(
        [(it, d) for it in items for d in dates], columns=["item_id", "date"]
    )
    n = len(df)

    df[TARGET_COL] = rng.integers(0, 10, size=n)
    df["store_id"] = store_id
    df["dept_id"] = "FOODS_1"
    df["cat_id"] = "FOODS"
    df["state_id"] = store_id.split("_")[0]

    for col in [
        "lag_7", "lag_14", "lag_28", "lag_35", "lag_42",
        "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28",
        "rolling_mean_7_lag28", "rolling_std_7_lag28",
        "rolling_mean_28_lag28", "rolling_std_28_lag28",
        "sell_price", "price_change_pct", "price_relative_to_category_mean",
    ]:
        df[col] = rng.normal(size=n)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    for col in [
        "is_holiday", "is_snap_CA", "is_snap_TX", "is_snap_WI",
        "is_price_decrease", "is_price_increase",
    ]:
        df[col] = rng.integers(0, 2, size=n)

    # Early-series NaN lags, as the real feature pipeline produces.
    df.loc[df.index[:5], "lag_28"] = np.nan

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


class TestSplitTrainVal:
    def test_no_overlap(self):
        train, val = split_train_val(make_df(60), val_days=28)
        assert set(train["date"]).isdisjoint(set(val["date"]))

    def test_val_length(self):
        _, val = split_train_val(make_df(60), val_days=28)
        assert val["date"].nunique() == 28

    def test_total_rows_preserved(self):
        df = make_df(60)
        train, val = split_train_val(df, val_days=28)
        assert len(train) + len(val) == len(df)

    def test_train_precedes_val(self):
        train, val = split_train_val(make_df(60), val_days=28)
        assert train["date"].max() < val["date"].min()

    def test_custom_val_days(self):
        _, val = split_train_val(make_df(60), val_days=14)
        assert val["date"].nunique() == 14


class TestTrainModel:
    def test_returns_fitted_regressor(self):
        df = make_feature_df()
        model = train_model(df[FEATURE_COLS], df[TARGET_COL], SMALL_PARAMS)
        assert isinstance(model, LGBMRegressor)
        # A fitted model exposes the booster without raising.
        assert model.booster_ is not None

    def test_predict_length(self):
        df = make_feature_df()
        model = train_model(df[FEATURE_COLS], df[TARGET_COL], SMALL_PARAMS)
        preds = model.predict(df[FEATURE_COLS])
        assert len(preds) == len(df)

    def test_respects_n_estimators_without_early_stopping(self):
        df = make_feature_df()
        params = {**SMALL_PARAMS, "n_estimators": 10}
        model = train_model(df[FEATURE_COLS], df[TARGET_COL], params)
        assert model.n_estimators_ == 10

    def test_deterministic(self):
        df = make_feature_df()
        m1 = train_model(df[FEATURE_COLS], df[TARGET_COL], SMALL_PARAMS)
        m2 = train_model(df[FEATURE_COLS], df[TARGET_COL], SMALL_PARAMS)
        np.testing.assert_allclose(
            m1.predict(df[FEATURE_COLS]), m2.predict(df[FEATURE_COLS])
        )

    def test_handles_category_and_nan(self):
        # Category-dtype IDs and NaN lags must train without raising.
        df = make_feature_df()
        assert df[CATEGORICAL_COLS].dtypes.map(lambda d: d.name == "category").all()
        assert df["lag_28"].isna().any()
        model = train_model(df[FEATURE_COLS], df[TARGET_COL], SMALL_PARAMS)
        assert model.booster_ is not None

    def test_early_stopping_with_validation(self):
        df = make_feature_df(n_days=120)
        train_df, val_df = split_train_val(df, val_days=28)
        model = train_model(
            train_df[FEATURE_COLS],
            train_df[TARGET_COL],
            SMALL_PARAMS,
            val_df[FEATURE_COLS],
            val_df[TARGET_COL],
            early_stopping_rounds=5,
        )
        assert model.best_iteration_ >= 1


class TestMetrics:
    def test_rmse_perfect(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_rmse_known(self):
        # errors 3 and 4 -> sqrt((9 + 16) / 2)
        assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))

    def test_mape_known(self):
        # 10% error on each -> 10.0
        assert mape([100, 200], [110, 180]) == pytest.approx(10.0)

    def test_mape_masks_zero_actuals(self):
        # The zero-actual row is dropped; only the 10% error counts.
        assert mape([0, 100], [5, 110]) == pytest.approx(10.0)

    def test_mape_all_zero_is_nan(self):
        assert math.isnan(mape([0, 0], [5, 10]))


class TestHorizonSafeFeatures:
    def test_excludes_leaky_columns(self):
        for col in HORIZON_UNSAFE_COLS:
            assert col not in HORIZON_SAFE_FEATURE_COLS

    def test_keeps_safe_columns(self):
        # lag_28 is horizon-safe; price/calendar/id features are too.
        assert "lag_28" in HORIZON_SAFE_FEATURE_COLS
        assert "sell_price" in HORIZON_SAFE_FEATURE_COLS
        assert "item_id" in HORIZON_SAFE_FEATURE_COLS

    def test_is_strict_subset(self):
        assert set(HORIZON_SAFE_FEATURE_COLS) < set(FEATURE_COLS)


class TestLoadFeatures:
    def test_concats_per_store(self, tmp_path):
        df1 = make_feature_df(n_items=2, n_days=10, store_id="CA_1")
        df2 = make_feature_df(n_items=2, n_days=10, store_id="CA_2")
        df1.to_parquet(tmp_path / "features_CA_1.parquet", index=False)
        df2.to_parquet(tmp_path / "features_CA_2.parquet", index=False)

        out = _load_features(tmp_path)
        assert len(out) == len(df1) + len(df2)
        # Concatenation must restore category dtype for LightGBM.
        assert out["store_id"].dtype.name == "category"
        assert set(out["store_id"].unique()) == {"CA_1", "CA_2"}

    def test_stores_filter(self, tmp_path):
        make_feature_df(n_items=2, n_days=10, store_id="CA_1").to_parquet(
            tmp_path / "features_CA_1.parquet", index=False
        )
        make_feature_df(n_items=2, n_days=10, store_id="CA_2").to_parquet(
            tmp_path / "features_CA_2.parquet", index=False
        )
        out = _load_features(tmp_path, stores=["CA_1"])
        assert set(out["store_id"].unique()) == {"CA_1"}

    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_features(tmp_path)
