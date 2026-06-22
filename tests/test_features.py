from unittest.mock import patch
import pandas as pd
from src.features import (
    save_features,
    _add_lag_features,
    _add_rolling_features,
)


def make_long_df(n_items=2, n_stores=2, n_days=40):
    """Minimal long-format DataFrame for testing feature helpers."""
    rows = []
    for store in [f"S{i}" for i in range(n_stores)]:
        for item in [f"I{i}" for i in range(n_items)]:
            for day in range(n_days):
                rows.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "dept_id": "DEPT_1",
                        "cat_id": "CAT_1",
                        "state_id": "CA",
                        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                        "wm_yr_wk": 11000 + day // 7,
                        "sales": float(day % 7 + 1),
                        "sell_price": 2.99,
                        "event_type_1": None,
                        "event_type_2": None,
                        "snap_CA": day % 3 == 0,
                        "snap_TX": False,
                        "snap_WI": False,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )


class TestLagFeatures:
    def test_first_7_days_are_nan(self):
        df = make_long_df(n_days=40)
        out = _add_lag_features(df.copy())
        first_7 = out[out["date"] < pd.Timestamp("2020-01-08")]
        assert first_7["lag_7"].isna().all()

    def test_lag_7_correct_value(self):
        df = make_long_df(n_items=1, n_stores=1, n_days=40)
        out = _add_lag_features(df.copy())
        # Row at index 7 should have lag_7 == row 0's sales
        assert out.loc[7, "lag_7"] == out.loc[0, "sales"]

    def test_no_cross_series_leakage(self):
        """Lag values must not bleed across item-store boundaries."""
        df = make_long_df(n_items=2, n_stores=1, n_days=40)
        out = _add_lag_features(df.copy())
        # The first 7 rows of the second series should still be NaN
        series_2_start = out[out["item_id"] == "I1"].head(7)
        assert series_2_start["lag_7"].isna().all()


class TestRollingFeatures:
    def test_rolling_mean_non_negative(self):
        df = make_long_df()
        out = _add_rolling_features(df.copy())
        assert (out["rolling_mean_7"].dropna() >= 0).all()

    def test_rolling_uses_shifted_sales(self):
        """rolling_mean_7 on day t must not include day t's own sales."""
        df = make_long_df(n_items=1, n_stores=1, n_days=40)
        # Set day 7 to a huge spike
        df.loc[df["date"] == pd.Timestamp("2020-01-08"), "sales"] = 999
        out = _add_rolling_features(df.copy())
        # rolling_mean_7 on that day should NOT include 999
        spike_row = out[out["date"] == pd.Timestamp("2020-01-08")]
        assert spike_row["rolling_mean_7"].iloc[0] != 999


class TestSaveFeatures:
    @patch("src.features.upload_to_blob")
    def test_blob_upload_called_with_processed_container(self, mock_upload, tmp_path):
        df = make_long_df()
        out_path = tmp_path / "features.parquet"
        save_features(df, local_path=out_path)
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args
        assert call_args.args[1] == "processed"
        assert call_args.kwargs.get("blob_name") == "features.parquet"
