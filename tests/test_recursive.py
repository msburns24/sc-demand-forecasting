import numpy as np
import pandas as pd

from src.baselines import naive_seasonal_forecast
from src.recursive import recursive_forecast

RECOMPUTE = [
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28",
]


def make_df(n_days=84, mode="weekly", stores=("S1", "S2")):
    rows = []
    for s in stores:
        for d in range(n_days):
            value = d % 7 if mode == "weekly" else d
            rows.append(
                {
                    "store_id": s,
                    "item_id": "I1",
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d),
                    "sales": float(value),
                }
            )
    df = pd.DataFrame(rows)
    for col in RECOMPUTE:
        df[col] = 0.0  # placeholder; recursive_forecast overwrites these
    for col in ["store_id", "item_id"]:
        df[col] = df[col].astype("category")
    return df


class _DummyLag7Model:
    """Predicts exactly lag_7 — turns recursion into a seasonal-naive tiler."""

    def predict(self, X):
        return X["lag_7"].to_numpy()


class TestRecursiveForecast:
    def test_matches_naive_lag7_for_lag7_model(self):
        # A lag_7 'model' rolled recursively must reproduce the horizon-safe
        # lag-7 seasonal-naive forecast — validates the recompute + alignment.
        df = make_df(mode="weekly")
        rec = recursive_forecast(_DummyLag7Model(), df, RECOMPUTE, val_days=28)
        naive = naive_seasonal_forecast(df, lag=7, val_days=28).copy()
        naive["_series"] = naive["store_id"].astype(str) + "|" + naive["item_id"].astype(str)

        rec_k = rec.sort_values(["_series", "date"]).reset_index(drop=True)
        nai_k = naive.sort_values(["_series", "date"]).reset_index(drop=True)
        np.testing.assert_allclose(
            rec_k["prediction"].to_numpy(), nai_k["prediction"].to_numpy()
        )

    def test_output_shape_and_nonnegative(self):
        df = make_df()
        rec = recursive_forecast(_DummyLag7Model(), df, ["lag_7"], val_days=28)
        assert len(rec) == 2 * 28
        assert (rec["prediction"] >= 0).all()
