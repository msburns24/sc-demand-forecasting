import pandas as pd

from src.baselines import naive_seasonal_forecast


def make_sales(n_days=56, mode="weekly", stores=("S1",), items=("I1",)):
    """Long-format demand with a deterministic pattern per (store, item)."""
    rows = []
    for s in stores:
        for it in items:
            for d in range(n_days):
                value = d % 7 if mode == "weekly" else d  # weekly or monotonic ramp
                rows.append(
                    {
                        "store_id": s,
                        "item_id": it,
                        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d),
                        "sales": value,
                    }
                )
    df = pd.DataFrame(rows)
    for col in ["store_id", "item_id"]:
        df[col] = df[col].astype("category")
    return df


class TestNaiveSeasonalForecast:
    def test_output_is_validation_rows(self):
        df = make_sales(n_days=56, stores=("S1", "S2"))
        val = naive_seasonal_forecast(df, lag=7, val_days=28)
        assert len(val) == 2 * 28  # two series x 28 horizon days

    def test_weekly_series_is_perfectly_predicted_by_lag7(self):
        # A purely weekly series repeats exactly, so same-weekday naive is exact.
        df = make_sales(n_days=56, mode="weekly")
        val = naive_seasonal_forecast(df, lag=7, val_days=28)
        assert (val["prediction"] == val["sales"]).all()

    def test_lag28_repeats_last_28_days(self):
        # Monotonic ramp: prediction at offset o is sales 28 days earlier.
        df = make_sales(n_days=84, mode="ramp")
        val = naive_seasonal_forecast(df, lag=28, val_days=28)
        assert (val["prediction"] == val["sales"] - 28).all()

    def test_horizon_safe_ignores_validation_actuals(self):
        df = make_sales(n_days=56, mode="weekly")
        base = naive_seasonal_forecast(df, lag=7, val_days=28)["prediction"].to_numpy()

        # Corrupt the validation-window actuals; predictions must not change.
        cutoff = df["date"].max() - pd.Timedelta(days=27)
        tampered = df.copy()
        tampered.loc[tampered["date"] >= cutoff, "sales"] = 999
        after = naive_seasonal_forecast(tampered, lag=7, val_days=28)["prediction"].to_numpy()

        assert (base == after).all()
        assert (after != 999).all()

    def test_no_cross_series_leakage(self):
        # Two constant series with different levels; each keeps its own level.
        df = make_sales(n_days=56, mode="weekly", stores=("S1", "S2"))
        df.loc[df["store_id"] == "S1", "sales"] = 1
        df.loc[df["store_id"] == "S2", "sales"] = 2
        val = naive_seasonal_forecast(df, lag=7, val_days=28)
        assert (val.loc[val["store_id"] == "S1", "prediction"] == 1).all()
        assert (val.loc[val["store_id"] == "S2", "prediction"] == 2).all()
