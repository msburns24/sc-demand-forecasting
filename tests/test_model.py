import pandas as pd
from src.model import split_train_val


def make_df(n_days=60):
    dates = pd.date_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {"date": dates, "sales": range(n_days), "store_id": "S1", "item_id": "I1"}
    )


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
