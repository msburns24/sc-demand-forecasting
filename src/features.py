from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from pandas import DataFrame

from src._logging import logger
from src.data.blob_io import upload_to_blob


LOCAL_FEATURES_PATH = Path(__file__).parent.parent / "data/processed/features.parquet"
CALENDAR_COLUMNS = [
    "d",
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]


def build_features(
    df: DataFrame,
    calendar: DataFrame,
    prices: DataFrame,
) -> DataFrame:
    """
    Build the full feature matrix from the raw M5 DataFrames.

    Parameters
    ----------
    df : DataFrame
        Raw sales data (wide format, `sales_train_evaluation.csv`).
    calendar : DataFrame
        Calendar with date, event, and SNAP flags.
    prices : DataFrame
        Sell prices per store/item/week.

    Returns
    -------
    DataFrame
        Long-format feature matrix, one row per item/store/date.
        Input DataFrames are not modified.
    """
    df = df.copy()
    df = (
        df.pipe(_melt_sales)
        .pipe(_merge_calendar, calendar)
        .pipe(_merge_prices, prices)
        .pipe(_impute_sell_prices)
        .pipe(_add_lag_features)
        .pipe(_add_rolling_features)
        .pipe(_add_calendar_features)
        .pipe(_add_price_features)
        .pipe(_encode_categorical_columns)
    )
    logger.info(f"Feature matrix shape: {df.shape}")
    return df


def save_features(df: DataFrame, local_path: Path | None = None) -> None:
    """
    Save feature matrix locally and upload to blob processed container.
    """
    local_path = LOCAL_FEATURES_PATH if local_path is None else local_path
    _ensure_local_directories(local_path)
    _save_to_local_parquet(df, local_path)
    upload_to_blob(local_path, "processed", blob_name="features.parquet")
    logger.success("Complete.")
    return


# ---- Helpers -----------------------------------------------------------------


def _melt_sales(
    df: DataFrame,
    day_column_prefix: str = "d_",
    var_name: str = "d",
    value_name: str = "sales",
) -> DataFrame:
    logger.info("Melting sales to long format...")
    day_columns = [c for c in df.columns if c.startswith(day_column_prefix)]
    return df.melt(
        id_vars=None,  # Infer remaining
        value_vars=day_columns,
        var_name=var_name,
        value_name=value_name,
    )


def _merge_calendar(
    df: DataFrame,
    calendar: DataFrame,
    how: Literal["left", "right", "outer", "inner"] = "left",
    on: str = "d",
    use_cols: Iterable[str] = CALENDAR_COLUMNS,
    date_col: str = "date",
) -> DataFrame:
    logger.info("Merging calendar...")
    cal = calendar[use_cols].copy()
    cal[date_col] = pd.to_datetime(cal[date_col])
    return df.merge(cal, how=how, on=on)


def _merge_prices(
    df: DataFrame,
    prices: DataFrame,
    how: Literal["left", "right", "outer", "inner"] = "left",
    on: str | Iterable[str] = ("store_id", "item_id", "wm_yr_wk"),
) -> DataFrame:
    logger.info("Merging prices...")
    return df.merge(prices, how=how, on=on)


def _impute_sell_prices(
    df: DataFrame,
    price_col: str = "sell_price",
    store_col: str = "store_id",
    item_col: str = "item_id",
    date_col: str = "date",
    ignore_index: bool = True,
) -> DataFrame:
    logger.info("Imputing sell prices...")
    df = df.sort_values([store_col, item_col, date_col])
    df[price_col] = df.groupby([store_col, item_col])[price_col].transform(
        lambda s: s.ffill().bfill()
    )

    item_mean = df.groupby(item_col)[price_col].transform("mean")
    df[price_col] = df[price_col].fillna(item_mean)

    if ignore_index:
        df = df.reset_index(drop=True)
    return df


def _add_lag_features(
    df: DataFrame,
    lags: Iterable[int] = (7, 14, 28),
    value_col: str = "sales",
    prefix: str = "lag_",
    store_col: str = "store_id",
    item_col: str = "item_id",
    observed: bool = True,
) -> DataFrame:
    logger.info("Building lag features...")
    df = df.copy()
    for lag in lags:
        df[f"{prefix}{lag}"] = df.groupby([store_col, item_col], observed=observed)[
            value_col
        ].shift(lag)
    return df


def _add_rolling_features(
    df: DataFrame,
    windows: Iterable[int] = (7, 28),
    value_col: str = "sales",
    mean_prefix: str = "rolling_mean_",
    std_prefix: str = "rolling_std_",
    store_col: str = "store_id",
    item_col: str = "item_id",
) -> DataFrame:
    logger.info("Building rolling features...")
    df = df.copy()
    group_key = df[store_col].astype(str) + "_" + df[item_col].astype(str)
    shifted = df.groupby([store_col, item_col])[value_col].shift(1)
    for window in windows:
        df[f"{mean_prefix}{window}"] = shifted.groupby(group_key).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{std_prefix}{window}"] = shifted.groupby(group_key).transform(
            lambda s: s.rolling(window, min_periods=1).std()
        )
    return df


def _add_calendar_features(df: DataFrame) -> DataFrame:
    logger.info("Building calendar features...")
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["event_type_1"].notna().astype(int)
    df["is_snap_CA"] = df["snap_CA"].astype(int)
    df["is_snap_TX"] = df["snap_TX"].astype(int)
    df["is_snap_WI"] = df["snap_WI"].astype(int)
    return df


def _add_price_features(df: DataFrame) -> DataFrame:
    logger.info("Building price features...")
    df = df.copy()
    df["price_change_pct"] = (
        df.groupby(["store_id", "item_id"])["sell_price"].pct_change().fillna(0)
    )
    df["is_price_decrease"] = df["price_change_pct"].lt(0).astype(int)
    df["is_price_increase"] = df["price_change_pct"].gt(0).astype(int)
    df["price_relative_to_category_mean"] = df["sell_price"].div(
        (df.groupby(["dept_id", "date"])["sell_price"].transform("mean"))
    )
    return df


def _encode_categorical_columns(df: DataFrame) -> DataFrame:
    logger.info("Encoding categoricals...")
    for col in ["item_id", "store_id", "dept_id", "cat_id", "state_id"]:
        df[col] = df[col].astype("category")
    return df


def _ensure_local_directories(local_path: Path) -> None:
    logger.info("Ensuring local directories...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return


def _save_to_local_parquet(df: DataFrame, local_path: Path) -> None:
    logger.info(f"Saving features to {local_path}...")
    df.to_parquet(local_path, index=False)
    return
