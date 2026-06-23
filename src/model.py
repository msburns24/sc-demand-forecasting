"""
Model training utilities: train/val split, LightGBM training.
"""

import pandas as pd
from pandas import DataFrame

from src._logging import logger


FEATURE_COLS = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "day_of_week",
    "week_of_year",
    "month",
    "is_weekend",
    "is_holiday",
    "is_snap_CA",
    "is_snap_TX",
    "is_snap_WI",
    "sell_price",
    "price_change_pct",
    "is_price_decrease",
    "is_price_increase",
    "price_relative_to_category_mean",
    "item_id",
    "store_id",
    "dept_id",
    "cat_id",
    "state_id",
]

TARGET_COL = "sales"


def split_train_val(df: DataFrame, val_days: int = 28) -> tuple[DataFrame, DataFrame]:
    """
    Temporal train/validation split.

    All observations before the cutoff date go to train;
    all observations on or after go to validation.
    No shuffling — order is preserved.

    Parameters
    ----------
    df : DataFrame
        Long-format feature matrix with a 'date' column.
    val_days : int
        Number of days to hold out as validation (default: 28, matching M5).

    Returns
    -------
    (train_df, val_df) : Tuple[DataFrame, DataFrame]
    """
    cutoff = df["date"].max() - pd.Timedelta(days=val_days - 1)
    train = df[df["date"] < cutoff].copy()
    val = df[df["date"] >= cutoff].copy()

    logger.info(f"Split cutoff: {cutoff.date()}")
    logger.info(
        f"Train: {len(train):,} rows, {train['date'].min().date()} "
        f"→ {train['date'].max().date()}"
    )
    logger.info(
        f"Val:   {len(val):,} rows, {val['date'].min().date()} "
        f"→ {val['date'].max().date()}"
    )

    assert len(train) + len(val) == len(df), "Split lost rows"
    assert train["date"].max() < val["date"].min(), "Train/val dates overlap"

    return train, val
