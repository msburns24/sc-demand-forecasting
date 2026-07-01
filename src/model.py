"""
Model training utilities: train/val split, LightGBM training, metrics.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike
from pandas import DataFrame

from src._logging import logger


FEATURE_COLS = [
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_35",
    "lag_42",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "rolling_mean_7_lag28",
    "rolling_std_7_lag28",
    "rolling_mean_28_lag28",
    "rolling_std_28_lag28",
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

# Features that are NOT knowable across a full 28-day-ahead horizon at forecast
# time: they depend on sales that fall inside the horizon window. For a genuine
# multi-step forecast made at the cutoff, only lags >= the horizon (28) and the
# non-demand features (calendar, price, ids) are available. Using the columns
# below evaluates a leakage-prone "1-step-ahead with oracle lags" setup, which
# is optimistic. Exclude them for a leakage-free ("horizon-safe") evaluation.
HORIZON_UNSAFE_COLS = [
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
]

HORIZON_SAFE_FEATURE_COLS = [c for c in FEATURE_COLS if c not in HORIZON_UNSAFE_COLS]


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


def train_model(
    X_train: DataFrame,
    y_train: ArrayLike,
    params: dict,
    X_val: DataFrame | None = None,
    y_val: ArrayLike | None = None,
    early_stopping_rounds: int = 50,
) -> LGBMRegressor:
    """
    Train a single global LightGBM regressor across all SKU-store series.

    A single model is trained over every series (the M5-winning approach):
    cross-series information sharing, one artifact to maintain, and graceful
    handling of short/cold-start series.

    The five identifier columns in ``FEATURE_COLS``
    (``item_id``, ``store_id``, ``dept_id``, ``cat_id``, ``state_id``) arrive as
    pandas ``category`` dtype from ``build_features``, so LightGBM detects them
    as categorical automatically — no explicit ``categorical_feature`` list is
    needed. Lag/rolling NaNs (the first 28 days of each series) are handled
    natively by LightGBM.

    Parameters
    ----------
    X_train : DataFrame
        Training feature matrix (columns are ``FEATURE_COLS``).
    y_train : ArrayLike
        Training target (``TARGET_COL``).
    params : dict
        LightGBM hyperparameters, passed to ``LGBMRegressor(**params)``.
    X_val, y_val : DataFrame, ArrayLike, optional
        Validation set. When both are given, training uses early stopping
        against the validation RMSE.
    early_stopping_rounds : int
        Rounds without validation improvement before stopping (default: 50).
        Only used when a validation set is provided.

    Returns
    -------
    LGBMRegressor
        The fitted model.
    """
    model = LGBMRegressor(**params)
    logger.info(f"Training LightGBM on {len(X_train):,} rows, {X_train.shape[1]} features")

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_metric"] = "rmse"
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

    model.fit(X_train, y_train, **fit_kwargs)

    logger.info(f"Trained {model.n_estimators_} trees (best iteration: {model.best_iteration_})")
    return model


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error. The primary metric for this project."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean absolute percentage error, in percent.

    M5 demand is intermittent — many actuals are zero, for which percentage
    error is undefined. Those rows are masked out, so MAPE is reported only
    over non-zero actuals. RMSE (above) is the leakage-free primary metric;
    MAPE is a secondary, business-friendly figure. Returns ``nan`` if every
    actual is zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
