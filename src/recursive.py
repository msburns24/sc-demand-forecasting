"""
Recursive multi-step evaluation of the LightGBM demand model (SCDF-18 follow-up).

The horizon-safe direct model (``python src/train.py``) is honest but throws
away short lags (``lag_7``/``lag_14``) and ``shift(1)`` rolling features, because
they aren't knowable across a 28-day horizon. Recursion recovers them the
legitimate way: predict day 1, feed that prediction back as history, recompute
the short-lag / rolling features, predict day 2, and so on across the horizon.

This measures the true multi-step performance of the *full* feature set and
answers whether recursion beats the ~2.17 RMSE horizon-safe direct model.
"""

import sys
import time
from pathlib import Path
from typing import Annotated, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402
from pandas import DataFrame  # noqa: E402

from src._logging import logger, configure_logging  # noqa: E402
from src.cli import create_app  # noqa: E402
from src.model import (  # noqa: E402
    FEATURE_COLS,
    TARGET_COL,
    mape,
    rmse,
    train_model,
)
from src.train import PROCESSED_DATA_DIR, _load_config, _load_features  # noqa: E402


ROOT_DIR = _ROOT
CONFIG_PATH = ROOT_DIR / "config" / "model_config.yaml"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "recursive_metrics.json"

# Max history lookback needed to recompute the recursive features (lag_28 /
# rolling_28). Everything else (lag_35+, shift(28) rollings, calendar, price)
# is horizon-safe and taken straight from the feature table.
HISTORY_BACK = 28


def recursive_forecast(
    model,
    df: DataFrame,
    feature_cols: list[str],
    val_days: int = 28,
    date_col: str = "date",
    value_col: str = TARGET_COL,
    group_cols: tuple[str, str] = ("store_id", "item_id"),
) -> DataFrame:
    """
    Roll ``model`` forward ``val_days`` days, feeding each day's prediction back
    into the demand-derived features before predicting the next day.

    Only the leakage-prone features are recomputed from the evolving history
    (``lag_7/14/28`` and the ``shift(1)`` rolling mean/std over 7 and 28 days);
    all other columns are used as-is from ``df`` (they are already horizon-safe
    or externally known). Predictions are clipped at zero (demand is
    non-negative).

    Returns the validation rows with a ``prediction`` column.
    """
    sa, sb = group_cols
    cutoff = df[date_col].max() - pd.Timedelta(days=val_days - 1)
    lo = cutoff - pd.Timedelta(days=HISTORY_BACK)

    # Only the small val / history windows need a per-series key (avoid
    # materialising a 59M-row string column on the full frame).
    val = df[df[date_col] >= cutoff].copy()
    val["_series"] = val[sa].astype(str) + "|" + val[sb].astype(str)
    val = val.sort_values(["_series", date_col]).reset_index(drop=True)
    counts = val.groupby("_series", observed=True).size()
    assert (counts == val_days).all(), "Every series must have exactly val_days rows"
    series_order = val["_series"].to_numpy()[::val_days]
    n = len(series_order)

    # History matrix S[series, day]; days 0..HISTORY_BACK-1 are pre-cutoff
    # actuals, days HISTORY_BACK.. are filled with predictions as we roll.
    hist_dates = pd.date_range(lo, periods=HISTORY_BACK)
    hist = df[(df[date_col] >= lo) & (df[date_col] < cutoff)].copy()
    hist["_series"] = hist[sa].astype(str) + "|" + hist[sb].astype(str)
    actuals = (
        hist.pivot_table(index="_series", columns=date_col, values=value_col, aggfunc="first")
        .reindex(index=series_order, columns=hist_dates)
    )
    S = np.zeros((n, HISTORY_BACK + val_days), dtype=np.float64)
    S[:, :HISTORY_BACK] = np.nan_to_num(actuals.to_numpy(dtype=float), nan=0.0)

    # column -> function of (S, j) recomputing that feature for horizon day j
    recompute = {
        "lag_7": lambda S, j: S[:, j - 7],
        "lag_14": lambda S, j: S[:, j - 14],
        "lag_28": lambda S, j: S[:, j - 28],
        "rolling_mean_7": lambda S, j: S[:, j - 7:j].mean(axis=1),
        "rolling_mean_28": lambda S, j: S[:, j - 28:j].mean(axis=1),
        "rolling_std_7": lambda S, j: S[:, j - 7:j].std(axis=1, ddof=1),
        "rolling_std_28": lambda S, j: S[:, j - 28:j].std(axis=1, ddof=1),
    }

    preds = np.empty(len(val), dtype=float)
    for h in range(val_days):
        j = HISTORY_BACK + h
        rows = val.iloc[h::val_days]  # one row per series, in series_order
        X = rows[feature_cols].copy()
        for col, fn in recompute.items():
            if col in X.columns:
                X[col] = fn(S, j)
        yhat = np.clip(model.predict(X), 0.0, None)
        S[:, j] = yhat
        preds[h::val_days] = yhat

    val["prediction"] = preds
    return val


# ---- Entry Point -------------------------------------------------------------

app = create_app(name="recursive")


@app.command()
def main(
    config: Annotated[Path, typer.Option("-c", "--config", exists=True, dir_okay=False)] = CONFIG_PATH,
    features_dir: Annotated[Path, typer.Option("-f", "--features-dir", file_okay=False)] = PROCESSED_DATA_DIR,
    metrics_out: Annotated[Path, typer.Option("--metrics-out")] = DEFAULT_METRICS_PATH,
    val_days: Annotated[int, typer.Option("--val-days")] = 28,
    stores: Annotated[Optional[str], typer.Option("--stores")] = None,
) -> None:
    """
    Train a 1-step model on the full feature set (pre-cutoff data only, with an
    inner holdout for early stopping) and evaluate it recursively over the
    28-day horizon. Offline: metrics saved to JSON.
    """
    configure_logging(name="recursive")
    params, training_cfg = _load_config(config)
    early = int(training_cfg.get("early_stopping_rounds", 50))
    store_list = [s.strip() for s in stores.split(",")] if stores else None

    df = _load_features(features_dir, stores=store_list)

    cutoff = df["date"].max() - pd.Timedelta(days=val_days - 1)
    lo = cutoff - pd.Timedelta(days=HISTORY_BACK)
    inner_cutoff = cutoff - pd.Timedelta(days=val_days)

    # Recursion only needs the last HISTORY_BACK + val_days days; keep just that
    # window and drop the full 59M-row frame before training to bound peak memory.
    recursion_df = df[df["date"] >= lo].copy()

    # Inner holdout (last val_days before the real cutoff) for early stopping,
    # so the real validation window never touches training. Select only the
    # needed columns to avoid a second full-frame copy.
    tr = df["date"] < inner_cutoff
    iv = (df["date"] >= inner_cutoff) & (df["date"] < cutoff)
    X_tr, y_tr = df.loc[tr, FEATURE_COLS], df.loc[tr, TARGET_COL]
    X_iv, y_iv = df.loc[iv, FEATURE_COLS], df.loc[iv, TARGET_COL]
    del df

    logger.info(f"Training 1-step model on full feature set ({len(FEATURE_COLS)} features)")
    start = time.perf_counter()
    model = train_model(X_tr, y_tr, params, X_iv, y_iv, early)
    train_seconds = time.perf_counter() - start
    del X_tr, y_tr, X_iv, y_iv

    logger.info("Rolling model forward recursively over the horizon...")
    val_pred = recursive_forecast(model, recursion_df, FEATURE_COLS, val_days=val_days)

    metrics = {
        "rmse": round(rmse(val_pred[TARGET_COL], val_pred["prediction"]), 4),
        "mape": round(mape(val_pred[TARGET_COL], val_pred["prediction"]), 4),
        "n_val": int(len(val_pred)),
        "n_features": len(FEATURE_COLS),
        "method": "recursive_multistep",
        "val_days": int(val_days),
        "stores": store_list or "all",
        "train_seconds": round(train_seconds, 1),
    }
    _save_metrics(metrics, metrics_out)
    logger.success(f"Recursive multi-step RMSE={metrics['rmse']} MAPE={metrics['mape']}%")
    logger.success(f"Saved -> {metrics_out}")
    return


def _save_metrics(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return


if __name__ == "__main__":
    app()
