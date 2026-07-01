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
from src.train import (  # noqa: E402
    CATEGORICAL_COLS,
    LOAD_COLS,
    PROCESSED_DATA_DIR,
    _downcast_numeric,
    _load_config,
)


ROOT_DIR = _ROOT
CONFIG_PATH = ROOT_DIR / "config" / "model_config.yaml"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "recursive_metrics.json"
# Per-row validation predictions, consumed by src.evaluation (SCDF-20) for the
# per-segment comparison without re-running this ~18-min pipeline.
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "recursive_val_predictions.parquet"
PRED_COLS = ["store_id", "item_id", "date", "sales", "prediction"]

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
    sample_frac: Annotated[
        float,
        typer.Option(
            "--sample-frac",
            help="Fraction of training rows to fit on (<1.0 bounds the fit memory).",
        ),
    ] = 1.0,
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

    X_tr, y_tr, X_iv, y_iv, recursion_df = _load_recursion_data(
        features_dir, store_list, val_days, sample_frac=sample_frac
    )

    logger.info(f"Training 1-step model on full feature set ({len(FEATURE_COLS)} features)")
    n_train = len(X_tr)
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
        "sample_frac": sample_frac,
        "n_train": int(n_train),
        "train_seconds": round(train_seconds, 1),
    }
    _save_metrics(metrics, metrics_out)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    val_pred[PRED_COLS].to_parquet(PREDICTIONS_PATH, index=False)

    logger.success(f"Recursive multi-step RMSE={metrics['rmse']} MAPE={metrics['mape']}%")
    logger.success(f"Saved metrics -> {metrics_out}")
    logger.success(f"Saved predictions -> {PREDICTIONS_PATH}")
    return


def _load_recursion_data(features_dir: Path, stores, val_days: int, sample_frac: float = 1.0):
    """
    Load training / inner-holdout / recursion frames per store, never holding the
    full 59M-row matrix plus a big copy at once.

    On this machine the Windows commit limit (~physical RAM, no page file) is the
    real ceiling, not resident memory — so we build the training matrices by
    concatenating per-store slices rather than slicing one giant frame. Returns
    `(X_tr, y_tr, X_iv, y_iv, recursion_df)`.

    `sample_frac < 1.0` randomly subsamples the *training* rows (per store, seed
    42) to bound the LightGBM fit-array size; the recursion window is never
    subsampled, so every series is still forecast across the full horizon.
    """
    paths = sorted(features_dir.glob("features_*.parquet"))
    if stores:
        wanted = {f"features_{s}.parquet" for s in stores}
        paths = [p for p in paths if p.name in wanted]
    if not paths:
        raise FileNotFoundError(
            f"No feature parquet files found in '{features_dir}'. "
            "Build them first with `python -m src.features`."
        )

    # All stores share the M5 calendar, so the cutoff is global.
    max_date = pd.read_parquet(paths[0], columns=["date"])["date"].max()
    cutoff = max_date - pd.Timedelta(days=val_days - 1)
    lo = cutoff - pd.Timedelta(days=HISTORY_BACK)
    inner_cutoff = cutoff - pd.Timedelta(days=val_days)

    xtr, ytr, xiv, yiv, rec = [], [], [], [], []
    for p in paths:
        d = _downcast_numeric(pd.read_parquet(p, columns=LOAD_COLS))
        tr = d["date"] < inner_cutoff
        iv = (d["date"] >= inner_cutoff) & (d["date"] < cutoff)
        tr_x = d.loc[tr, FEATURE_COLS]
        tr_y = d.loc[tr, TARGET_COL]
        if sample_frac < 1.0:
            tr_x = tr_x.sample(frac=sample_frac, random_state=42)
            tr_y = tr_y.loc[tr_x.index]
        xtr.append(tr_x)
        ytr.append(tr_y)
        xiv.append(d.loc[iv, FEATURE_COLS])
        yiv.append(d.loc[iv, TARGET_COL])
        rec.append(d[d["date"] >= lo])
        del d

    X_tr = pd.concat(xtr, ignore_index=True)
    del xtr
    y_tr = pd.concat(ytr, ignore_index=True)
    X_iv = pd.concat(xiv, ignore_index=True)
    y_iv = pd.concat(yiv, ignore_index=True)
    recursion_df = pd.concat(rec, ignore_index=True)

    # Concat collapses mismatched per-file category sets to object — re-cast.
    for frame in (X_tr, X_iv, recursion_df):
        for col in CATEGORICAL_COLS:
            frame[col] = frame[col].astype("category")

    logger.info(f"Loaded train={len(X_tr):,} inner_val={len(X_iv):,} recursion={len(recursion_df):,}")
    return X_tr, y_tr, X_iv, y_iv, recursion_df


def _save_metrics(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return


if __name__ == "__main__":
    app()
