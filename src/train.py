"""
Offline training entry point for the demand-forecasting LightGBM model.

Reads the engineered feature matrix from local parquet (built by
``python -m src.features``), performs a temporal train/validation split, trains a
single global LightGBM model, evaluates it on the held-out window, and writes the
fitted model and metrics to disk. No cloud / Azure ML dependency.

Run with either:

    python src/train.py            # script mode (AC)
    python -m src.train            # module mode
"""

import sys
import time
from pathlib import Path
from typing import Annotated, Optional

# Allow `python src/train.py` (script mode): ensure the repo root is importable
# for the absolute `from src...` imports below. Harmless under `-m src.train`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json  # noqa: E402

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402
import yaml  # noqa: E402
from pandas import DataFrame  # noqa: E402

from src._logging import logger, configure_logging  # noqa: E402
from src.cli import create_app  # noqa: E402
from src.model import (  # noqa: E402
    FEATURE_COLS,
    HORIZON_SAFE_FEATURE_COLS,
    TARGET_COL,
    mape,
    rmse,
    split_train_val,
    train_model,
)


ROOT_DIR = _ROOT
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
CONFIG_PATH = ROOT_DIR / "config" / "model_config.yaml"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "lgbm_v1.pkl"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "metrics.json"

# Re-applied after concatenating per-store parquet files: concat unions
# mismatched per-file category sets back to `object`, which LightGBM rejects.
CATEGORICAL_COLS = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]

# Only these columns are read from parquet. The feature files also carry heavy
# string columns (id, d, event_name_*, etc.) that the model never uses; reading
# all 40 columns for the full 59M-row dataset exhausts memory, so we project
# down to exactly what training needs (features + target + the split key).
LOAD_COLS = FEATURE_COLS + ["date", TARGET_COL]


# ---- Entry Point -------------------------------------------------------------

app = create_app(name="train")


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config",
            help="Path to model config YAML (default: 'config/model_config.yaml').",
            dir_okay=False,
            exists=True,
        ),
    ] = CONFIG_PATH,
    features_dir: Annotated[
        Path,
        typer.Option(
            "-f",
            "--features-dir",
            help="Directory with feature parquet files (default: 'data/processed/').",
            file_okay=False,
        ),
    ] = PROCESSED_DATA_DIR,
    model_out: Annotated[
        Path,
        typer.Option("--model-out", help="Where to save the fitted model pickle."),
    ] = DEFAULT_MODEL_PATH,
    metrics_out: Annotated[
        Path,
        typer.Option("--metrics-out", help="Where to save the metrics JSON."),
    ] = DEFAULT_METRICS_PATH,
    val_days: Annotated[
        Optional[int],
        typer.Option("--val-days", help="Validation horizon in days (overrides config)."),
    ] = None,
    stores: Annotated[
        Optional[str],
        typer.Option(
            "--stores",
            help="Comma-separated store ids to train on, e.g. 'CA_1,TX_1' (default: all).",
        ),
    ] = None,
    max_rows: Annotated[
        Optional[int],
        typer.Option("--max-rows", help="Cap total rows for a fast smoke run (default: all)."),
    ] = None,
    horizon_safe: Annotated[
        bool,
        typer.Option(
            "--horizon-safe/--full-features",
            help=(
                "Horizon-safe (default) drops features that leak across the 28-day "
                "horizon (lag_7/14, shift-1 rolling_*). --full-features keeps them for "
                "the optimistic, leakage-prone comparison only."
            ),
        ),
    ] = True,
) -> None:
    """
    Train the global LightGBM demand-forecasting model offline.

    Loads features from `data/processed/`, splits temporally, trains, evaluates
    (RMSE / MAPE on the held-out window), and saves the model + metrics locally.
    """
    configure_logging(name="train")

    params, training_cfg = _load_config(config)
    val_days = val_days if val_days is not None else int(training_cfg.get("val_days", 28))
    early_stopping_rounds = int(training_cfg.get("early_stopping_rounds", 50))
    store_list = [s.strip() for s in stores.split(",")] if stores else None

    df = _load_features(features_dir, stores=store_list, max_rows=max_rows)

    feature_cols = HORIZON_SAFE_FEATURE_COLS if horizon_safe else FEATURE_COLS
    logger.info(
        f"Feature set: {'horizon-safe' if horizon_safe else 'full'} "
        f"({len(feature_cols)} features)"
    )

    train_df, val_df = split_train_val(df, val_days=val_days)
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]

    start = time.perf_counter()
    model = train_model(X_train, y_train, params, X_val, y_val, early_stopping_rounds)
    train_seconds = time.perf_counter() - start

    preds = model.predict(X_val)
    metrics = {
        "rmse": round(rmse(y_val, preds), 4),
        "mape": round(mape(y_val, preds), 4),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_features": len(feature_cols),
        "feature_set": "horizon_safe" if horizon_safe else "full",
        "best_iteration": int(model.best_iteration_ or model.n_estimators_),
        "val_days": int(val_days),
        "stores": store_list or "all",
        "train_seconds": round(train_seconds, 1),
    }

    _save_model(model, model_out)
    _save_metrics(metrics, metrics_out)
    _log_summary(metrics, model_out, metrics_out)
    return


# ---- Helpers -----------------------------------------------------------------


def _load_config(path: Path) -> tuple[dict, dict]:
    """Load YAML config; return (model_params, training_settings)."""
    logger.info(f"Loading config: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("model", {}), cfg.get("training", {})


def _load_features(
    features_dir: Path,
    stores: Optional[list[str]] = None,
    max_rows: Optional[int] = None,
) -> DataFrame:
    """
    Load the feature matrix from local parquet (fully offline).

    Prefers a single combined `features.parquet`; otherwise concatenates the
    per-store `features_{store_id}.parquet` files that `src.features` writes.
    """
    combined = features_dir / "features.parquet"
    if combined.exists():
        logger.info(f"Loading combined features: {combined}")
        df = _downcast_numeric(pd.read_parquet(combined, columns=LOAD_COLS))
        if stores:
            df = df[df["store_id"].isin(stores)].copy()
    else:
        paths = sorted(features_dir.glob("features_*.parquet"))
        if stores:
            wanted = {f"features_{s}.parquet" for s in stores}
            paths = [p for p in paths if p.name in wanted]
        if not paths:
            raise FileNotFoundError(
                f"No feature parquet files found in '{features_dir}'. "
                "Build them first with `python -m src.features`."
            )
        logger.info(f"Loading {len(paths)} per-store feature file(s)...")
        df = pd.concat(
            (_downcast_numeric(pd.read_parquet(p, columns=LOAD_COLS)) for p in paths),
            ignore_index=True,
        )

    if df.empty:
        raise ValueError("Loaded feature set is empty — check the --stores values.")

    # Restore category dtype (concat collapses mismatched per-file categories).
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if max_rows and len(df) > max_rows:
        logger.info(f"Subsetting to first {max_rows:,} of {len(df):,} rows (smoke run).")
        df = df.head(max_rows).copy()

    logger.info(f"Feature matrix: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def _downcast_numeric(df: DataFrame) -> DataFrame:
    """
    Downcast 64-bit numeric columns to 32-bit, in place, column by column.

    Applied per parquet file *before* concatenation so we never materialise a
    consolidated 64-bit block for the full ~59M-row dataset (that intermediate
    is what exhausts memory). float32 precision is ample for these features
    (LightGBM bins them internally) and `sales` fits comfortably in int32.
    """
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def _save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved model: {path}")
    return


def _save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Saved metrics: {path}")
    return


def _log_summary(metrics: dict, model_out: Path, metrics_out: Path) -> None:
    logger.success("Training complete.")
    logger.success(f"  Validation RMSE : {metrics['rmse']}")
    logger.success(f"  Validation MAPE : {metrics['mape']}% (non-zero actuals)")
    logger.success(f"  Train / Val rows: {metrics['n_train']:,} / {metrics['n_val']:,}")
    logger.success(f"  Best iteration  : {metrics['best_iteration']}")
    logger.success(f"  Train time      : {metrics['train_seconds']}s")
    logger.success(f"  Model  -> {model_out}")
    logger.success(f"  Metrics-> {metrics_out}")
    return


if __name__ == "__main__":
    app()
