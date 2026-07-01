"""
Naive seasonal baselines for the demand forecast.

Every model needs a reference point. A seasonal-naive forecast — "repeat the
most recent same-weekday demand" — is the right baseline for M5. This module
implements it in a **horizon-safe** way: predictions for the 28-day validation
window reference only actual demand observed at or before the cutoff (the last
`lag` observed days are tiled across the horizon), so it is a fair comparison
against the horizon-safe LightGBM model.
"""

import json
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from pandas import DataFrame

from src._logging import logger, configure_logging
from src.cli import create_app
from src.model import mape, rmse


ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_METRICS_PATH = OUTPUTS_DIR / "baseline_metrics.json"

GROUP_COLS = ["store_id", "item_id"]
DATE_COL = "date"
VALUE_COL = "sales"


def naive_seasonal_forecast(
    df: DataFrame,
    lag: int = 7,
    val_days: int = 28,
    date_col: str = DATE_COL,
    value_col: str = VALUE_COL,
    group_cols: Optional[list[str]] = None,
) -> DataFrame:
    """
    Horizon-safe seasonal-naive forecast for the last `val_days`.

    For each series, horizon day with offset `o` (0-indexed from the cutoff)
    is predicted by the actual demand `lag` days earlier in the last observed
    `lag`-day block — i.e. the block is tiled across the horizon. Every
    reference date is strictly before the cutoff, so there is no leakage of
    validation-window actuals (unlike a plain `y[t-lag]` which, for horizon
    days beyond the first `lag`, would peek at unobserved demand).

    - `lag=7` repeats the last observed week (same-weekday naive).
    - `lag=28` repeats the last observed 28 days.

    Parameters
    ----------
    df : DataFrame
        Long-format demand with `date`, `sales` and the `group_cols`.
    lag : int
        Seasonal period in days (7 = weekly, 28 = 4-weekly).
    val_days : int
        Held-out horizon length (default 28, matching M5 / the model split).
    group_cols : list[str], optional
        Series identifier columns (default `["store_id", "item_id"]`).

    Returns
    -------
    DataFrame
        The validation rows with a `prediction` column added.
    """
    group_cols = list(group_cols) if group_cols is not None else list(GROUP_COLS)

    cutoff = df[date_col].max() - pd.Timedelta(days=val_days - 1)
    val = df[df[date_col] >= cutoff].copy()
    val["offset"] = (val[date_col] - cutoff).dt.days
    val["ref_date"] = (
        cutoff - pd.Timedelta(days=lag) + pd.to_timedelta(val["offset"] % lag, unit="D")
    )

    # Only the last `lag` observed days can ever be referenced — keep just those.
    ref_lo = cutoff - pd.Timedelta(days=lag)
    hist = df[(df[date_col] >= ref_lo) & (df[date_col] < cutoff)][
        [*group_cols, date_col, value_col]
    ].rename(columns={date_col: "ref_date", value_col: "prediction"})

    val = val.merge(hist, on=[*group_cols, "ref_date"], how="left")
    val["prediction"] = val["prediction"].fillna(0.0)
    return val


# ---- Entry Point -------------------------------------------------------------

app = create_app(name="baselines")


@app.command()
def main(
    features_dir: Annotated[
        Path,
        typer.Option(
            "-f",
            "--features-dir",
            help="Directory with feature parquet files (default: 'data/processed/').",
            file_okay=False,
        ),
    ] = PROCESSED_DATA_DIR,
    metrics_out: Annotated[
        Path,
        typer.Option("--metrics-out", help="Where to save baseline metrics JSON."),
    ] = DEFAULT_METRICS_PATH,
    val_days: Annotated[int, typer.Option("--val-days")] = 28,
    stores: Annotated[
        Optional[str],
        typer.Option("--stores", help="Comma-separated store ids (default: all)."),
    ] = None,
) -> None:
    """
    Compute lag-7 and lag-28 seasonal-naive baselines on the validation window
    and report RMSE / MAPE. Offline: results are saved to JSON, not MLflow.
    """
    configure_logging(name="baselines")
    store_list = [s.strip() for s in stores.split(",")] if stores else None
    df = _load_sales(features_dir, stores=store_list)

    results = {}
    for lag in (7, 28):
        val = naive_seasonal_forecast(df, lag=lag, val_days=val_days)
        results[f"naive_lag_{lag}"] = {
            "rmse": round(rmse(val[VALUE_COL], val["prediction"]), 4),
            "mape": round(mape(val[VALUE_COL], val["prediction"]), 4),
        }
        logger.success(
            f"naive_lag_{lag}: RMSE={results[f'naive_lag_{lag}']['rmse']} "
            f"MAPE={results[f'naive_lag_{lag}']['mape']}%"
        )

    payload = {
        "baselines": results,
        "n_val": int(len(val)),
        "val_days": int(val_days),
        "stores": store_list or "all",
    }
    _save_metrics(payload, metrics_out)
    logger.success(f"Saved baseline metrics -> {metrics_out}")
    return


# ---- Helpers -----------------------------------------------------------------


def _load_sales(features_dir: Path, stores: Optional[list[str]] = None) -> DataFrame:
    """Load only the columns the baseline needs (ids, date, sales) — lightweight."""
    cols = [*GROUP_COLS, DATE_COL, VALUE_COL]
    combined = features_dir / "features.parquet"
    if combined.exists():
        df = pd.read_parquet(combined, columns=cols)
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
        df = pd.concat(
            (pd.read_parquet(p, columns=cols) for p in paths), ignore_index=True
        )

    for col in GROUP_COLS:
        df[col] = df[col].astype("category")
    df[VALUE_COL] = df[VALUE_COL].astype("int32")
    logger.info(f"Loaded sales: {df.shape[0]:,} rows")
    return df


def _save_metrics(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return


if __name__ == "__main__":
    app()
