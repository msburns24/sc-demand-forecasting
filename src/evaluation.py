"""
Per-ABC-XYZ-segment evaluation of the demand forecast (SCDF-20).

Aggregate RMSE/MAPE hides *where* the model is good or bad. This module breaks
validation error down by ABC-XYZ cell (9 cells) and compares the direct
horizon-safe model, the recursive multi-step model, and the naive seasonal
baseline per cell — the finding that shows AX (high-value, stable) items forecast
best and CZ (low-value, erratic) worst. Offline: results saved to CSV/JSON.
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json  # noqa: E402

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402
from pandas import DataFrame  # noqa: E402

from src._logging import logger, configure_logging  # noqa: E402
from src.baselines import naive_seasonal_forecast  # noqa: E402
from src.cli import create_app  # noqa: E402
from src.model import HORIZON_SAFE_FEATURE_COLS, TARGET_COL, mape, rmse  # noqa: E402
from src.segmentation import classify_mts_mto, segment_skus  # noqa: E402
from src.train import DEFAULT_MODEL_PATH, PROCESSED_DATA_DIR, _load_features  # noqa: E402


ROOT_DIR = _ROOT
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_TABLE_PATH = OUTPUTS_DIR / "segment_performance.csv"
DEFAULT_JSON_PATH = OUTPUTS_DIR / "segment_performance.json"
# Kept in sync with src.recursive.PREDICTIONS_PATH.
RECURSIVE_PRED_PATH = PROCESSED_DATA_DIR / "recursive_val_predictions.parquet"

ID_COLS = ["store_id", "item_id"]


def segment_metrics(
    pred_df: DataFrame,
    segments: DataFrame,
    y_true_col: str = TARGET_COL,
    y_pred_col: str = "prediction",
) -> DataFrame:
    """
    Per-ABC-XYZ-cell RMSE / MAPE for one set of per-row predictions.

    Merges the `(store_id, item_id) -> abc_xyz` labels onto `pred_df`, then groups
    by cell. Returns one row per cell with `abc_xyz`, `n_skus`, `n_obs`, `rmse`,
    `mape`. (MAPE is over non-zero actuals — see `src.model.mape`.)
    """
    merged = pred_df.merge(segments[[*ID_COLS, "abc_xyz"]], on=ID_COLS, how="left")
    rows = []
    for cell, g in merged.groupby("abc_xyz", observed=True):
        rows.append(
            {
                "abc_xyz": cell,
                "n_skus": int(g.groupby(ID_COLS, observed=True).ngroups),
                "n_obs": int(len(g)),
                "rmse": round(rmse(g[y_true_col], g[y_pred_col]), 4),
                "mape": round(mape(g[y_true_col], g[y_pred_col]), 4),
            }
        )
    return pd.DataFrame(rows)


# ---- Entry Point -------------------------------------------------------------

app = create_app(name="evaluation")


@app.command()
def main(
    features_dir: Annotated[
        Path, typer.Option("-f", "--features-dir", file_okay=False)
    ] = PROCESSED_DATA_DIR,
    model_path: Annotated[
        Path, typer.Option("--model", dir_okay=False)
    ] = DEFAULT_MODEL_PATH,
    table_out: Annotated[Path, typer.Option("--table-out")] = DEFAULT_TABLE_PATH,
    val_days: Annotated[int, typer.Option("--val-days")] = 28,
    stores: Annotated[Optional[str], typer.Option("--stores")] = None,
) -> None:
    """
    Score the direct + recursive models and the naive baseline per ABC-XYZ cell
    on the validation window, and write the segment-performance table.
    """
    configure_logging(name="evaluation")
    store_list = [s.strip() for s in stores.split(",")] if stores else None

    df = _load_features(features_dir, stores=store_list)
    cutoff = df["date"].max() - pd.Timedelta(days=val_days - 1)

    # Val slice with feature columns for the direct model.
    val_df = df[df["date"] >= cutoff].copy()

    # Baseline needs full history (computed before we drop the big frame).
    logger.info("Computing naive lag-7 baseline...")
    baseline = naive_seasonal_forecast(df[[*ID_COLS, "date", TARGET_COL]], lag=7, val_days=val_days)

    # Segment labels from the TRAINING window only (revenue = units x price).
    logger.info("Building ABC-XYZ segments on the training window...")
    seg_src = df.loc[df["date"] < cutoff, [*ID_COLS, "date", TARGET_COL, "sell_price"]].copy()
    del df
    seg_src["revenue"] = seg_src[TARGET_COL] * seg_src["sell_price"]
    segments = segment_skus(seg_src, value_col="revenue")
    del seg_src
    logger.info(f"Segmented {len(segments):,} SKU-store combinations")

    # Per-row predictions for each source (y_true=sales, y_pred=prediction).
    logger.info("Predicting with the direct horizon-safe model...")
    model = joblib.load(model_path)
    direct = val_df[[*ID_COLS, "date", TARGET_COL]].copy()
    direct["prediction"] = model.predict(val_df[HORIZON_SAFE_FEATURE_COLS])
    del val_df

    sources = {"direct": direct, "baseline": baseline}
    if RECURSIVE_PRED_PATH.exists():
        sources["recursive"] = pd.read_parquet(RECURSIVE_PRED_PATH)
        logger.info("Loaded recursive per-row predictions.")
    else:
        logger.warning(
            f"{RECURSIVE_PRED_PATH.name} not found — run `python -m src.recursive` "
            "to include the recursive column. Skipping it."
        )

    per_cell = {name: segment_metrics(pred, segments) for name, pred in sources.items()}
    overall = {
        name: {
            "rmse": round(rmse(pred[TARGET_COL], pred["prediction"]), 4),
            "mape": round(mape(pred[TARGET_COL], pred["prediction"]), 4),
        }
        for name, pred in sources.items()
    }

    table = _assemble_table(per_cell)
    _save(table, overall, table_out)
    _log_table(table, overall)
    return


# ---- Helpers -----------------------------------------------------------------


def _assemble_table(per_cell: dict[str, DataFrame]) -> DataFrame:
    """Join the per-source per-cell metrics into one wide table with lift columns."""
    base = per_cell["direct"].rename(columns={"rmse": "direct_rmse", "mape": "direct_mape"})
    table = base[["abc_xyz", "n_skus", "n_obs", "direct_rmse", "direct_mape"]].copy()

    if "recursive" in per_cell:
        rec = per_cell["recursive"].rename(
            columns={"rmse": "recursive_rmse", "mape": "recursive_mape"}
        )
        table = table.merge(rec[["abc_xyz", "recursive_rmse", "recursive_mape"]], on="abc_xyz")

    bas = per_cell["baseline"].rename(columns={"rmse": "baseline_rmse", "mape": "baseline_mape"})
    table = table.merge(bas[["abc_xyz", "baseline_rmse", "baseline_mape"]], on="abc_xyz")

    table["abc_class"] = table["abc_xyz"].str[0]
    table["xyz_class"] = table["abc_xyz"].str[1]
    table["policy"] = classify_mts_mto(table["abc_class"], table["xyz_class"]).to_numpy()

    table["direct_vs_baseline_pct"] = (
        (table["baseline_rmse"] - table["direct_rmse"]) / table["baseline_rmse"] * 100
    ).round(1)
    if "recursive_rmse" in table.columns:
        table["recursive_vs_baseline_pct"] = (
            (table["baseline_rmse"] - table["recursive_rmse"]) / table["baseline_rmse"] * 100
        ).round(1)

    return table.sort_values("abc_xyz").reset_index(drop=True)


def _save(table: DataFrame, overall: dict, table_out: Path) -> None:
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_out, index=False)
    json_out = table_out.with_suffix(".json")
    payload = {"overall": overall, "by_segment": table.to_dict(orient="records")}
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.success(f"Saved segment table -> {table_out}")
    logger.success(f"Saved segment JSON  -> {json_out}")
    return


def _log_table(table: DataFrame, overall: dict) -> None:
    logger.success("Per-segment RMSE (direct / recursive / baseline):")
    for _, r in table.iterrows():
        rec = f" rec={r['recursive_rmse']:.3f}" if "recursive_rmse" in table.columns else ""
        logger.success(
            f"  {r['abc_xyz']} (n={r['n_skus']:>5}): direct={r['direct_rmse']:.3f}{rec} "
            f"base={r['baseline_rmse']:.3f}  lift={r['direct_vs_baseline_pct']:+.1f}%"
        )
    for name, m in overall.items():
        logger.success(f"  overall {name}: RMSE={m['rmse']} MAPE={m['mape']}%")
    return


if __name__ == "__main__":
    app()
