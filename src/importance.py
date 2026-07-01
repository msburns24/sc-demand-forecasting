"""
SHAP-based feature importance analysis for the direct LightGBM model (SCDF-22).

Aggregate importance (e.g. LightGBM's own split/gain importance) says *which*
features matter but not *how*, and hides whether a feature's effect is
uniform across product categories. SHAP (SHapley Additive exPlanations)
values attribute each prediction to its input features in a way that's
additive and comparable across rows, letting us rank global importance and
also compare importance *within* a subgroup -- e.g. is a SNAP-day flag only
relevant for FOODS? Offline: SHAP is computed on a random sample of the
validation window (not the full ~850K-row window), since exact tree SHAP
cost scales with rows x trees.
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import typer  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402
from pandas import DataFrame  # noqa: E402

from src._logging import logger, configure_logging  # noqa: E402
from src.cli import create_app  # noqa: E402
from src.model import HORIZON_SAFE_FEATURE_COLS  # noqa: E402
from src.train import DEFAULT_MODEL_PATH, PROCESSED_DATA_DIR, _load_features  # noqa: E402


ROOT_DIR = _ROOT
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_IMPORTANCE_PATH = OUTPUTS_DIR / "shap_importance.csv"
DEFAULT_IMPORTANCE_BY_CAT_PATH = OUTPUTS_DIR / "shap_importance_by_cat.csv"
DEFAULT_SAMPLE_SIZE = 5000


def sample_rows(df: DataFrame, n: int, random_state: int = 42) -> DataFrame:
    """
    Random row sample for expensive SHAP computation, capped at `n`.

    Exact tree SHAP cost scales with rows x trees x average tree depth; on
    the full ~850K-row validation window that's too slow for an offline
    script. A random sample keeps the mean |SHAP| ranking stable (large-
    sample means are robust to subsampling) while bounding runtime.
    """
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def compute_shap_values(model, X: DataFrame) -> np.ndarray:
    """
    SHAP values for a global LightGBM regressor via exact TreeExplainer.

    Tree SHAP is exact (not sampled/approximated like KernelSHAP) and native
    to gradient-boosted trees, so no background dataset or model wrapper is
    needed beyond the fitted `LGBMRegressor` itself. Returns an
    `(n_samples, n_features)` array aligned to `X`'s columns.
    """
    explainer = shap.TreeExplainer(model)
    return np.asarray(explainer.shap_values(X))


def mean_abs_importance(shap_values: np.ndarray, feature_names: list[str]) -> DataFrame:
    """
    Global feature ranking by mean |SHAP value| across the sample.

    Returns one row per feature, columns `[feature, mean_abs_shap, rank]`,
    sorted descending by `mean_abs_shap` (rank 1 = most important).
    """
    importance = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out["mean_abs_shap"] = out["mean_abs_shap"].round(4)
    return out


def importance_by_group(
    shap_values: np.ndarray, feature_names: list[str], group: ArrayLike
) -> DataFrame:
    """
    Per-group feature ranking by mean |SHAP value| (e.g. `group` = `cat_id`).

    A global ranking can hide subgroup-specific behavior -- e.g. a feature
    that's a top-5 driver within one product category and irrelevant in
    another. Returns columns
    `[group, feature, mean_abs_shap, rank_within_group]`, one row per
    (group, feature) pair.
    """
    group = np.asarray(group)
    if len(group) != len(shap_values):
        raise ValueError("group must be the same length as shap_values")

    wide = pd.DataFrame(np.abs(shap_values), columns=feature_names)
    wide["group"] = group
    long = wide.melt(id_vars="group", var_name="feature", value_name="abs_shap")
    out = (
        long.groupby(["group", "feature"], observed=True)["abs_shap"]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"abs_shap": "mean_abs_shap"})
    )
    out["rank_within_group"] = (
        out.groupby("group")["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    )
    return out.sort_values(["group", "rank_within_group"]).reset_index(drop=True)


# ---- Entry Point -------------------------------------------------------------

app = create_app(name="importance")


@app.command()
def main(
    features_dir: Annotated[
        Path, typer.Option("-f", "--features-dir", file_okay=False)
    ] = PROCESSED_DATA_DIR,
    model_path: Annotated[
        Path, typer.Option("--model", dir_okay=False)
    ] = DEFAULT_MODEL_PATH,
    importance_out: Annotated[
        Path, typer.Option("--importance-out")
    ] = DEFAULT_IMPORTANCE_PATH,
    importance_by_cat_out: Annotated[
        Path, typer.Option("--importance-by-cat-out")
    ] = DEFAULT_IMPORTANCE_BY_CAT_PATH,
    val_days: Annotated[int, typer.Option("--val-days")] = 28,
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size", help="Rows sampled from the validation window for SHAP."
        ),
    ] = DEFAULT_SAMPLE_SIZE,
    stores: Annotated[Optional[str], typer.Option("--stores")] = None,
) -> None:
    """
    SHAP feature importance for the direct model, globally and per `cat_id`.

    Loads the validation window, draws a random row sample (SHAP is too
    expensive to run on the full ~850K-row window), computes exact tree SHAP
    values, and writes the global top-feature ranking plus a per-category
    breakdown -- the latter surfaces category-specific effects (e.g. a
    SNAP-day flag mattering for FOODS but not HOUSEHOLD) that a single global
    ranking would hide.
    """
    configure_logging(name="importance")
    store_list = [s.strip() for s in stores.split(",")] if stores else None

    df = _load_features(features_dir, stores=store_list)
    cutoff = df["date"].max() - pd.Timedelta(days=val_days - 1)
    val_df = df[df["date"] >= cutoff].copy()
    del df

    sample = sample_rows(val_df, n=sample_size)
    del val_df
    logger.info(f"Sampled {len(sample):,} validation rows for SHAP.")

    logger.info("Loading model and computing SHAP values...")
    model = joblib.load(model_path)
    X_sample = sample[HORIZON_SAFE_FEATURE_COLS]
    shap_values = compute_shap_values(model, X_sample)

    importance = mean_abs_importance(shap_values, HORIZON_SAFE_FEATURE_COLS)
    _save_table(importance, importance_out)
    _log_importance(importance)

    by_cat = importance_by_group(shap_values, HORIZON_SAFE_FEATURE_COLS, sample["cat_id"])
    _save_table(by_cat, importance_by_cat_out)
    logger.success(f"Saved per-category SHAP breakdown -> {importance_by_cat_out}")
    return


# ---- Helpers -----------------------------------------------------------------


def _save_table(table: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    json_path = path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(table.to_dict(orient="records"), fh, indent=2)
    logger.success(f"Saved -> {path}")
    logger.success(f"Saved -> {json_path}")
    return


def _log_importance(table: DataFrame, top_n: int = 15) -> None:
    logger.success(f"Top {top_n} features by mean |SHAP|:")
    for _, r in table.head(top_n).iterrows():
        logger.success(f"  {r['rank']:>2}. {r['feature']:<32} {r['mean_abs_shap']:.4f}")
    return


if __name__ == "__main__":
    app()
