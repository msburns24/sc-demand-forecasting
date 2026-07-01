import numpy as np
import pandas as pd
from pandas import DataFrame


def classify_abc(
    df: DataFrame,
    sales_col: str,
    store_col: str = "store_id",
    item_col: str = "item_id",
    a_threshold: float = 0.70,
    b_threshold: float = 0.90,
) -> DataFrame:
    """
    Classify SKU-store combinations into A/B/C tiers by cumulative sales
    contribution (Pareto principle), computed independently per store.

    - **A:** items driving the first `a_threshold` of each store's sales
      (highest value)
    - **B:** items from `a_threshold` to `b_threshold`
    - **C:** remainder
    """
    item_revenue = (
        df.groupby([store_col, item_col], observed=True)[sales_col]
        .sum()
        .reset_index()
        .sort_values([store_col, sales_col], ascending=[True, False])
    )
    store_totals = (
        item_revenue.groupby(store_col, observed=True)[sales_col].transform(
            "sum"
        )  # Copy sum for each row
    )
    cumulative_pct = (
        item_revenue.groupby(store_col, observed=True)[sales_col]
        .cumsum()
        .div(store_totals)
    )
    item_revenue["abc_class"] = np.select(
        [cumulative_pct <= a_threshold, cumulative_pct <= b_threshold],
        ["A", "B"],
        default="C",
    )
    return df.merge(
        item_revenue[[store_col, item_col, "abc_class"]],
        on=[store_col, item_col],
        how="left",
    )


def classify_xyz(
    df: DataFrame,
    demand_col: str,
    date_col: str,
    store_col: str = "store_id",
    item_col: str = "item_id",
    x_threshold: float = 0.5,
    y_threshold: float = 1.0,
) -> DataFrame:
    """
    Classify SKU-store combinations into X/Y/Z tiers by demand variability,
    measured as coefficient of variation (CV = std / mean) on weekly demand.

    - **X:** CV < `x_threshold` (predictable)
    - **Y:** `x_threshold` <= CV < `y_threshold` (moderate)
    - **Z:** CV >= `y_threshold`, or zero/undefined demand (erratic)
    """
    weekly = (
        df.assign(_week=pd.to_datetime(df[date_col]).dt.to_period("W"))
        .groupby([store_col, item_col, "_week"], observed=True)[demand_col]
        .sum()
        .reset_index()
    )
    stats = (
        weekly.groupby([store_col, item_col], observed=True)[demand_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    cv = (stats["std"] / stats["mean"]).fillna(float("inf"))
    stats["xyz_class"] = np.select(
        [cv < x_threshold, cv < y_threshold],
        ["X", "Y"],
        default="Z",
    )
    return df.merge(
        stats[["store_id", "item_id", "xyz_class"]],
        on=["store_id", "item_id"],
        how="left",
    )


def segment_skus(
    df: DataFrame,
    value_col: str,
    date_col: str = "date",
    store_col: str = "store_id",
    item_col: str = "item_id",
) -> DataFrame:
    """
    Build the per-SKU-store ABC-XYZ label table.

    Composes `classify_abc` and `classify_xyz` on `value_col` (typically revenue
    = units x price), then collapses to one row per `(store_id, item_id)`. This
    is the shared segment lookup for evaluation (SCDF-20) and error-stats
    (SCDF-21): merge it onto per-row predictions by `(store_id, item_id)`.

    Parameters
    ----------
    df : DataFrame
        Long-format demand with `store_id`, `item_id`, `date`, and `value_col`.
    value_col : str
        Column used for both ABC (cumulative contribution) and XYZ (weekly CV).

    Returns
    -------
    DataFrame
        Columns `[store_id, item_id, abc_class, xyz_class, abc_xyz]`, one row per
        SKU-store.
    """
    labelled = classify_abc(df, sales_col=value_col, store_col=store_col, item_col=item_col)
    labelled = classify_xyz(
        labelled, demand_col=value_col, date_col=date_col, store_col=store_col, item_col=item_col
    )
    segments = (
        labelled.groupby([store_col, item_col], observed=True)[["abc_class", "xyz_class"]]
        .first()
        .reset_index()
    )
    segments["abc_xyz"] = segments["abc_class"] + segments["xyz_class"]
    return segments


def classify_mts_mto(abc: pd.Series, xyz: pd.Series) -> pd.Series:
    """
    Map ABC and XYZ class labels to an inventory stocking policy.

    - **MTS** (Make to Stock): AX, AY, AZ, BX, BY, CX
    - **MTS/review** (MTS with periodic review): BZ, CY
    - **MTO** (Make to Order): CZ
    """
    combined = abc + xyz
    policy = np.select(
        [combined == "CZ", combined.isin(["BZ", "CY"])],
        ["MTO", "MTS/review"],
        default="MTS",
    )
    return pd.Series(policy, index=abc.index)
