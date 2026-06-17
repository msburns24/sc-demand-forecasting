import numpy as np
import pandas as pd
from pandas import DataFrame


def classify_abc(
    df: DataFrame,
    revenue_col: str,
    a_threshold: float = 0.70,
    b_threshold: float = 0.90,
) -> DataFrame:
    """
    Classify SKU-store combinations into A/B/C tiers by cumulative revenue
    contribution (Pareto principle), computed independently per store.

    A: items driving the first a_threshold of each store's revenue (highest value)
    B: items from a_threshold to b_threshold
    C: remainder
    """
    item_revenue = (
        df.groupby(["store_id", "item_id"], observed=True)[revenue_col]
        .sum()
        .reset_index()
        .sort_values(["store_id", revenue_col], ascending=[True, False])
    )
    store_totals = item_revenue.groupby("store_id", observed=True)[
        revenue_col
    ].transform("sum")
    cumulative_pct = (
        item_revenue.groupby("store_id", observed=True)[revenue_col].cumsum()
        / store_totals
    )

    item_revenue["abc_class"] = np.select(
        [cumulative_pct <= a_threshold, cumulative_pct <= b_threshold],
        ["A", "B"],
        default="C",
    )

    return df.merge(
        item_revenue[["store_id", "item_id", "abc_class"]],
        on=["store_id", "item_id"],
        how="left",
    )


def classify_xyz(
    df: DataFrame,
    demand_col: str,
    date_col: str,
    x_threshold: float = 0.5,
    y_threshold: float = 1.0,
) -> DataFrame:
    """
    Classify SKU-store combinations into X/Y/Z tiers by demand variability,
    measured as coefficient of variation (CV = std / mean) on weekly demand.

    X: CV < x_threshold (predictable)
    Y: x_threshold ≤ CV < y_threshold (moderate)
    Z: CV ≥ y_threshold, or zero/undefined demand (erratic)
    """
    weekly = (
        df.assign(_week=pd.to_datetime(df[date_col]).dt.to_period("W"))
        .groupby(["store_id", "item_id", "_week"], observed=True)[demand_col]
        .sum()
        .reset_index()
    )

    stats = (
        weekly.groupby(["store_id", "item_id"], observed=True)[demand_col]
        .agg(["mean", "std"])
        .reset_index()
    )

    # NaN arises from zero-demand (0/0) or single-observation series (std=NaN).
    # Both cases are unclassifiable as X or Y — default to Z.
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
