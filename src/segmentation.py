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
