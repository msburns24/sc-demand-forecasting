import numpy as np
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
