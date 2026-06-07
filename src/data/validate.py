"""
reads files directly from Blob and checks: row counts, date ranges, null rates
per column, SKU-store combination count (~30,490)

This is where you verify the upload actually worked and that the data is what
you expect. Read each file back from Blob into a DataFrame and check: row
counts, date ranges, null rates per column, and that you have ~30,490 unique
SKU-store combinations. The last check is the most important — if that number
is wildly off, something went wrong in the download or upload.

The approach: write a standalone script (or notebook section) that calls
download_from_blob, loads into pandas, and runs assertions. Make the checks
explicit and loud — raise or print clearly if something fails, rather than
silently producing a wrong number. You want this to be runnable by anyone who
clones the repo.

For the ~30,490 count: that comes from 3,049 unique items × 10 stores. A
quick df.groupby(['item_id', 'store_id']).ngroups will tell you what you
actually have.
"""

import tempfile
from pathlib import Path

import pandas as pd

from src._logging import logger, configure_logging
from src.data.blob_io import download_from_blob


_EXPECTED_SKU_STORE_COMBOS = 30_490
_EXPECTED_N_ITEMS = 3_049
_EXPECTED_N_STORES = 10
_NULL_RATE_THRESHOLD = 0.01

_RAW_BLOBS = [
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "calendar.csv",
]


def _check_null_rates(df: pd.DataFrame, name: str) -> None:
    null_rates = df.isnull().mean()
    flagged = null_rates[null_rates > _NULL_RATE_THRESHOLD]
    if flagged.empty:
        logger.info(
            f"  [{name}] null rates OK (all columns <= {_NULL_RATE_THRESHOLD:.0%})"
        )
    else:
        for col, rate in flagged.items():
            logger.warning(f"  [{name}] HIGH NULL RATE: '{col}' = {rate:.2%}")


def _check_sales(df: pd.DataFrame, name: str) -> None:
    day_cols = [c for c in df.columns if c.startswith("d_")]
    n_days = len(day_cols)
    sku_store_count = df.groupby(["item_id", "store_id"]).ngroups
    n_items = df["item_id"].nunique()
    n_stores = df["store_id"].nunique()

    logger.info(
        f"  [{name}] rows={len(df):,}  day_cols={n_days}  sku×store={sku_store_count:,}"
    )
    logger.info(f"  [{name}] unique items={n_items}  unique stores={n_stores}")

    if sku_store_count != _EXPECTED_SKU_STORE_COMBOS:
        raise AssertionError(
            f"[{name}] expected {_EXPECTED_SKU_STORE_COMBOS:,} SKU-store combos, "
            f"got {sku_store_count:,}"
        )
    if n_items != _EXPECTED_N_ITEMS:
        raise AssertionError(
            f"[{name}] expected {_EXPECTED_N_ITEMS:,} unique items, got {n_items:,}"
        )
    if n_stores != _EXPECTED_N_STORES:
        raise AssertionError(
            f"[{name}] expected {_EXPECTED_N_STORES} unique stores, got {n_stores}"
        )

    _check_null_rates(df[["item_id", "store_id"] + day_cols], name)
    logger.info(f"  [{name}] PASSED")


def _check_sell_prices(df: pd.DataFrame) -> None:
    name = "sell_prices"
    logger.info(
        f"  [{name}] rows={len(df):,}  "
        f"wm_yr_wk range=[{df['wm_yr_wk'].min()}, {df['wm_yr_wk'].max()}]"
    )
    _check_null_rates(df, name)
    logger.info(f"  [{name}] PASSED")


def _check_calendar(df: pd.DataFrame) -> None:
    name = "calendar"
    df["date"] = pd.to_datetime(df["date"])
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    logger.info(
        f"  [{name}] rows={len(df):,}  date range=[{date_min}, {date_max}]  "
        f"d range=[{df['d'].iloc[0]}, {df['d'].iloc[-1]}]"
    )
    _check_null_rates(df[["date", "wm_yr_wk", "weekday", "month", "year", "d"]], name)
    logger.info(f"  [{name}] PASSED")


def validate_blobs() -> None:
    """
    Download each raw blob and run data quality checks.

    Checks performed
    ----------------
    - Row counts and column counts
    - Date / week ranges
    - Null rates per column (warns if > 1%)
    - Unique SKU-store combos == 30,490 for both sales files
    """
    configure_logging(name="validate", console=True)
    logger.info(f"Validating {len(_RAW_BLOBS)} raw blobs...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for blob_name in _RAW_BLOBS:
            local_path = tmp_dir / blob_name
            logger.info(f"Downloading '{blob_name}'...")
            download_from_blob("raw", blob_name, local_path)

            df = pd.read_csv(local_path)
            logger.info(f"  shape={df.shape}")

            if blob_name.startswith("sales_train"):
                _check_sales(df, blob_name)
            elif blob_name == "sell_prices.csv":
                _check_sell_prices(df)
            elif blob_name == "calendar.csv":
                _check_calendar(df)

    logger.info("All validation checks passed.")


if __name__ == "__main__":
    validate_blobs()
