from pathlib import Path
from typing import Annotated, Iterable, Literal, Optional

import pandas as pd
import typer
from pandas import DataFrame

from src._logging import logger, configure_logging
from src.cli import create_app
from src.data.blob_io import upload_to_blob, download_from_blob


ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOCAL_FEATURES_PATH = PROCESSED_DATA_DIR / "features.parquet"
CALENDAR_COLUMNS = [
    "d",
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]


def build_features(
    df: DataFrame,
    calendar: DataFrame,
    prices: DataFrame,
) -> DataFrame:
    """
    Build the full feature matrix from the raw M5 DataFrames.

    Parameters
    ----------
    df : DataFrame
        Raw sales data (wide format, `sales_train_evaluation.csv`).
    calendar : DataFrame
        Calendar with date, event, and SNAP flags.
    prices : DataFrame
        Sell prices per store/item/week.

    Returns
    -------
    DataFrame
        Long-format feature matrix, one row per item/store/date.
        Input DataFrames are not modified.
    """
    df = df.copy()
    df = (
        df.pipe(_melt_sales)
        .pipe(_merge_calendar, calendar)
        .pipe(_merge_prices, prices)
        .pipe(_impute_sell_prices)
        .pipe(_add_lag_features)
        .pipe(_add_rolling_features)
        .pipe(_add_calendar_features)
        .pipe(_add_price_features)
        .pipe(_encode_categorical_columns)
    )
    logger.info(f"Feature matrix shape: {df.shape}")
    return df


def save_features(df: DataFrame, local_path: Path | None = None) -> None:
    """
    Save feature matrix locally and upload to blob processed container.
    """
    local_path = LOCAL_FEATURES_PATH if local_path is None else local_path
    _ensure_local_directories(local_path)
    _save_to_local_parquet(df, local_path)
    upload_to_blob(local_path, "processed", blob_name="features.parquet")
    logger.success("Complete.")
    return


# ---- Entry Point -------------------------------------------------------------

app = create_app(name="features")


@app.command()
def main(
    input_dir: Annotated[
        Path,
        typer.Option(
            "-i",
            "--input",
            help="Path to input CSV directory (default: 'data/raw/').",
            dir_okay=True,
            file_okay=False,
            exists=True,
        ),
    ] = RAW_DATA_DIR,
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Path to output parquet directory (default: 'data/processed/').",
            dir_okay=True,
            file_okay=False,
            exists=True,
        ),
    ] = PROCESSED_DATA_DIR,
) -> None:
    """
    Build the full feature matrix from the raw M5 DataFrames.

    Input directory must include files `sales_train_evaluation.csv`,
    `calendar.csv`, and `sell_prices.csv`.
    """
    configure_logging(name="features")
    input_paths = _get_input_paths(input_dir)
    store_ids = _load_store_ids(input_paths[0])
    output_paths = _get_output_paths(output_dir, store_ids)

    _maybe_download_from_blob(*input_paths)
    for store_id, output_path in output_paths.items():
        _load_and_build_one_store(*input_paths, output_path, store_id)

    logger.success("All stores complete.")
    return


# ---- Helpers -----------------------------------------------------------------


def _get_input_paths(input_dir: Path) -> tuple[Path, Path, Path]:
    """Returns paths to sales, calendar, and prices."""
    return (
        input_dir / "sales_train_evaluation.csv",
        input_dir / "calendar.csv",
        input_dir / "sell_prices.csv",
    )


def _get_output_paths(output_dir: Path, store_ids: list[str]) -> dict[str, Path]:
    return {sid: output_dir / f"features_{sid}.parquet" for sid in store_ids}


def _maybe_download_from_blob(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            logger.info(f"File '{path.name}' found.")
            continue

        logger.info(f"Downloading '{path.name}' from blob...")
        download_from_blob("raw", path.name, path.parent)
    return


def _load_store_ids(sales_path: Path) -> list[str]:
    df = pd.read_csv(sales_path)
    return sorted(set(df["store_id"].to_list()))


def _load_and_build_one_store(
    sales_path: Path,
    calendar_path: Path,
    prices_path: Path,
    output_path: Path,
    store_id: str,
) -> None:
    logger.info(f"Loading raw data for store: '{store_id}'")
    sales = _load_from_csv_for_store_id(sales_path, store_id)
    calendar = _load_from_csv_for_store_id(calendar_path, store_id=None)
    prices = _load_from_csv_for_store_id(prices_path, store_id)
    features = build_features(sales, calendar, prices)

    features.to_parquet(output_path, index=False)
    logger.info(f"Saved data to parquet: '{output_path.name}'")
    logger.success(f"Feature engineering complete for store: '{store_id}'\n")
    return


# ---- Pipeline ----------------------------------------------------------------


def _load_from_csv_for_store_id(
    path: Path,
    store_id: Optional[str] = None,
) -> DataFrame:
    logger.info(f"Loading data from CSV: '{path}'")
    df = pd.read_csv(path)
    logger.info(f"Read {len(df):,} rows from file.")

    if store_id:
        logger.info(f"Filtering for store id '{store_id}'...")
        df = df.query(f"store_id == '{store_id}'").reset_index(drop=True)
        logger.info(f"Filtered to {len(df):,} rows.")

    return df


def _melt_sales(
    df: DataFrame,
    day_column_prefix: str = "d_",
    var_name: str = "d",
    value_name: str = "sales",
) -> DataFrame:
    logger.info("Melting sales to long format...")
    day_columns = [c for c in df.columns if c.startswith(day_column_prefix)]
    id_columns = [c for c in df.columns if c not in day_columns]
    return df.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name=var_name,
        value_name=value_name,
    )


def _merge_calendar(
    df: DataFrame,
    calendar: DataFrame,
    how: Literal["left", "right", "outer", "inner"] = "left",
    on: str = "d",
    use_cols: Iterable[str] = CALENDAR_COLUMNS,
    date_col: str = "date",
) -> DataFrame:
    logger.info("Merging calendar...")
    cal = calendar[use_cols].copy()
    cal[date_col] = pd.to_datetime(cal[date_col])
    return df.merge(cal, how=how, on=on)


def _merge_prices(
    df: DataFrame,
    prices: DataFrame,
    how: Literal["left", "right", "outer", "inner"] = "left",
    on: str | Iterable[str] = ("store_id", "item_id", "wm_yr_wk"),
) -> DataFrame:
    logger.info("Merging prices...")
    return df.merge(prices, how=how, on=on)


def _impute_sell_prices(
    df: DataFrame,
    price_col: str = "sell_price",
    store_col: str = "store_id",
    item_col: str = "item_id",
    date_col: str = "date",
    ignore_index: bool = True,
) -> DataFrame:
    logger.info("Imputing sell prices...")
    df = df.sort_values([store_col, item_col, date_col])
    df[price_col] = df.groupby([store_col, item_col])[price_col].transform(
        lambda s: s.ffill().bfill()
    )

    item_mean = df.groupby(item_col)[price_col].transform("mean")
    df[price_col] = df[price_col].fillna(item_mean)

    if ignore_index:
        df = df.reset_index(drop=True)
    return df


def _add_lag_features(
    df: DataFrame,
    lags: Iterable[int] = (7, 14, 28),
    value_col: str = "sales",
    prefix: str = "lag_",
    store_col: str = "store_id",
    item_col: str = "item_id",
    observed: bool = True,
) -> DataFrame:
    logger.info("Building lag features...")
    for lag in lags:
        df[f"{prefix}{lag}"] = df.groupby([store_col, item_col], observed=observed)[
            value_col
        ].shift(lag)
    return df


def _add_rolling_features(
    df: DataFrame,
    windows: Iterable[int] = (7, 28),
    value_col: str = "sales",
    mean_prefix: str = "rolling_mean_",
    std_prefix: str = "rolling_std_",
    store_col: str = "store_id",
    item_col: str = "item_id",
) -> DataFrame:
    logger.info("Building rolling features...")
    group_key = df[store_col].astype(str) + "_" + df[item_col].astype(str)
    shifted = df.groupby([store_col, item_col])[value_col].shift(1)
    for window in windows:
        df[f"{mean_prefix}{window}"] = shifted.groupby(group_key).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{std_prefix}{window}"] = shifted.groupby(group_key).transform(
            lambda s: s.rolling(window, min_periods=1).std()
        )
    return df


def _add_calendar_features(df: DataFrame) -> DataFrame:
    logger.info("Building calendar features...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["event_type_1"].notna().astype(int)
    df["is_snap_CA"] = df["snap_CA"].astype(int)
    df["is_snap_TX"] = df["snap_TX"].astype(int)
    df["is_snap_WI"] = df["snap_WI"].astype(int)
    return df


def _add_price_features(df: DataFrame) -> DataFrame:
    logger.info("Building price features...")
    df["price_change_pct"] = (
        df.groupby(["store_id", "item_id"])["sell_price"].pct_change().fillna(0)
    )
    df["is_price_decrease"] = df["price_change_pct"].lt(0).astype(int)
    df["is_price_increase"] = df["price_change_pct"].gt(0).astype(int)
    df["price_relative_to_category_mean"] = df["sell_price"].div(
        (df.groupby(["dept_id", "date"])["sell_price"].transform("mean"))
    )
    return df


def _encode_categorical_columns(df: DataFrame) -> DataFrame:
    logger.info("Encoding categoricals...")
    for col in ["item_id", "store_id", "dept_id", "cat_id", "state_id"]:
        df[col] = df[col].astype("category")
    return df


def _ensure_local_directories(local_path: Path) -> None:
    logger.info("Ensuring local directories...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return


def _save_to_local_parquet(df: DataFrame, local_path: Path) -> None:
    logger.info(f"Saving features to {local_path}...")
    df.to_parquet(local_path, index=False)
    return


if __name__ == "__main__":
    app()
