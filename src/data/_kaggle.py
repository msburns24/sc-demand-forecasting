"""
Scripts to download data from the Kaggle competition.
"""

import shutil
from pathlib import Path
from typing import Optional

import dotenv
import kagglehub

from src._logging import logger, configure_logging


ROOT_DIR = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data"
COMPETITION_NAME = "m5-forecasting-accuracy"
dotenv.load_dotenv(ROOT_DIR / ".env")


def download_dataset_from_kaggle(output_dir: Optional[Path] = None) -> None:
    """
    Download the Kaggle dataset and extract the CSV files. By default, output
    files to `<project-root>/data/`.
    """
    output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    configure_logging(name="kaggle", console=True)
    logger.info(f"Downloading dataset: '{COMPETITION_NAME}'")
    kagglehub.competition_download(
        COMPETITION_NAME,
        output_dir=str(output_dir),
        force_download=True,
    )

    output_paths = [f for f in output_dir.iterdir() if f.is_file()]
    filenames = [path.name for path in output_paths]

    logger.info(f"Extracted {len(output_paths)} files: {filenames}")
    logger.debug("File paths:")
    logger.debug(f"  {output_paths[0].parent}/")
    for i, path in enumerate(output_paths, start=1):
        prefix = "└──" if i == len(output_paths) else "├──"
        size = path.stat().st_size
        logger.debug(f"  {prefix} {path.name:<35} {size: 12,} B")

    logger.debug("Removing `.complete` directory from kaggle download...")
    shutil.rmtree(output_dir / ".complete")
    return
