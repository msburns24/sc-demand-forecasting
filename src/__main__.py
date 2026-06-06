from __future__ import annotations
from pathlib import Path
from typing import Annotated, Optional

import typer

from src.data._kaggle import download_dataset_from_kaggle
from src.data.blob_io import upload_data_to_blob


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)


@app.command()
def download(
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Output directory (default: `data/`)",
            dir_okay=True,
            file_okay=False,
        ),
    ] = None,
) -> None:
    """Download dataset from Kaggle."""
    download_dataset_from_kaggle(output_dir)
    return


@app.command()
def upload(
    data_dir: Annotated[
        Optional[Path],
        typer.Option(
            "-i",
            "--input",
            help="Input directory (default: `data/`)",
            dir_okay=True,
            file_okay=False,
        ),
    ] = None,
) -> None:
    """Upload data files to Azure Blob Storage (raw container)."""
    upload_data_to_blob(data_dir)
    return


if __name__ == "__main__":
    app()
