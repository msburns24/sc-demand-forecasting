"""
Upload/download data files to/from Azure Blob Storage.
"""

import os
from pathlib import Path
from typing import Optional

import dotenv
from azure.storage.blob import BlobServiceClient
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from src._logging import logger, configure_logging


ROOT_DIR = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
dotenv.load_dotenv(ROOT_DIR / ".env")


def upload_data_to_blob(data_dir: Optional[Path] = None) -> None:
    """Upload all files in data_dir to the raw Azure Blob Storage container."""
    data_dir = DEFAULT_DATA_DIR if data_dir is None else data_dir
    configure_logging(name="upload_data_to_blob", console=True)

    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_name = os.environ.get("AZURE_STORAGE_CONTAINER_RAW", "raw")

    client = BlobServiceClient.from_connection_string(connection_string)
    container_client = client.get_container_client(container_name)

    files = [f for f in data_dir.iterdir() if f.is_file()]
    logger.info(f"Uploading {len(files)} files to container '{container_name}'")

    progress = Progress(
        TextColumn("[bold]{task.description:<40}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        for path in files:
            size = path.stat().st_size
            task_id: TaskID = progress.add_task(path.name, total=size)

            def make_hook(tid: TaskID):
                def hook(transferred: int, _total: Optional[int]) -> None:
                    progress.update(tid, completed=transferred)

                return hook

            with path.open("rb") as f:
                container_client.upload_blob(
                    path.name, f, overwrite=True, progress_hook=make_hook(task_id)
                )

    logger.info("Upload complete")
