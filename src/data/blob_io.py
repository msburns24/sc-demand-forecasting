"""
Upload/download data files to/from Azure Blob Storage.
"""

from pathlib import Path
from typing import Callable, Literal, Optional

from azure.storage.blob import BlobServiceClient, ContainerClient
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
from config.settings import settings


ROOT_DIR = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"


def download_from_blob(
    container: Literal["raw", "processed"],
    blob_name: str,
    output_path: Path,
    progress_hook: Optional[Callable[[int, int | None], None]] = None,
) -> None:
    """
    Download file from blob
    """
    client = _get_container_client(container)

    logger.info(f"Downloading blob '{blob_name}' to file: '{output_path}'")
    with open(output_path, "wb") as file:
        download_stream = client.download_blob(
            blob=blob_name,
            progress_hook=progress_hook,
        )
        file.write(download_stream.readall())

    size = output_path.stat().st_size
    logger.info(f"Finished downloading to '{output_path.name}' ({size:,} B).")
    return


def upload_to_blob(
    local_path: Path,
    container: Literal["raw", "processed"],
    blob_name: Optional[str] = None,
    progress_hook: Optional[Callable[[int, int | None], None]] = None,
) -> None:
    """
    Upload file `local_path` to blob
    """
    if blob_name is None:
        blob_name = local_path.name
        logger.debug(f"Using default blob name from local path: '{blob_name}'")

    client = _get_container_client(container)
    logger.info(f"Uploading file to blob: '{local_path}'")
    logger.debug(f"Blob name: '{blob_name}'")
    with open(local_path, "rb") as file:
        client.upload_blob(blob_name, file, overwrite=True, progress_hook=progress_hook)
    logger.info(f"Finished uploading blob: '{blob_name}'")
    return


def upload_files_to_blob(data_dir: Optional[Path] = None) -> None:
    """
    Upload all files in `data_dir` to the raw Azure Blob Storage container.
    """
    data_dir = DEFAULT_DATA_DIR if data_dir is None else data_dir
    configure_logging(name="blob_io", console=True)

    files = [f for f in data_dir.iterdir() if f.is_file()]
    logger.info(f"Uploading {len(files)} files to container 'raw'")
    logger.debug(f"File paths: {files}")

    progress = Progress(
        TextColumn("[bold]{task.description:<40}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )

    logger.info(f"Starting upload of {len(files):,} files...")
    with progress:
        for path in files:
            size = path.stat().st_size
            task_id: TaskID = progress.add_task(path.name, total=size)
            progress_hook = _make_progress_hook(progress, task_id)
            upload_to_blob(path, "raw", progress_hook=progress_hook)

    logger.info("Upload complete.")
    return


# ---- Private -----------------------------------------------------------------


def _get_container_client(
    container: Literal["raw", "processed"],
) -> ContainerClient:
    if container not in ("raw", "processed"):
        msg = f"Unknown container '{container}'. Expected one of: ('raw', 'processed')"
        logger.error(msg)
        raise ValueError(msg)

    connection_string = settings.azure_storage_connection_string
    logger.debug("Connecting to blob client...")
    blob_client = BlobServiceClient.from_connection_string(connection_string)
    logger.debug("Connected to blob client.")

    logger.debug(f"Connecting to container client: '{container}'")
    container_client = blob_client.get_container_client(container)
    logger.debug("Connected to container client.")

    return container_client


def _make_progress_hook(progress: Progress, task_id: TaskID):
    def hook(transferred: int, _total: Optional[int]) -> None:
        progress.update(task_id, completed=transferred)

    return hook
