from unittest.mock import MagicMock, patch

import pytest

import src.data.blob_io as blob_io_module
from src.data.blob_io import download_from_blob, upload_to_blob


FAKE_CONN_STR = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net"


def _make_sdk_mocks():
    """Return (mock_from_conn_str, mock_blob_service, mock_container_client)."""
    mock_container = MagicMock()
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container
    return mock_service, mock_container


@patch("src.data.blob_io.settings")
@patch("src.data.blob_io.BlobServiceClient")
def test_upload_to_blob_calls_upload_blob(mock_bsc_cls, mock_settings, tmp_path):
    """upload_to_blob opens the file and calls container_client.upload_blob."""
    mock_settings.azure_storage_connection_string = FAKE_CONN_STR
    mock_settings.azure_storage_container_raw = "raw"
    mock_settings.azure_storage_container_processed = "processed"

    mock_service, mock_container = _make_sdk_mocks()
    mock_bsc_cls.from_connection_string.return_value = mock_service

    local_file = tmp_path / "sales.csv"
    local_file.write_text("a,b\n1,2\n")

    upload_to_blob(local_file, "raw", blob_name="sales.csv")

    mock_bsc_cls.from_connection_string.assert_called_once_with(FAKE_CONN_STR)
    mock_service.get_container_client.assert_called_once_with("raw")
    mock_container.upload_blob.assert_called_once()

    call_args = mock_container.upload_blob.call_args
    assert call_args.args[0] == "sales.csv"
    assert call_args.kwargs.get("overwrite") is True


@patch("src.data.blob_io.settings")
@patch("src.data.blob_io.BlobServiceClient")
def test_upload_to_blob_derives_blob_name_from_path(
    mock_bsc_cls, mock_settings, tmp_path
):
    """When blob_name is omitted, the local filename is used as the blob name."""
    mock_settings.azure_storage_connection_string = FAKE_CONN_STR
    mock_settings.azure_storage_container_raw = "raw"
    mock_settings.azure_storage_container_processed = "processed"

    mock_service, mock_container = _make_sdk_mocks()
    mock_bsc_cls.from_connection_string.return_value = mock_service

    local_file = tmp_path / "prices.csv"
    local_file.write_text("x\n1\n")

    upload_to_blob(local_file, "raw")  # no explicit blob_name

    call_args = mock_container.upload_blob.call_args
    assert call_args.args[0] == "prices.csv"


@patch("src.data.blob_io.settings")
@patch("src.data.blob_io.BlobServiceClient")
def test_upload_to_blob_processed_container(mock_bsc_cls, mock_settings, tmp_path):
    """upload_to_blob routes to the 'processed' container when requested."""
    mock_settings.azure_storage_connection_string = FAKE_CONN_STR
    mock_settings.azure_storage_container_raw = "raw"
    mock_settings.azure_storage_container_processed = "processed"

    mock_service, mock_container = _make_sdk_mocks()
    mock_bsc_cls.from_connection_string.return_value = mock_service

    local_file = tmp_path / "output.csv"
    local_file.write_text("y\n2\n")

    upload_to_blob(local_file, "processed", blob_name="output.csv")

    mock_service.get_container_client.assert_called_once_with("processed")


@patch("src.data.blob_io.settings")
@patch("src.data.blob_io.BlobServiceClient")
def test_download_from_blob_writes_content(mock_bsc_cls, mock_settings, tmp_path):
    """download_from_blob calls download_blob and writes the result to disk."""
    mock_settings.azure_storage_connection_string = FAKE_CONN_STR
    mock_settings.azure_storage_container_raw = "raw"
    mock_settings.azure_storage_container_processed = "processed"

    fake_content = b"id,val\n1,2\n"
    mock_download_stream = MagicMock()
    mock_download_stream.readall.return_value = fake_content

    mock_container = MagicMock()
    mock_container.download_blob.return_value = mock_download_stream

    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container
    mock_bsc_cls.from_connection_string.return_value = mock_service

    output_path = tmp_path / "downloaded.csv"
    download_from_blob("raw", "sales.csv", output_path)

    mock_bsc_cls.from_connection_string.assert_called_once_with(FAKE_CONN_STR)
    mock_service.get_container_client.assert_called_once_with("raw")
    mock_container.download_blob.assert_called_once_with(
        blob="sales.csv", progress_hook=None
    )
    assert output_path.read_bytes() == fake_content


@patch("src.data.blob_io.settings")
@patch("src.data.blob_io.BlobServiceClient")
def test_download_from_blob_passes_progress_hook(mock_bsc_cls, mock_settings, tmp_path):
    """A progress_hook passed to download_from_blob is forwarded to download_blob."""
    mock_settings.azure_storage_connection_string = FAKE_CONN_STR
    mock_settings.azure_storage_container_raw = "raw"
    mock_settings.azure_storage_container_processed = "processed"

    mock_download_stream = MagicMock()
    mock_download_stream.readall.return_value = b""

    mock_container = MagicMock()
    mock_container.download_blob.return_value = mock_download_stream

    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container
    mock_bsc_cls.from_connection_string.return_value = mock_service

    hook = MagicMock()
    output_path = tmp_path / "out.csv"
    output_path.write_bytes(b"")  # create file so stat() works
    download_from_blob("raw", "blob.csv", output_path, progress_hook=hook)

    call_kwargs = mock_container.download_blob.call_args.kwargs
    assert call_kwargs["progress_hook"] is hook


@patch("src.data.blob_io.settings")
@patch("src.data.blob_io.BlobServiceClient")
def test_get_container_client_raises_on_unknown_container(mock_bsc_cls, mock_settings):
    """Passing an unknown container name raises ValueError."""
    mock_settings.azure_storage_connection_string = FAKE_CONN_STR

    with pytest.raises(ValueError, match="Unknown container"):
        blob_io_module._get_container_client("unknown")  # type: ignore[arg-type]
