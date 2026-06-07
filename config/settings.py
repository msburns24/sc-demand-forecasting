"""
pydantic-settings to load Azure connection strings and container names from
environment variables (never hardcoded).
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).parent.parent
ENV_PATH = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
    )

    # Azure Storage
    azure_storage_connection_string: str
    azure_storage_container_raw: str = "raw"
    azure_storage_container_processed: str = "processed"

    # Azure Identity
    azure_tenant_id: str
    azure_client_id: str
    azure_client_secret: str

    # Azure Subscription / Resource Group
    azure_subscription_id: str
    azure_resource_group: str = "rg-scdf-dev"

    # Azure ML
    azure_ml_workspace_name: str = "aml-scdf"

    # Azure Container Registry
    azure_container_registry: str = "acrscdf.azurecr.io"

    # Model config
    model_name: str = "lgbm-sc-forecast"
    model_version: str = "1"
    lead_time_days: int = 7

    # Kaggle
    kaggle_api_token: str


settings = Settings()  # type: ignore[call-arg]

__all__ = ["settings"]

if __name__ == "__main__":
    print(settings)
