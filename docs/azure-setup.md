# Azure Setup

This document records the Azure resources provisioned for this project and the
commands used to create them. All resources live in a single resource group for
easy teardown.

## Prerequisites

- Azure CLI installed (`az --version`)
- Logged in: `az login --tenant <your-tenant-id>`
- Correct subscription set:
  `az account set --subscription <your-subscription-id>`
- ML extension installed: `az extension add --name ml`

## Resources

| Resource           | Name          | Type                             | Purpose                                    |
| :----------------- | :------------ | :------------------------------- | :----------------------------------------- |
| Resource Group     | `rg-scdf-dev` | Resource Group                   | Container for all project resources        |
| Storage Account    | `stscdfdata`  | Storage Account (Standard LRS)   | Raw and processed data layer               |
| Blob Container     | `raw`         | Blob Container                   | M5 source files                            |
| Blob Container     | `processed`   | Feature-engineered Parquet files |                                            |
| ML Workspace       | `aml-scdf`    | Azure ML Workspace               | Experiment tracking and model registry     |
| Container Registry | `acrscdf`     | ACR (Basic)                      | Docker image storage for inference service |
| Service Principal  | `sp-scdf-dev` | Service Principal                | Programmatic access for local dev and CI   |

## Provisioning Commands

### Resource Group

```bash
az group create --name rg-scdf-dev --location eastus
```

### Storage Account and Containers

```bash
az storage account create \
  --name stscdfdata \
  --resource-group rg-scdf-dev \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2

az storage container create --name raw --account-name stscdfdata
az storage container create --name processed --account-name stscdfdata
```

### Azure ML Workspace

```bash
STORAGE_ID=$(az storage account show \
  --name stscdfdata \
  --resource-group rg-scdf-dev \
  --query id \
  --output tsv)

az ml workspace create \
  --name aml-scdf \
  --resource-group rg-scdf-dev \
  --location eastus \
  --storage-account $STORAGE_ID
```

### Azure Container Registry

```bash
az acr create \
  --name acrscdf \
  --resource-group rg-scdf-dev \
  --sku Basic \
  --location eastus
```

### Service Principal

```bash
az ad sp create-for-rbac \
  --name sp-scdf-dev \
  --role Contributor \
  --scopes /subscriptions/<your-subscription-id>/resourceGroups/rg-scdf-dev \
  --sdk-auth
```

Save the JSON output — this becomes the `AZURE_CREDENTIALS` GitHub Actions
secret. It cannot be retrieved again after this step.

### ACR Role Assignment

Note: `--assignee-object-id` is required here; `--assignee` with the appId
returns a Bad Request from the Azure API. Use literal values rather than shell
variables if the command fails.

```bash
az role assignment create \
  --assignee-object-id <sp-object-id> \
  --assignee-principal-type ServicePrincipal \
  --role AcrPush \
  --scope /subscriptions/<your-subscription-id>/resourceGroups/rg-scdf-dev/providers/Microsoft.ContainerRegistry/registries/acrscdf
```

## Environment Variables

Required in `.env` (see `.env.example` for the full list):

```
AZURE_STORAGE_CONNECTION_STRING=...   # from: az storage account show-connection-string
AZURE_STORAGE_CONTAINER_RAW=raw
AZURE_STORAGE_CONTAINER_PROCESSED=processed
AZURE_ML_WORKSPACE=aml-scdf
AZURE_ML_RESOURCE_GROUP=rg-scdf-dev
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
ACR_LOGIN_SERVER=acrscdf.azurecr.io
AZURE_CLIENT_ID=<sp-app-id>
AZURE_CLIENT_SECRET=<sp-password>
AZURE_TENANT_ID=<your-tenant-id>
```

## Teardown

To delete all resources at once:

```bash
az group delete --name rg-scdf-dev --yes
```
