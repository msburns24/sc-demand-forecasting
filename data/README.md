# Data Directory

Raw data files are excluded from version control (`.gitignore` covers `data/*.csv`).
This directory is intentionally empty in the repo. All pipeline code reads from
Azure Blob Storage — not from local disk.

## Getting the Data

### Step 1 — Download from Kaggle

The M5 Competition dataset is available on Kaggle:

```
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
```

Download and extract the following files into this directory:

| File | Description |
| ---- | ----------- |
| `sales_train_evaluation.csv` | Daily unit sales for 30,490 SKU-store combinations (1,941 days) |
| `calendar.csv` | Date metadata: events, SNAP days, week IDs |
| `sell_prices.csv` | Weekly sell prices per item per store |

A Kaggle account and competition acceptance is required. Install the Kaggle CLI
and run:

```bash
pip install kaggle
kaggle competitions download -c m5-forecasting-accuracy -p data/
unzip data/m5-forecasting-accuracy.zip -d data/
```

### Step 2 — Upload to Azure Blob Storage

Once the files are in `data/`, upload them to the `raw` container in Azure Blob
Storage using the helper in `src/data/blob_io.py`:

```bash
python -c "from src.data.blob_io import upload_files_to_blob; upload_files_to_blob()"
```

This calls `upload_to_blob()` for each of the three files and places them at
`stscdfdata/raw/<filename>`.

Requires a `.env` file at the repo root with `AZURE_STORAGE_CONNECTION_STRING`
set. See `.env.example` for the full list of required variables.

### Step 3 — Verify

After uploading, confirm the files landed correctly:

```bash
python -m src.data.validate
```

The validation script reads each file directly from Blob and checks row counts,
date ranges, null rates, and the expected 30,490 SKU-store combination count.
