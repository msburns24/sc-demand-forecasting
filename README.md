# Supply Chain Demand Forecasting & Inventory Optimization

((Brief introduction))

## Problem

Retailers managing thousands of SKUs face a fundamental tension: overstocking
ties up working capital, while under-stocking drives lost sales and customer
attrition. A uniform inventory policy fails both problems simultaneously —
treating a high-velocity staple the same as a slow-moving seasonal item wastes
capital on one and creates stockouts on the other. This project builds a
data-driven inventory policy engine that segments SKUs by revenue contribution
and demand variability, forecasts demand with a production-grade LightGBM
pipeline, and translates forecast error distributions into differentiated
safety stock recommendations.

## Approach

The solution is structured as three sequential layers, each building on the
last:

1. **SKU Segmentation (ABC-XYZ):** Not all SKUs deserve the same inventory
   policy. ABC classification groups items by cumulative revenue contribution
   (A: top 70%, B: next 20%, C: remainder). XYZ classification groups by demand
   variability using coefficient of variation on weekly sales (X: CV < 0.5, Y:
   0.5–1.0, Z: > 1.0). The resulting 3×3 matrix drives every downstream policy
   decision — from service level targets to make-to-stock vs. make-to-order
   classification.

2. **Demand Forecasting:** A single LightGBM model trained across all 30,490
   SKU-store combinations on the M5 competition dataset. Features include lag
   sales (7, 14, 28-day), rolling statistics, calendar effects (day-of-week,
   holidays, SNAP days), and price dynamics. Training and experiment tracking
   run through Azure ML; the registered model is consumed directly by the
   inference service. Performance is evaluated per ABC-XYZ segment — aggregate
   MAPE obscures the meaningful variation between predictable AX items and
   erratic CZ items.

3. **Safety Stock Optimization:** Forecast error distributions are computed per
   segment and fed into a safety stock formula parameterized by service level.
   Service level targets are derived from the ABC-XYZ policy matrix (99% for
   AX/BX/AY, down to 85% for CZ). The output is a policy recommendation table
   mapping every segment to a safety stock target, expected days of supply, and
   MTS/MTO classification — with an explicit analysis of the holding vs.
   stockout cost trade-off.

## Architecture

```
M5 Dataset (Kaggle)
      │
      ▼
Azure Blob Storage (raw/)
      │
      ▼
Feature Pipeline ──────────────────►  Azure Blob Storage (processed/)
(src/features.py)                                │
                                                 ▼
                                      Azure ML — Model Training
                                   (LightGBM + experiment tracking)
                                                 │
                                                 ▼
                                      Azure ML Model Registry
                                                 │
                                                 ▼
                                    Azure Container Registry (ACR)
                                                 │
                                                 ▼
                                    Azure Container Instances (ACI)
                                      FastAPI inference endpoint
                                   POST /predict → forecast + policy
```

The pipeline is cloud-native end-to-end: raw data lives in Blob Storage,
features are written back to Blob as Parquet after engineering, training runs
are tracked in Azure ML, and the trained model is served via a containerized
FastAPI service deployed to ACI. Configuration is managed via
`pydantic-settings` with no secrets hardcoded; CI/CD via GitHub Actions
rebuilds and pushes the inference image on every merge to main.

A few notes on the architecture section: the ASCII diagram is a reasonable
placeholder but you'll want to replace it with the Inkscape SVG once that's
done (the SCDF-27 architecture diagram task). The prose paragraph after it is
doing useful work — it summarizes the "why cloud-native" story in a way that a
hiring manager reading quickly will pick up on without needing to parse the
diagram.

## Key Results

TBD.
