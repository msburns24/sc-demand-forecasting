# SCDF-18/19 — LightGBM Training Results (Offline)

**Date:** 2026-07-01 · **Stories:** SCDF-18 (model) + SCDF-19 (baseline) ·
**Mode:** Offline (no Azure ML / MLflow)

## TL;DR

- Single global LightGBM over all 10 stores / 30,490 series, evaluated
  **honestly** (leakage-free) on the last 28 days.
- **Deployable model (direct, horizon-safe): RMSE `2.17`** — **19% better** than
  the best naive baseline.
- **Best accuracy (recursive multi-step): RMSE `2.09`** — **22% better** than
  baseline, at the cost of a 28-step roll-forward at inference.
- The earlier `1.94` was **lag-leakage optimism** (oracle lags) and is not a
  deployable number.
- Canonical artifact: `models/lgbm_v1.pkl` (direct model) + `outputs/*.json`.

## Results ladder (full dataset, 853.7k validation rows)

| Model | Features | Val RMSE | MAPE* | vs baseline | Note |
| --- | --- | --- | --- | --- | --- |
| Naive seasonal, lag-7 | — | 2.6769 | 82.9% | — | best baseline (SCDF-19) |
| Naive seasonal, lag-28 | — | 2.8292 | 85.1% | −5.7% | (SCDF-19) |
| **Direct, horizon-safe (tuned)** | **25** | **2.1741** | 57.1% | **+18.8%** | **canonical `lgbm_v1.pkl`** |
| **Recursive multi-step** | **31** | **2.0868** | 56.3% | **+22.0%** | best honest, complex serving |
| _Leaky (oracle lags)_ | _25_ | _1.9375_ | _55.2%_ | _+27.6%_ | _optimistic — discarded_ |

*MAPE over non-zero actuals. "vs baseline" = RMSE improvement over naive lag-7.

## The honest-evaluation story

We forecast **28 days ahead** from a fixed cutoff. `lag_7` for horizon day `T+k`
needs sales at `T+k-7` — for `k>7` that's inside the horizon, not yet observed.
So `lag_7`, `lag_14`, and `shift(1)` rolling features **leak the future**. The
first model (1.94) used them → optimistic. We corrected this two ways:

1. **Direct, horizon-safe (`2.17`):** train only on features knowable at the
   cutoff — lags `>= 28`, `shift(28)` rollings, calendar, price. Simple to serve
   (one prediction per row). This is the canonical `lgbm_v1.pkl`; `python
   src/train.py` is horizon-safe by default.
2. **Recursive multi-step (`2.09`):** keep the full feature set (incl.
   `lag_7/14`) but roll the model forward day by day, feeding each prediction
   back as history before predicting the next day. This uses the short lags
   *legitimately* and recovered ~40% of the gap between the honest direct model
   and the optimistic ceiling. Requires a 28-step roll-forward at inference
   (`src/recursive.py`).

## Naive baseline (SCDF-19)

`src/baselines.py::naive_seasonal_forecast` — horizon-safe seasonal naive that
tiles the last observed `lag`-day block across the 28-day horizon (references
only pre-cutoff actuals, so it's a fair comparison). lag-7 (2.68) beats lag-28
(2.83): weekly seasonality dominates. **This is the reference point the 2.17/2.09
are measured against** — the model beats a trivial forecaster by ~19–22%.

## Hyperparameter tuning — honest finding

A 5-config subset sweep moved full-data RMSE by ~0.1% (2.1771 → 2.1741). **For
this problem, features dominate hyperparameters**; the horizon-safe `shift(28)`
rolling features recovered far more (2.28 → 2.17) than tuning did. Winning config
(tweedie 1.2, lr 0.03, num_leaves 127, n_estimators 3000 + early stop) is in
`config/model_config.yaml`.

## Which model to deploy?

**Recommend the direct horizon-safe model (`lgbm_v1.pkl`) for the initial ACI
service (SCDF-31)** — one `predict` call per request, trivial to serve, 2.17
RMSE. Treat recursive multi-step (2.09, +4%) as a documented accuracy upgrade:
worth it if the 28-step roll-forward is acceptable in the serving path.

## Reproduce

```bash
python -m src.features                 # build features (incl. horizon-safe rollings)
python src/train.py                    # honest direct model -> lgbm_v1.pkl (2.17)
python -m src.baselines                # naive lag-7 / lag-28 -> baseline_metrics.json
python -m src.recursive                # recursive multi-step -> recursive_metrics.json (2.09)
python src/train.py --full-features    # optimistic/leaky comparison only (1.94)
python -m pytest tests/ -q             # 66 passed
```

## Quality

- **Tests:** 66 passed — model, metrics, horizon-safe features, feature loader,
  naive baseline (incl. a leakage check), and recursive forecast (a lag-7 dummy
  model reproduces the naive baseline, validating the roll-forward).
- **Lint:** ruff clean. **Notebook:** `notebooks/04-model.ipynb` runs
  top-to-bottom and teaches the horizon-safe reasoning.

## Still on the table

1. **`days-since-last-sale`** and other intermittency features (strong for M5).
2. **Rolling-origin CV** (multiple 28-day folds) for lower-variance estimates.
3. **WRMSSE** — the actual M5 metric — for leaderboard-comparable scoring.
4. Per-segment evaluation across ABC-XYZ cells (SCDF-20) and SHAP (SCDF-22).

## Engineering notes

- Full 59M-row training OOM'd repeatedly; fixed by (1) reading only needed
  parquet columns, (2) **per-file** 32-bit downcast before concat, and (3) in the
  recursive path, dropping the full frame before training and keeping only the
  56-day recursion window (avoids stacking two full-frame split copies).
- Concatenated per-store parquet collapses `category` dtype → re-cast after
  concat (LightGBM needs `category`). UTF-8 console logging for Windows cp1252.

## Deviations from the Jira AC (intentional, per request)

Offline scope: no Azure ML tracking / registry / MLflow / run screenshot —
replaced by local `models/lgbm_v1.pkl` + `outputs/*.json`. SCDF-19 results saved
to `outputs/baseline_metrics.json` instead of MLflow.
