# SCDF-18 — LightGBM Training Results (Offline)

**Date:** 2026-06-30 · **Story:** SCDF-18 · **Mode:** Offline (no Azure ML / MLflow)

## TL;DR

- The honest, deployable model is a **single global LightGBM** over all 10
  stores / 30,490 series using **leakage-free, horizon-safe features**.
- **Honest 28-day-ahead validation RMSE: `2.17`.**
- An earlier number (1.94) was **optimistic** — it used lag features that aren't
  knowable across a real 28-day horizon. Removing that leakage costs ~12% RMSE.
  This report is the corrected, defensible result.
- Canonical artifact: `models/lgbm_v1.pkl` (22 MB, 646 trees) +
  `outputs/metrics.json`.

## The honest-evaluation story (why 2.17, not 1.94)

We forecast **28 days ahead** from a fixed cutoff. A feature is only legitimate
if it's knowable at the cutoff for the whole horizon. `lag_7` for horizon day
`T+k` needs sales at `T+k-7` — for `k>7` that's **inside** the horizon, i.e. not
yet observed. So `lag_7`, `lag_14`, and all `shift(1)` rolling features leak the
future. The first model used them, making it "1-step-ahead with oracle lags" —
optimistic.

The fix: train only on **horizon-safe** features — lags `>= 28`, rolling stats
computed on `shift(28)`, plus calendar and price (both known in advance). Code:
`HORIZON_SAFE_FEATURE_COLS` in `src/model.py`; `python src/train.py` is now
horizon-safe by default (`--full-features` opts back into the leaky set for the
optimistic comparison only).

## Results ladder (full dataset, 58.3M train / 853.7k val rows)

| Model | Features | Val RMSE | MAPE* | Trees | Note |
| --- | --- | --- | --- | --- | --- |
| Leaky (oracle lags) | 25 | 1.9375 | 55.2% | 629 | **Optimistic — not deployable** |
| Honest, thin | 19 | 2.2798 | 61.9% | 997† | lag_28 + calendar + price only |
| Honest, enriched | 25 | 2.1771 | 57.1% | 653 | + `shift(28)` rollings, lags 35/42 |
| **Honest, enriched + tuned** | **25** | **2.1741** | **57.1%** | **646** | **canonical (`lgbm_v1.pkl`)** |

*MAPE over non-zero actuals only. †Hit the 1,000-tree cap (still improving) —
the thin model was under-trained; the enriched models converge via early stop.

**Reading the ladder:**
- **Leakage cost:** 1.94 → 2.17 is ~12% — that's how optimistic the oracle-lag
  number was. The 2.17 is the number to trust and quote.
- **Feature engineering earned it back:** adding horizon-safe `shift(28)` rolling
  means/stds recovered most of the gap (2.28 → 2.17).
- **Tuning was negligible:** 2.1771 → 2.1741 (~0.1%). Honest takeaway — for this
  problem, **features dominate hyperparameters**; the defaults were already fine.

## Hyperparameter sweep (3-store subset, ranking only)

Ran on CA_1+TX_1+WI_1 (17.5M rows) to rank configs cheaply. Subset RMSE isn't
comparable to full-data RMSE (easier subset); it's for ordering only.

| Config | Subset RMSE | Best iter |
| --- | --- | --- |
| base (lr0.05, leaves63, n1500) | 1.9535 | 414 |
| leaves127 | 1.9470 | 564 |
| lr0.03, leaves127, n3000 | 1.9488 | 764 |
| **lr0.03, leaves127, n3000, tweedie1.2** | **1.9466** | 731 |
| lr0.03, leaves255, mcs50, n3000 | 1.9496 | 455 |

Winner baked into `config/model_config.yaml`. The spread is ~0.35% — confirms
tuning headroom is small here.

## Canonical model

- `objective: tweedie` (`variance_power 1.2`), `lr 0.03`, `num_leaves 127`,
  `n_estimators 3000` + early stopping (stopped at 646), `min_child_samples 100`,
  `feature_fraction`/`bagging_fraction 0.8`.
- 25 horizon-safe features; `models/lgbm_v1.pkl` 22 MB (well under the 1 GB
  Docker target).

### Top features (horizon-safe model, split importance)

`item_id` ≫ `week_of_year` > `price_relative_to_category_mean` > `store_id` >
`sell_price` > **`rolling_mean_28_lag28`** > `dept_id` > `month` >
`rolling_std_28_lag28` > `rolling_mean_7_lag28`. The new horizon-safe rolling
features land in the top ~10 — they're pulling real weight, which is why they
recovered the leakage gap.

## Reproduce

```bash
python -m src.features                 # build features (incl. horizon-safe rollings)
python src/train.py                    # honest model (horizon-safe by default) -> lgbm_v1.pkl
python src/train.py --full-features    # optimistic/leaky comparison only
python -m pytest tests/ -q             # 59 passed
ruff check src/ tests/                 # clean
```

## Quality

- **Tests:** 59 passed (incl. `TestTrainModel`, `TestMetrics`,
  `TestLoadFeatures`, `TestHorizonSafeFeatures`).
- **Lint:** ruff clean. **Notebook:** `notebooks/04-model.ipynb` runs
  top-to-bottom and now teaches the leakage/horizon-safe reasoning.

## Is this the best we can do?

It's an honest, solid baseline-quality model — but not the ceiling. With
diminishing returns and added complexity, further gains could come from:

1. **A naive baseline (SCDF-19)** — still the most important missing piece; 2.17
   has no reference point until we know what a trivial model scores.
2. **Recursive multi-step** — feed predictions back so `lag_7`/`lag_14` become
   usable *legitimately*, recovering signal the horizon-safe cut discards.
3. **`days-since-last-sale`** and other intermittency features (strong for M5).
4. **Rolling-origin CV** (multiple 28-day folds) instead of one window — lower
   variance, more trustworthy.
5. **WRMSSE** — the actual M5 metric — for leaderboard-comparable scoring.

## Deviations from the Jira AC (intentional, per request)

Offline scope: no Azure ML tracking, model registry, MLflow, or run screenshot —
replaced by local `models/lgbm_v1.pkl` + `outputs/metrics.json`.

## Engineering notes

- Full 59M-row training OOM'd three times; fixed by (1) reading only the 27/33
  needed parquet columns, (2) **per-file** 32-bit downcast before concat (the
  bulk `select_dtypes` cast itself built a 7 GiB block), and (3) the temporal
  split's `.copy()` needing contiguous blocks — all now comfortably in memory.
- Concatenated per-store parquet collapses `category` dtype to `object`; the 5
  ID columns are re-cast after concat (LightGBM needs `category`).
- Forced UTF-8 console logging so the `→` split-log char doesn't crash on
  Windows cp1252.
