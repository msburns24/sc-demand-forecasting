# Jira Project: Supply Chain Demand Forecasting & Inventory Optimization

## Project Configuration

| Field            | Value                                         |
| ---------------- | --------------------------------------------- |
| **Project Name** | Supply Chain Demand Forecasting               |
| **Project Key**  | `SCDF`                                        |
| **Project Type** | Scrum                                         |
| **GitHub Repo**  | `msburns24/sc-demand-forecasting` (suggested) |

### Board Columns

Backlog → To Do → In Progress → In Review → Done

### Custom Fields (add under Project Settings → Fields)

| Field                  | Type       | Values                                                                                                                            |
| ---------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Portfolio Signal**   | Labels     | `Domain Knowledge`, `Statistical Rigor`, `MLOps`, `Business Framing`, `Data Engineering`, `Visualization`, `Cloud Infrastructure` |
| **Portfolio Artifact** | Checkbox   | Yes / No — marks stories that produce something visible in the writeup                                                            |
| **Sprint**             | (built-in) | Sprint 1–7                                                                                                                        |

### Labels (project-wide)

`forecasting`, `segmentation`, `safety-stock`, `feature-engineering`,
`evaluation`, `visualization`, `writeup`, `infra`, `testing`, `documentation`,
`azure`, `docker`, `deployment`

---

## Epics

| Epic Key | Epic Name                      | Goal                                                                                           | Sprint(s)  |
| -------- | ------------------------------ | ---------------------------------------------------------------------------------------------- | ---------- |
| SCDF-E1  | Project Setup & Infrastructure | Repo, environment, Azure Blob Storage, data ingestion, documentation scaffolding               | Sprint 1   |
| SCDF-E2  | ABC-XYZ Segmentation           | Classify all SKUs by value (ABC) and demand variability (XYZ); produce MTS/MTO decision matrix | Sprint 2   |
| SCDF-E3  | Demand Forecasting Pipeline    | Feature engineering, LightGBM training with Azure ML tracking, evaluation by segment           | Sprint 3–4 |
| SCDF-E4  | Safety Stock Optimization      | Translate forecast error into safety stock policy; quantify holding vs. stockout trade-off     | Sprint 5   |
| SCDF-E5  | Portfolio Packaging            | Quarto writeup, visualizations, repo polish, site deployment                                   | Sprint 6   |
| SCDF-E6  | Azure Deployment               | Dockerize inference service, push to ACR, deploy to ACI, expose live endpoint                  | Sprint 7   |

---

## Sprint 1 — Project Setup & Infrastructure

**Epic:** SCDF-E1  
**Goal:** Repo is structured like a real ML project, Azure Blob Storage is
wired up as the data layer, data is loaded and validated, business problem is
documented.  
**Duration:** 1 week

### SCDF-7 · Story · Set up repository structure

**Description:**

Initialize the GitHub repo with a structure that signals ML engineering
discipline. This is visible to anyone who clicks the GitHub link from the
portfolio.

**Acceptance Criteria:**

- [ ] Repo contains: `data/`, `notebooks/`, `src/`, `tests/`, `docs/`,
      `docker/`, `config/`, `.gitignore` for data files and virtual envs
- [ ] `requirements.txt` (or `pyproject.toml`) committed with pinned versions —
      includes `azure-storage-blob`, `azure-ai-ml`, `lightgbm`, `mlflow`,
      `fastapi`, `uvicorn`
- [ ] `config/settings.py` using `pydantic-settings` to load Azure connection
      strings and container names from environment variables (never hardcoded)
- [ ] `.env.example` committed showing required env vars without values; `.env`
      in `.gitignore`
- [ ] Pre-commit hooks configured (black, ruff)
- [ ] Initial commit message follows conventional commits format

**Labels:** `infra`, `azure`  
**Portfolio Signal:** MLOps, Cloud Infrastructure  
**Portfolio Artifact:** No  
**Estimate:** 3h

### SCDF-8 · Story · Provision Azure infrastructure

**Description:**

Set up the Azure resources this project will use: a Storage Account for data,
an Azure ML workspace for experiment tracking and model registry, and an Azure
Container Registry for the inference image. All provisioned via Azure CLI so
the setup is reproducible and documentable.

**Acceptance Criteria:**

- [ ] Resource group created: `rg-scdf-dev`
- [ ] Storage Account created: `stscdfdata` with two containers — `raw` (M5
      source files) and `processed` (feature-engineered parquet files)
- [ ] Azure ML workspace created: `aml-scdf` linked to the storage account
- [ ] Azure Container Registry created: `acrscdf` (Basic tier is fine)
- [ ] All resource names and connection strings documented in
      `docs/azure-setup.md` — commands used to provision, not just the results
- [ ] Service principal or managed identity configured for programmatic access;
      credentials stored in `.env` locally and documented for CI use

**Labels:** `infra`, `azure`  
**Portfolio Signal:** Cloud Infrastructure, MLOps  
**Portfolio Artifact:** No  
**Estimate:** 4h

### SCDF-9 · Story · Download M5 dataset and upload to Azure Blob Storage

**Description:**

Download the M5 Competition dataset from Kaggle and upload it to the `raw`
container in Azure Blob Storage. All downstream pipeline code reads from Blob —
not from local disk. This makes the pipeline portable and cloud-native from day
one.

**Acceptance Criteria:**

- [ ] All M5 files uploaded to `stscdfdata/raw/`: `sales_train_evaluation.csv`,
      `calendar.csv`, `sell_prices.csv`
- [ ] `src/data/blob_io.py` implements
      `upload_to_blob(local_path, container, blob_name)` and
      `download_from_blob(container, blob_name, local_path)` using
      `azure-storage-blob`
- [ ] Validation script reads files directly from Blob and checks: row counts,
      date ranges, null rates per column, SKU-store combination count (~30,490)
- [ ] `docs/data-dictionary.md` written with column descriptions, data types,
      known quirks
- [ ] Raw data excluded from git; `data/README.md` explains Kaggle download +
      Blob upload steps for reproducibility
- [ ] Unit tests mock the Blob client to test upload/download logic without
      live Azure calls

**Labels:** `documentation`, `infra`, `azure`  
**Portfolio Signal:** Data Engineering, Cloud Infrastructure  
**Portfolio Artifact:** No  
**Estimate:** 4h

### SCDF-10 · Story · Exploratory data analysis

**Description:**

Purposeful EDA that builds the business case for the project — not just
distribution plots, but observations that motivate the segmentation and
forecasting approach. This becomes the "Situation" section of the STAR writeup.

**Acceptance Criteria:**

- [ ] Notebook covers: sales volume distribution (log scale), zero-sales rate
      by category, temporal trends (weekly/seasonal patterns), price variation,
      SNAP day sales lift, top-10 vs. bottom-10 SKUs by revenue
- [ ] At least one finding that directly motivates ABC-XYZ segmentation (e.g.,
      "top 20% of SKUs account for 78% of revenue")
- [ ] At least one finding that motivates variable safety stock (e.g., "CV
      ranges from 0.1 to 2.4 across SKUs")
- [ ] Notebook is clean: markdown narrative cells explain the business
      implication of each finding, not just the chart

**Labels:** `documentation`, `visualization`  
**Portfolio Signal:** Business Framing, Domain Knowledge  
**Portfolio Artifact:** Yes — 2–3 EDA charts will appear in the writeup  
**Estimate:** 6h

### SCDF-11 · Story · Write project README with business framing

**Description:**

The README is the first thing a hiring manager sees when they click the GitHub
link. It must lead with the business problem and outcome, not the installation
instructions. The Azure stack should be visible but not the focus — it's the
infrastructure that makes the solution real.

**Acceptance Criteria:**

- [ ] README structure: Problem → Approach → Architecture → Key Results
      (placeholder until Sprint 5) → How to Run → Project Structure
- [ ] Problem statement is written for a non-technical reader: what is the
      business cost of bad inventory policy?
- [ ] Approach section names the methods without jargon-dumping: ABC-XYZ,
      LightGBM forecasting, safety stock optimization
- [ ] Architecture section: one-line description of the Azure stack — Blob
      Storage (data) → Azure ML (training + registry) → ACR + ACI (serving)
- [ ] "How to Run" section covers local setup with `.env` configuration and
      works from a clean clone

**Labels:** `documentation`  
**Portfolio Signal:** Business Framing, Cloud Infrastructure  
**Portfolio Artifact:** No  
**Estimate:** 2h

---

## Sprint 2 — ABC-XYZ Segmentation

**Epic:** SCDF-E2  
**Goal:** Every SKU-store combination has an ABC class, an XYZ class, and a
derived MTS/MTO recommendation. Hero visualization #1 is complete.  
**Duration:** 1 week

### SCDF-12 · Story · Implement ABC classification

**Description:**

Classify SKU-store combinations by cumulative revenue contribution using the
Pareto principle. This is a standard supply chain technique — implementing it
cleanly in `src/` (not just a notebook) signals domain credibility.

**Acceptance Criteria:**

- [ ] Function
      `classify_abc(df, revenue_col, a_threshold=0.70, b_threshold=0.90)` in
      `src/segmentation.py`
- [ ] A items: top ~70% of cumulative revenue; B items: next ~20%; C items:
      remainder
- [ ] Output: original dataframe with `abc_class` column added
- [ ] Unit tests in `tests/test_segmentation.py`: verify cumulative thresholds,
      verify all items classified, verify deterministic output

**Labels:** `segmentation`, `testing`  
**Portfolio Signal:** Domain Knowledge, Statistical Rigor  
**Portfolio Artifact:** No  
**Estimate:** 4h

### SCDF-13 · Story · Implement XYZ classification

**Description:**

Classify SKU-store combinations by demand variability using coefficient of
variation (CV) on weekly aggregated demand. XYZ is the variability dimension of
the segmentation matrix.

**Acceptance Criteria:**

- [ ] Function
      `classify_xyz(df, demand_col, date_col, x_threshold=0.5, y_threshold=1.0)`
      in `src/segmentation.py`
- [ ] CV calculated on weekly-aggregated demand per SKU-store combination
- [ ] X items: CV < 0.5 (predictable); Y items: 0.5 ≤ CV < 1.0 (moderate); Z
      items: CV ≥ 1.0 (erratic)
- [ ] Thresholds are parameterized, not hardcoded — signals good engineering
      practice
- [ ] Unit tests: verify CV calculation, verify threshold behavior at
      boundaries, verify handling of zero-demand series

**Labels:** `segmentation`, `testing`  
**Portfolio Signal:** Domain Knowledge, Statistical Rigor  
**Portfolio Artifact:** No  
**Estimate:** 4h

### SCDF-14 · Story · Build ABC-XYZ matrix visualization

**Description:**

The 3×3 ABC-XYZ heatmap is Hero Visual #1 — it's the first thing in the writeup
after the TL;DR callout. It must communicate the business insight (not just the
counts) at a glance.

**Acceptance Criteria:**

- [ ] Heatmap shows count and % of SKUs per cell, plus average revenue per cell
- [ ] Color encodes strategic priority: AX (dark/high) through CZ (light/low)
- [ ] Each cell annotated with the MTS/MTO recommendation and suggested service
      level tier
- [ ] Chart uses portfolio color palette (consistent with other project
      visuals)
- [ ] Saved as both PNG (for portfolio) and the generating code is in a clean
      notebook cell

**Labels:** `segmentation`, `visualization`  
**Portfolio Signal:** Business Framing, Visualization  
**Portfolio Artifact:** Yes — this is Hero Visual #1  
**Estimate:** 4h

#### Task - ABC-XYZ Heatmap

Polish the programmatic heatmap output in Inkscape for portfolio use: apply
Catppuccin palette (Mauve → Lavender → Surface1 gradient across the priority
axis), refine typography to match site font, add MTS/MTO annotation labels with
consistent styling. This is an SVG edit of the matplotlib output, not a
from-scratch design.

### SCDF-15 · Story · Define MTS/MTO decision rules and document

**Description:**

Translate the ABC-XYZ matrix into explicit, defensible inventory policy
recommendations. This is the "Task" section of the STAR writeup — what were you
actually trying to decide?

**Acceptance Criteria:**

- [ ] Decision table documented in `docs/inventory-policy.md`:
  - [ ] AX, BX, AY → Make-to-Stock, high service level (99%)
  - [ ] BY, CX, AZ → Make-to-Stock, standard service level (95%)
  - [ ] BZ, CY → Make-to-Stock, low service level (90%) or review for MTO
  - [ ] CZ → Make-to-Order or discontinue
- [ ] Each rule includes the business rationale (not just the label)
- [ ] A `classify_mts_mto(abc, xyz)` function in `src/segmentation.py` encodes
      the rules
- [ ] Unit tests for the decision function

**Labels:** `segmentation`, `documentation`, `testing`  
**Portfolio Signal:** Business Framing, Domain Knowledge  
**Portfolio Artifact:** Yes — decision table appears in writeup  
**Estimate:** 3h

---

## Sprint 3 — Feature Engineering & Model Training

**Epic:** SCDF-E3  
**Goal:** Feature pipeline is modular and tested; LightGBM model trained on
full dataset; baseline comparison complete.  
**Duration:** 1 week

### SCDF-16 · Story · Build feature engineering pipeline

**Description:**

The feature pipeline is the most technically defensible part of the project —
it shows you understand time series ML, not just "fit a model." Build it in
`src/features.py` so it's testable and reusable. Processed features are written
back to Azure Blob Storage (`processed` container) as Parquet, so training
reads from Blob end-to-end.

**Acceptance Criteria:**

- [ ] `build_features(df, calendar_df, prices_df)` function in
      `src/features.py`
- [ ] Lag features: 7, 14, 28-day sales lags
- [ ] Rolling statistics: 7, 28-day rolling mean and std
- [ ] Calendar features: day of week, week of year, month, is*weekend,
      is_holiday, is_snap*{CA,TX,WI}
- [ ] Price features: sell_price, price_change (vs. prior week),
      price_relative_to_category_mean
- [ ] Categorical encodings: item_id, store_id, dept_id, cat_id, state_id
      (label-encoded)
- [ ] Output written to `stscdfdata/processed/features.parquet` via
      `blob_io.py`
- [ ] Function is pure (no side effects on input dataframe)
- [ ] Unit tests: verify no data leakage, verify output shape, verify Blob
      write is called with correct args (mocked)

**Labels:** `feature-engineering`, `testing`, `azure`  
**Portfolio Signal:** Statistical Rigor, Data Engineering, Cloud Infrastructure
**Portfolio Artifact:** No  
**Estimate:** 8h

### SCDF-17 · Story · Implement train/validation split

**Description:**

M5 has a natural evaluation setup — the last 28 days are the test set.
Implement a proper temporal split that prevents leakage and mirrors how the
model would be used in production.

**Acceptance Criteria:**

- [ ] `split_train_val(df, val_days=28)` in `src/model.py`
- [ ] No random shuffling — strictly temporal: all data before cutoff date is
      train, after is validation
- [ ] Document why this matters (data leakage) in a notebook markdown cell —
      this is interview content
- [ ] Verify split produces no overlap between train and validation sets

**Labels:** `feature-engineering`, `testing`  
**Portfolio Signal:** Statistical Rigor  
**Portfolio Artifact:** No  
**Estimate:** 2h

### SCDF-18 · Story · Train LightGBM model with Azure ML tracking

**Description:**

Train a single LightGBM model across all SKU-store combinations (the M5 winning
approach). Use Azure ML to track experiments and register the trained model —
this replaces a local MLflow setup with a cloud-native equivalent that signals
real production awareness.

**Acceptance Criteria:**

- [ ] `train_model(X_train, y_train, params)` in `src/model.py`
- [ ] Azure ML experiment tracking: log params, metrics (RMSE, MAPE), and
      register the trained model to the Azure ML model registry as
      `lgbm-sc-forecast:v1`
- [ ] Training reads features from `stscdfdata/processed/features.parquet` via
      Blob
- [ ] `params` sourced from `config/model_config.yaml`, not hardcoded
- [ ] Training script runnable from CLI: `python src/train.py`
- [ ] Trained model also saved locally to `models/lgbm_v1.pkl` for Docker
      packaging (excluded from git)
- [ ] Notebook documents: why single model vs. per-series, why LightGBM vs.
      alternatives, key hyperparameter choices — these are interview talking
      points
- [ ] Screenshot of Azure ML experiment run (metrics dashboard) included in
      `docs/` — becomes a portfolio visual

**Labels:** `forecasting`, `azure`  
**Portfolio Signal:** MLOps, Cloud Infrastructure, Statistical Rigor  
**Portfolio Artifact:** Yes — Azure ML experiment screenshot appears in
writeup  
**Estimate:** 7h

### SCDF-19 · Story · Implement naive seasonal baseline

**Description:**

Every model needs a baseline to beat. A naive seasonal baseline (same day last
week / last year) is the right comparison for M5 — and documenting that you
chose it deliberately over a simpler mean baseline signals forecasting
literacy.

**Acceptance Criteria:**

- [ ] `naive_seasonal_forecast(df, lag=7)` in `src/baselines.py`
- [ ] Also implement `naive_seasonal_forecast(df, lag=28)` for monthly
      seasonality
- [ ] RMSE and MAPE calculated for both baselines on the validation set
- [ ] Results stored in MLflow alongside the LightGBM run for direct comparison

**Labels:** `forecasting`, `evaluation`  
**Portfolio Signal:** Statistical Rigor  
**Portfolio Artifact:** No  
**Estimate:** 3h

---

## Sprint 4 — Model Evaluation by Segment

**Epic:** SCDF-E3 (continued)  
**Goal:** Model performance is understood at the ABC-XYZ cell level. CZ
behavior is characterized and explained, not hidden.  
**Duration:** 1 week

### SCDF-20 · Story · Evaluate model performance by ABC-XYZ segment

**Description:**

Aggregate-level MAPE is a weak signal. Breaking performance down by ABC-XYZ
segment is the finding that makes this project senior — it shows you understand
_why_ the model performs differently on different inventory types.

**Acceptance Criteria:**

- [ ] RMSE and MAPE calculated per ABC-XYZ cell on the validation set
- [ ] Results table: 9 cells × 2 metrics, plus count of SKUs per cell
- [ ] Key finding documented: AX items should have lowest MAPE; CZ items
      predictably highest
- [ ] Comparison vs. naive seasonal baseline per cell (LightGBM vs. baseline
      improvement by segment)

**Labels:** `evaluation`, `visualization`  
**Portfolio Signal:** Statistical Rigor, Business Framing  
**Portfolio Artifact:** Yes — segment performance table appears in writeup  
**Estimate:** 5h

### SCDF-21 · Story · Residual analysis and forecast error characterization

**Description:**

Safety stock calculation in Sprint 5 requires knowing the distribution of
forecast errors per segment. This story produces the statistical inputs for the
optimization, and demonstrates you think about downstream use of model outputs.

**Acceptance Criteria:**

- [ ] For each ABC-XYZ cell: mean error (bias), σ of errors, distribution shape
      (normal? skewed?)
- [ ] Plot error distributions per cell — are AX errors tighter than CZ? (They
      should be)
- [ ] `src/evaluation.py` function
      `compute_error_stats(y_true, y_pred, segments)` returns a dataframe of σ
      per cell
- [ ] Document any systematic bias (over- or under-forecasting) by segment —
      this is a finding, not a failure

**Labels:** `evaluation`, `visualization`  
**Portfolio Signal:** Statistical Rigor  
**Portfolio Artifact:** Yes — error distribution plots appear in writeup  
**Estimate:** 5h

### SCDF-22 · Story · Feature importance analysis

**Description:**

Which features matter most? A SHAP-based importance analysis shows you think
about model interpretability, not just accuracy — relevant for supply chain
contexts where stakeholders want to understand model behavior.

**Acceptance Criteria:**

- [ ] SHAP values computed for a sample of predictions (not full dataset —
      computational cost)
- [ ] Bar chart: top 15 features by mean |SHAP|
- [ ] At least one supply chain insight documented: e.g., "SNAP days are a
      top-5 feature for FOODS category, but irrelevant for HOUSEHOLD"
- [ ] Chart uses portfolio color palette

**Labels:** `evaluation`, `visualization`  
**Portfolio Signal:** Business Framing, Domain Knowledge  
**Portfolio Artifact:** Yes — feature importance chart appears in writeup  
**Estimate:** 4h

---

## Sprint 5 — Safety Stock Optimization

**Epic:** SCDF-E4  
**Goal:** Every SKU-store combination has a data-driven safety stock
recommendation. Hero Visual #2 (holding vs. stockout trade-off curve) is
complete. MTS/MTO policy table is final.  
**Duration:** 1 week

### SCDF-23 · Story · Implement safety stock calculation

**Description:**

Translate forecast error distributions into safety stock recommendations using
the standard formula, parameterized by service level per ABC-XYZ segment.

**Acceptance Criteria:**

- [ ] `compute_safety_stock(sigma_demand, lead_time, service_level)` in
      `src/inventory.py`
- [ ] Formula: `SS = Z(service_level) × σ_demand × √(lead_time)`
- [ ] Z-scores sourced from scipy, not hardcoded
- [ ] Service levels by segment: AX/BX/AY = 99%, BY/CX/AZ = 95%, BZ/CY = 90%,
      CZ = 85%
- [ ] Lead time is a configurable parameter (default: 7 days)
- [ ] Unit tests: verify Z-score lookup, verify formula arithmetic, verify
      output shape matches input

**Labels:** `safety-stock`, `testing`  
**Portfolio Signal:** Domain Knowledge, Statistical Rigor  
**Portfolio Artifact:** No  
**Estimate:** 4h

### SCDF-24 · Story · Build holding cost vs. stockout cost trade-off analysis

**Description:**

Hero Visual #2 — the curve that connects model outputs to a business decision.
This is the piece that makes the writeup unmistakably senior: you're not just
reporting MAPE, you're showing what service level to target and why.

**Acceptance Criteria:**

- [ ] For a representative AX item and a representative CZ item: plot safety
      stock units vs. service level (85%–99.9%)
- [ ] Overlay estimated annual holding cost (assume $X/unit/year — document the
      assumption) and expected annual stockout cost (assume $Y/stockout —
      document)
- [ ] Mark the optimal service level where total cost is minimized
- [ ] Chart clearly shows why AX and CZ items have different optimal service
      levels
- [ ] Assumptions box on the chart (or in caption): makes the analysis
      reproducible and intellectually honest

**Labels:** `safety-stock`, `visualization`  
**Portfolio Signal:** Business Framing, Domain Knowledge, Statistical Rigor  
**Portfolio Artifact:** Yes — this is Hero Visual #2  
**Estimate:** 6h

### SCDF-25 · Story · Generate final policy recommendation table

**Description:**

The deliverable a real supply chain team would actually use — a table mapping
every ABC-XYZ segment to its inventory policy, target service level, expected
safety stock days, and MTS/MTO classification.

**Acceptance Criteria:**

- [ ] Output table: ABC-XYZ cell → MTS/MTO → Service Level → Avg Safety Stock
      (days of supply) → Avg MAPE → SKU Count
- [ ] Saved as `outputs/inventory_policy_recommendations.csv`
- [ ] Notebook cell with a formatted display version (for writeup)
- [ ] One-paragraph executive summary written: "Based on this analysis, we
      recommend..."

**Labels:** `safety-stock`, `documentation`  
**Portfolio Signal:** Business Framing  
**Portfolio Artifact:** Yes — policy table appears in writeup  
**Estimate:** 3h

### SCDF-26 · Story · Sensitivity analysis — lead time assumptions

**Description:**

What happens to safety stock requirements if lead time increases by 50%? This
tests robustness and signals the kind of thinking a senior DS does before
presenting recommendations.

**Acceptance Criteria:**

- [ ] Re-run safety stock calculation with lead_time = 7, 14, 21 days
- [ ] Plot: safety stock (days of supply) vs. lead time, faceted by ABC-XYZ
      cell
- [ ] Key finding: AX items are relatively insensitive to lead time vs. CZ
      items (because σ dominates)
- [ ] Document in writeup under "Limitations & Next Steps"

**Labels:** `safety-stock`, `visualization`  
**Portfolio Signal:** Statistical Rigor, Business Framing  
**Portfolio Artifact:** Yes — sensitivity chart appears in writeup  
**Estimate:** 3h

---

## Sprint 6 — Portfolio Packaging

**Epic:** SCDF-E5  
**Goal:** Project is live on portfolio. GitHub repo is polished. Writeup
follows STAR structure with all visuals in place.  
**Duration:** 1 week

### SCDF-27 · Story · Write Quarto project page (STAR structure)

**Description:**

The writeup is as important as the analysis. Structure it so a recruiter gets
the business value in 30 seconds and a hiring manager can drill into technical
depth. The Azure deployment is a visible part of the story — this project now
covers both the Senior DS signal (business decision, quantified outcome) and
the MLE signal (cloud-native, deployed endpoint).

**Acceptance Criteria:**

- [ ] TL;DR callout block at top: 3 bullets — problem, what was built,
      quantified result
- [ ] Sections follow STAR: Situation (business context + EDA findings) → Task
      (what decision needed to be made) → Action (ABC-XYZ, forecasting, safety
      stock — with tabsets separating business narrative from code) → Result
      (policy table + cost curve + MAPE improvement vs. baseline)
- [ ] Architecture diagram (Mermaid) showing full system: M5 data → Azure Blob
      Storage → Feature Pipeline → Azure ML Training → Model Registry → ACR →
      ACI Endpoint
- [ ] Azure ML experiment screenshot embedded (from SCDF-11)
- [ ] Live ACI endpoint URL linked — "Try it" demo button
- [ ] All charts embedded with captions; code in `code-fold: true` blocks
- [ ] "Limitations & Next Steps" section — 3–5 honest limitations (e.g.,
      synthetic cost assumptions, static lead time, no real-time retraining, no
      autoscaling)
- [ ] Links to GitHub repo at top and bottom
- [ ] Tags updated to include `Azure`, `MLOps`, `Cloud Deployment`

**Labels:** `writeup`, `visualization`  
**Portfolio Signal:** Business Framing, Cloud Infrastructure  
**Portfolio Artifact:** Yes — this is the primary portfolio artifact  
**Estimate:** 10h

#### Task · Header Banner

Design a wide-format header SVG for the Quarto project page (matching the
`notebook_vs_production_header.svg` format from the sentiment project).
Concept: a horizontal supply chain flow — warehouse → shelves → shipping
container → store — rendered as minimal flat icons in Catppuccin Base/Surface
colors with a Mauve or Sapphire accent line connecting them.

#### Task - Architecture Diagram

Design the full Azure pipeline architecture diagram in Inkscape: M5 Blob →
Feature Pipeline → Azure ML Training → Model Registry → ACR → ACI Endpoint. Use
Catppuccin Blue for Azure services, Teal for data flow arrows, Peach for the
model artifact, and Surface0/Surface1 for component backgrounds. Export as SVG
for inline embedding in Quarto.

#### Task - Methodology Section Icons

Design a set of 3 small inline icons in Inkscape for the methodology section of
the writeup — one each for Segmentation (a 3×3 grid), Forecasting (a time
series line with a dashed future segment), and Safety Stock (a buffer zone bar
chart). Catppuccin accent colors, consistent stroke weight, ~48×48px, exported
as SVGs for inline use in Quarto.

### SCDF-28 · Story · Create project thumbnail

**Description:**

Thumbnails are what eyes land on in the projects listing. It needs to
communicate "supply chain + data science" in a single image.

**Acceptance Criteria:**

- [ ] SVG or PNG thumbnail, consistent with portfolio visual style
- [ ] Incorporates the ABC-XYZ matrix heatmap or the trade-off curve (the most
      visually distinctive output)
- [ ] Dimensions match other project thumbnails on the site
- [ ] Added to `projects/sc-demand-forecasting/` folder in portfolio repo

**Labels:** `writeup`, `visualization`  
**Portfolio Signal:** (meta — portfolio UX)  
**Portfolio Artifact:** Yes  
**Estimate:** 2h

#### SCDF-39 - Task · Thumbnail & Listing Card

Design project thumbnail in Inkscape — incorporate the ABC-XYZ 3×3 grid motif
with Catppuccin Mauve/Lavender/Peach as the cell fill gradient (high-value AX
in Mauve, low-value CZ in Surface1). Should read clearly at ~300×200px. Export
as SVG.

### SCDF-29 · Story · Polish GitHub repo for public viewing

**Description:**

Hiring managers look at commit history before reading the README. The repo
needs to look like a real project, not a homework dump.

**Acceptance Criteria:**

- [ ] README updated with final quantified results (fill in the placeholders
      from SCDF-4)
- [ ] Commit history is clean: conventional commits, Jira ticket references
      (e.g., `SCDF-9: add lag feature pipeline`)
- [ ] `notebooks/` directory has numbered, named notebooks: `01-eda.ipynb`,
      `02-segmentation.ipynb`, `03-features.ipynb`, `04-model.ipynb`,
      `05-evaluation.ipynb`, `06-safety-stock.ipynb`
- [ ] All notebooks have markdown narrative cells — not just code
- [ ] `tests/` directory has passing tests (`pytest` runs clean)
- [ ] No data files committed; `data/README.md` has Kaggle download
      instructions
- [ ] GitHub repo description and topics set: `supply-chain`,
      `demand-forecasting`, `lightgbm`, `inventory-optimization`,
      `data-science`

**Labels:** `documentation`, `infra`  
**Portfolio Signal:** MLOps  
**Portfolio Artifact:** No  
**Estimate:** 4h

### SCDF-30 · Story · Add project to portfolio site and deploy

**Description:**

Get the project live on matthewburns.net/projects/ with correct metadata,
thumbnail, and categories. This project now carries both DS and MLE signals —
make sure the tags reflect both.

**Acceptance Criteria:**

- [ ] Project appears in the Quarto listing with thumbnail, description, and
      tags
- [ ] Tags include: `Supply Chain`, `Forecasting`, `Inventory Optimization`,
      `LightGBM`, `Azure`, `MLOps`, `Domain Knowledge`
- [ ] Open Graph metadata set on the project page (title, description, image)
- [ ] Netlify deploy preview reviewed before merging to main
- [ ] LinkedIn post drafted (3-bullet summary + link) for launch — mention the
      live Azure endpoint explicitly

**Labels:** `writeup`  
**Portfolio Signal:** (meta)  
**Portfolio Artifact:** Yes  
**Estimate:** 2h

#### SCDF-40 - Task - Social Card / OG Image

Design a 1200×630px Open Graph image in Inkscape for LinkedIn/Slack link
previews. Layout: project title left-aligned, a simplified ABC-XYZ grid motif
right-aligned, your name and matthewburns.net in the footer. Catppuccin Mocha
base with Mauve/Lavender text. Export as PNG (OG images must be raster for
reliable social card rendering).

---

## Sprint 7 — Azure Deployment

**Epic:** SCDF-E6  
**Goal:** A live FastAPI inference endpoint running in Azure Container
Instances, built from a Docker image stored in Azure Container Registry, with a
GitHub Actions CI pipeline that rebuilds and pushes on every merge to main.  
**Duration:** 1 week

### SCDF-31 · Story · Build FastAPI inference service

**Description:**

Wrap the trained LightGBM model in a FastAPI app that accepts a SKU-store-date
payload and returns a point forecast plus the ABC-XYZ segment and recommended
safety stock. This is the interface between the ML model and anything that
would consume it in a real supply chain system.

**Acceptance Criteria:**

- [ ] `docker/app/main.py` implements a FastAPI app with:
  - [ ] `GET /health` — returns `{"status": "ok"}` and model version
  - [ ] `POST /predict` — accepts
        `{"item_id": str, "store_id": str, "date": str}`, returns
        `{"forecast": float, "abc_class": str, "xyz_class": str, "safety_stock_units": float, "service_level": float}`
- [ ] Model loaded from `models/lgbm_v1.pkl` at startup (or pulled from Azure
      ML model registry)
- [ ] Input validation via Pydantic models — invalid inputs return 422 with a
      clear error message
- [ ] `docker/app/requirements.txt` is minimal: only inference dependencies,
      not training ones
- [ ] Local test: `uvicorn main:app --reload` runs and returns correct
      predictions
- [ ] Unit tests for the predict endpoint using `TestClient`

**Labels:** `deployment`, `testing`  
**Portfolio Signal:** MLOps, Cloud Infrastructure  
**Portfolio Artifact:** No  
**Estimate:** 6h

### SCDF-32 · Story · Dockerize the inference service

**Description:**

Package the FastAPI app into a Docker image. The Dockerfile itself is a
portfolio artifact — anyone reviewing the repo will open it to gauge ML
engineering maturity.

**Acceptance Criteria:**

- [ ] `docker/Dockerfile` uses a slim Python base image (`python:3.11-slim`)
- [ ] Multi-stage build: builder stage installs dependencies; runtime stage
      copies only what's needed — minimizes image size
- [ ] Model file copied into image at build time (or downloaded from Azure ML
      registry at startup — document the trade-off in a comment)
- [ ] Image builds cleanly with `docker build -t scdf-inference .` from repo
      root
- [ ] Container runs locally with `docker run -p 8000:8000 scdf-inference` and
      health check passes
- [ ] `.dockerignore` excludes: `data/`, `notebooks/`, `tests/`, `docs/`,
      `*.ipynb`, `*.pyc`
- [ ] Image size documented in `docs/deployment.md` (target: under 1GB)

**Labels:** `deployment`, `docker`, `azure`  
**Portfolio Signal:** MLOps, Cloud Infrastructure  
**Portfolio Artifact:** No  
**Estimate:** 4h

### SCDF-33 · Story · Push image to ACR and deploy to ACI

**Description:**

Push the Docker image to Azure Container Registry and deploy it as an Azure
Container Instance with a public endpoint. This is the "live demo" link that
goes on the portfolio page and in the README.

**Acceptance Criteria:**

- [ ] Image tagged and pushed to `acrscdf.azurecr.io/scdf-inference:v1`
- [ ] ACI instance deployed: `az container create` with at least 1 vCPU, 1.5GB
      RAM, port 8000 exposed
- [ ] Public FQDN assigned and verified: `curl https://<fqdn>/health` returns
      200
- [ ] `docs/deployment.md` documents the full deployment commands, resource
      specs, and estimated monthly cost at current usage (signals
      cost-awareness)
- [ ] ACI container logs checked: no startup errors, model loaded successfully
- [ ] Live endpoint URL added to README and noted for SCDF-20 (writeup)

**Labels:** `deployment`, `azure`  
**Portfolio Signal:** Cloud Infrastructure, MLOps  
**Portfolio Artifact:** Yes — live endpoint URL linked in writeup  
**Estimate:** 5h

### SCDF-34 · Story · GitHub Actions CI/CD pipeline

**Description:**

A CI pipeline that runs tests on every PR and rebuilds + pushes the Docker
image to ACR on every merge to main. This is the artifact that most clearly
separates an MLE portfolio from a DS portfolio.

**Acceptance Criteria:**

- [ ] `.github/workflows/ci.yml`: triggers on PR to main
  - [ ] Installs dependencies
  - [ ] Runs `pytest tests/` — fails the PR if tests fail
  - [ ] Linting with ruff
- [ ] `.github/workflows/deploy.yml`: triggers on push to main
  - [ ] Logs in to ACR using GitHub Actions secret (`AZURE_CREDENTIALS`)
  - [ ] Builds and pushes Docker image tagged with git SHA and `latest`
  - [ ] (Optional stretch) re-deploys ACI with new image tag
- [ ] GitHub Actions secrets documented (names only, not values) in
      `docs/ci-cd.md`
- [ ] Both workflows show green on the repo — badge added to README
- [ ] `docs/ci-cd.md` includes a diagram of the pipeline flow: PR → tests →
      merge → build → push → deploy

**Labels:** `deployment`, `azure`, `infra`, `testing`  
**Portfolio Signal:** MLOps, Cloud Infrastructure  
**Portfolio Artifact:** Yes — CI badge in README; pipeline diagram in writeup  
**Estimate:** 5h

---

## Summary: Stories by Sprint

| Sprint    | Stories                                 | Est. Hours | Key Deliverable                                                      |
| --------- | --------------------------------------- | ---------- | -------------------------------------------------------------------- |
| Sprint 1  | SCDF-1, SCDF-1a, SCDF-2, SCDF-3, SCDF-4 | ~19h       | Repo + Azure infra + Blob data layer + EDA + README                  |
| Sprint 2  | SCDF-5 to SCDF-8                        | ~15h       | ABC-XYZ matrix + MTS/MTO decision table                              |
| Sprint 3  | SCDF-9 to SCDF-12                       | ~20h       | Feature pipeline (→ Blob) + LightGBM + Azure ML tracking + baselines |
| Sprint 4  | SCDF-13 to SCDF-15                      | ~14h       | Segment-level evaluation + error stats + SHAP                        |
| Sprint 5  | SCDF-16 to SCDF-19                      | ~16h       | Safety stock policy + trade-off curve + sensitivity                  |
| Sprint 6  | SCDF-20 to SCDF-23                      | ~18h       | Quarto writeup + polished repo + live on site                        |
| Sprint 7  | SCDF-24 to SCDF-27                      | ~20h       | Docker image → ACR → ACI live endpoint + CI pipeline                 |
| **Total** | **27 stories**                          | **~122h**  |                                                                      |

---

## Portfolio Artifact Tracker

Stories that produce something visible in the portfolio writeup:

| Story   | Artifact                          | Visual Type     |
| ------- | --------------------------------- | --------------- |
| SCDF-3  | EDA charts (2–3)                  | Charts          |
| SCDF-7  | ABC-XYZ heatmap                   | Hero Visual #1  |
| SCDF-8  | MTS/MTO decision table            | Table           |
| SCDF-11 | Azure ML experiment screenshot    | Screenshot      |
| SCDF-13 | Segment performance table         | Table           |
| SCDF-14 | Error distribution plots          | Charts          |
| SCDF-15 | Feature importance (SHAP)         | Chart           |
| SCDF-17 | Holding vs. stockout trade-off    | Hero Visual #2  |
| SCDF-18 | Policy recommendations table      | Table           |
| SCDF-19 | Sensitivity analysis chart        | Chart           |
| SCDF-20 | Quarto writeup (primary artifact) | Full page       |
| SCDF-21 | Thumbnail                         | Image           |
| SCDF-23 | Live portfolio page               | Deployed page   |
| SCDF-26 | Live ACI endpoint (demo link)     | URL             |
| SCDF-27 | CI badge + pipeline diagram       | Badge / Diagram |
