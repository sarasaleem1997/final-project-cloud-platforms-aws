# VaultTech Project — Detailed Notes & Study Guide

**ESADE Cloud Platforms AWS — Final Project**
**Deadline:** April 14, 2026
**Student:** Sara Saleem

---

## Project Overview

**What is VaultTech?**
A real industrial case study: analyzing per-piece cycle times on a steel forging line that produces track chain links for heavy earthmoving equipment (bulldozers).

**The process:** Each piece travels ~58 seconds through 4 zones:
> Furnace → Main Press (4 strikes) → Auxiliary Press → Quench Bath

**The goal:** An XGBoost model predicts total bath time after only the 2nd strike (~18 seconds into the process), enabling real-time delay alerts while the piece is still on the line.

**The dataset:** ~180,000 raw PLC signal readings across 4 die matrices, spanning November 2025 – March 2026.

---

## Key Conceptual Pillars (Exam Focus)

### 1. Medallion Architecture (Bronze → Silver → Gold)
Each layer has a specific role:
- **Bronze** (PostgreSQL `bronze` schema): Raw data exactly as the sensors recorded it. Preserves original truth. Never modified.
- **Silver** (PostgreSQL `silver` schema): Cleaned, validated data. One row per piece with trustworthy cumulative times.
- **Gold** (`data/gold/pieces.parquet`): Enriched with partial times and production context. Ready for analytics and ML.

**Why this matters:** If your cleaning logic turns out to be wrong later, you can always go back to bronze and redo it. Each layer serves a different consumer — bronze for auditing, silver for analysis, gold for ML.

### 2. Why Analysis Must Be Per Die Matrix
There are 4 die matrices (moulds), each with different tooling geometry and expected travel times:
- 4974: median bath = 56.0s
- 5052: median bath = 58.6s
- 5090: median bath = 58.3s
- 5091: median bath = 59.3s

A piece taking 58s is slow for matrix 4974 but normal for matrix 5091. **You must always segment analysis by die matrix when groups have different baseline behaviors.**

### 3. Cumulative vs Partial Times
- **Cumulative times**: time elapsed from furnace exit to each stage (what the sensors record)
- **Partial times**: time spent in each segment = subtract consecutive cumulative values

Partial times are the diagnostic tool. When a piece is slow, the partial that deviates from reference tells you which segment caused the delay.

### 4. OEE Cycle Time vs Piece Travel Time
These are two completely different things:
- **Travel time** (~58s): how long ONE piece takes from furnace to bath
- **OEE cycle time** (~14s): how frequently a NEW piece is launched into the line

Multiple pieces are on the line simultaneously — it's a pipeline.

### 5. Early Prediction Value
The model predicts bath time after the 2nd strike (~18s in). This is valuable because:
- The piece is still on the line (40 seconds left in its journey)
- An alert can trigger real-time intervention
- Using later stage data would make prediction useless (piece almost done)

---

## Task 1: Set Up the Environment

### Steps Completed

**Step 1: Initialize git repository**
```bash
git init
git add -A
git commit -m "initial: student starter template"
```
**Why:** Git tracks all changes. The instructor evaluates work via git tags — each task needs a tag so they can check out the exact state of the code at that point.

**Step 2: Install dependencies**
```bash
uv sync
```
**Why:** `uv` is a Python package manager. `uv sync` reads `pyproject.toml` and installs all dependencies (pandas, XGBoost, Streamlit, etc.) into a local `.venv` virtual environment. Virtual environments isolate this project's packages from the rest of the machine.

**Step 3: Start PostgreSQL + run Flyway migrations**
```bash
docker compose up -d postgres flyway
```
**Why:**
- **PostgreSQL** runs in Docker — portable, no local install needed. `-d` = detached (background).
- **Flyway** applies versioned SQL migrations (V001–V008) to create the database schema. Versioned migrations mean schema changes are tracked like code — reproducible and auditable.
- Note: we start only `postgres` and `flyway` services, NOT `app` — the Dockerfile is a Task 9 placeholder.

**Fix applied:** Added `PGDATA_PATH=/home/sara_saleem/vaultech-pgdata` to `infra/.env` because PostgreSQL needs Linux file permissions on its data directory, which the OneDrive path doesn't support.

**Step 4: Seed the database**
```bash
uv run python scripts/seed.py --env infra/.env
```
**Why:** Loads raw factory sensor data from compressed CSV files into the bronze tables:
- `bronze.raw_lifetime`: 1,233,541 rows
- `bronze.raw_piece_info`: 359,534 rows

This is the **bronze layer** — raw data exactly as captured. No cleaning, no transformation.

**Step 5: Verify in JupyterLab**
```bash
uv run lab  # run in a separate WSL terminal that stays open
```
Ran a COUNT(*) query on both bronze tables to confirm data landed correctly.

**Git tag:** `task-01`

---

## Task 2: Understand the Case and Review the Data

**Deliverable:** `notebooks/00_explore_data.ipynb`

### Database Connection
```python
import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine(
    "postgresql+psycopg2://vaultech:vaultech_dev@localhost:5432/vaultech"
)
```

### The Raw Table Structure (Tall Format)
`bronze.raw_lifetime` has 3 columns:
- `timestamp`: when the PLC recorded the reading
- `signal`: which sensor/stage this is from
- `value`: cumulative travel time in seconds for that piece at that stage

**This is "tall" (long) format** — one row per signal reading, not one row per piece. A single piece appears multiple times — once per stage. Task 3 will pivot this into "wide" format (one row per piece, each stage as a column).

### The 7 Signals
| Signal | Stage | Typical value | Records |
|--------|-------|---------------|---------|
| lifetime_first | 2nd strike | ~17–19s | 179,628 |
| lifetime_second | 3rd strike | ~24–26s | 179,628 |
| lifetime_drill | 4th strike | ~37–40s | **150,434** ⚠️ |
| upsetting_lifetime | 1st strike | broken | 179,628 |
| lifetime_auxiliary_press | Auxiliary press | ~54–57s | 184,966 |
| lifetime_bath | Quench bath | ~56–59s | 179,628 |
| lifetime_general | General (≈ bath) | ~56–59s | 179,629 |

**Key findings:**
- **Drill signal (4th strike)** has ~29,000 fewer records than others → hardware reliability issue with that sensor
- **Upsetting signal (1st strike)** has 179,628 records but nearly all values are ~0.10s → broken at PLC level, discard entirely
- **Auxiliary press** has slightly more records (184,966) → occasional double-readings

### Data Quality Issues Found

**1. Zero values (~1.2% of all readings)**
Min value = 0.00 for every signal. A piece can't take 0 seconds to travel — these are **tracking failures** where the PLC lost track of a piece. Remove in Task 3.

**2. Extreme outliers (max values 553–736s vs expected ~58s)**
Pieces taking 10+ minutes are not genuinely slow — they're stuck pieces, unclosed cycles, or machine stops. Remove using Q3 + 3×IQR per signal per die matrix in Task 3.

**3. Broken upsetting signal (22.8% zeros, rest ~0.10)**
p25 = p50 = p75 = 0.10 → at least 75% of values are 0.10 or less. Not measuring travel time. Discard entirely in Task 3.

**4. Duplicates**
Gap analysis shows min gap = 0.00s → same signal fires twice at same timestamp. PLC occasionally registers the same piece twice. Deduplicate in Task 3.

### Sampling Frequency Analysis
- **Median gap between readings: ~13.9s** = OEE cycle time (one new piece every 14s)
- **Min gap: 0.00s** = duplicates
- **Max gap: 1,272,995s ≈ 353 hours** = production shutdowns (weekends, maintenance)

### Per Die Matrix Statistics
| Matrix | Median bath | Active period |
|--------|-------------|---------------|
| 4974 | 56.0s | Nov 13 – Nov 25, 2025 (~12 days) |
| 5052 | 58.6s | Nov 6, 2025 – Feb 25, 2026 |
| 5090 | 58.3s | Dec 4, 2025 – Feb 17, 2026 |
| 5091 | 59.3s | Nov 25, 2025 – Mar 11, 2026 |

**Multiple matrices can be active on the same day** (5052, 5091, 5090 overlapped). This is another reason analysis must be per matrix.

### Partial Times (Median, matrix 5052)
| Segment | Time |
|---------|------|
| Furnace → 2nd strike | 18.4s |
| 2nd → 3rd strike | 7.0s |
| 3rd → 4th strike | 13.8s |
| 4th strike → Aux press | 17.3s |
| Aux press → Bath | 1.6s |

The **4th strike → Aux press** segment is the longest partial — the piece exits the main press, gets deburred and coined, and transfers to the auxiliary press. Most delays are detectable in the earlier partials.

### Production Patterns
- 85 production days across ~4 months (out of ~126 calendar days)
- ~2,100–2,200 pieces per active production day
- **5052 shows declining production volume over time** → classic sign of tooling wear (die geometry degrades with accumulated strikes)

**Git tag:** `task-02`

---

## Task 3: Clean Raw Data (Bronze → Silver)

**Deliverable:** `notebooks/01_bronze_to_silver.ipynb`

**What this notebook does:** Reads raw bronze data, applies 8 cleaning steps, writes 169,161 validated pieces to `silver.clean_pieces`.

### The 8 Cleaning Steps

**Step 1: Incremental boundary**
Check the latest timestamp in silver. Only process bronze rows newer than that. This makes the notebook safe to re-run — it won't duplicate data.

**Step 2: Extract and filter raw signals**
```python
WHERE signal != upsetting_signal AND value > 0
```
- Drops upsetting signal entirely (broken at PLC level, values 0–6.7s)
- Removes zero values (tracking failures where PLC lost track of piece)
- Result: 1,041,278 rows (from 1,233,541)

**Step 3: Deduplicate timestamps**
```python
df.drop_duplicates(subset=["timestamp", "signal"], keep="last")
```
PLC occasionally fires the same signal twice at the same timestamp. Keep last reading. Removed: 6 duplicates.

**Step 4: Pivot and join**
- **Pivot**: tall format (one row per signal) → wide format (one row per piece, each signal as column)
- **Join**: merge lifetime data with piece identification (piece_id, die_matrix) on timestamp
- Result: 178,308 rows (6 signals collapsed into 1 row per piece)

**Step 5: Drop missing identification**
Remove pieces with no piece_id or die_matrix — can't analyze without knowing which matrix. Removed: 0 rows.

**Step 6: Remove outliers (Q3 + 3×IQR per signal per die matrix)**
```python
upper = q3 + 3 * iqr  # computed per signal per matrix
```
Removes stuck pieces, unclosed cycles, machine stops. Must be per matrix because each has different normal ranges. Removed: 9,147 pieces.

**Step 7: Validate monotonic order**
```python
2nd strike < 3rd strike < 4th strike < aux press < bath
```
Physically impossible if violated. Removed: 0 violations (outlier removal already cleaned these).

**Step 8: Compute OEE cycle time**
```python
df["gap"] = df["timestamp"].diff().dt.total_seconds()
df["oee"] = df["gap"].rolling(5).mean()
# Set values outside 11–16s to NULL
```
Rolling average of last 5 inter-piece intervals. Values outside 11–16s → NULL (piece is valid, OEE metric is not).

**Step 9: Write to silver**
```python
df.to_sql("clean_pieces", schema="silver", if_exists="append")
```

### Cleaning Report Results
| Step | Removed | Reason |
|------|---------|--------|
| Upsetting signal | 179,628 rows | Broken at PLC level |
| Zero values | 12,635 rows | Tracking failures |
| Duplicates | 6 rows | PLC double-reads |
| Pivot/join | format change | Tall → wide (not data loss) |
| Missing ID | 0 | All had piece_id + die_matrix |
| Outliers (3×IQR) | 9,147 pieces | Stuck pieces, machine stops |
| Monotonicity | 0 | Already cleaned by outlier step |
| **Silver result** | **169,161 pieces** | **94% retention rate** |

OEE valid: 131,066 (77.5%) | OEE null: 38,095 (gap starts, run boundaries)

**Git tag:** `task-03`

---

## Task 4: Enrich and Export to Gold

**Deliverables:** `notebooks/02_silver_to_gold.ipynb`, `notebooks/03_build_clean_dataset.ipynb`

**What this does:** Reads clean silver data, adds computed features, exports to parquet.

### Step 1: Compute Partial Times
```python
df["partial_furnace_to_2nd_strike_s"] = df["lifetime_2nd_strike_s"]
df["partial_2nd_to_3rd_strike_s"] = df["lifetime_3rd_strike_s"] - df["lifetime_2nd_strike_s"]
# etc.
```
Subtracts consecutive cumulative times to get time spent in each segment.

| Segment | Median |
|---------|--------|
| Furnace → 2nd strike | 18.0s |
| 2nd → 3rd strike | 6.8s |
| 3rd → 4th strike | 13.7s |
| 4th strike → Aux press | 17.5s |
| Aux press → Bath | 1.6s |

### Step 2: Mark Production Gaps + Assign Run IDs
```python
GAP_THRESHOLD = 5 * 60  # 5 minutes
df["is_gap_start"] = df["timestamp"].diff().dt.total_seconds() > GAP_THRESHOLD
df["production_run_id"] = df["is_gap_start"].cumsum()
```
939 production runs identified. Prevents time-series analysis from interpolating across weekends/maintenance stops.

### Step 3: Export to Parquet
```python
df_gold.to_parquet("data/gold/pieces.parquet", index=False)
```
169,161 rows, 17 columns, 4.6 MB. Parquet is columnar — much faster for analytics than CSV.

### Quality Gate (notebook 03)
Verified: 0 zeros, 0 extreme outliers, 0 monotonicity violations, 0 invalid OEE values.

Notable nulls in gold:
- `lifetime_4th_strike_s`: 28,182 nulls (drill sensor offline period)
- `oee_cycle_time_s`: 38,095 nulls (gap starts)

**Git tag:** `task-04`

---

## Task 5: Analyze Per Die Matrix

**Deliverable:** `notebooks/04_analyze_per_matrix.ipynb`

### Reference Profiles (median cumulative times)
| Matrix | 2nd strike | 3rd strike | 4th strike | Aux press | Bath |
|--------|-----------|-----------|-----------|-----------|------|
| 4974 | 17.3s | 23.9s | 37.1s | 54.2s | 56.0s |
| 5052 | 18.3s | 25.3s | 39.3s | 56.7s | 58.3s |
| 5090 | 17.7s | 24.6s | 38.5s | 56.5s | 58.1s |
| 5091 | 18.5s | 25.6s | 38.2s | 57.5s | 59.1s |

### Segment Variability (CV = std/median × 100%)
| Segment | CV |
|---------|-----|
| Furnace → 2nd strike | 12.9% (most unstable) |
| 2nd → 3rd strike | 10.2% |
| 4th strike → Aux press | 8.1% |
| 3rd → 4th strike | 7.8% |
| Aux press → Bath | 5.6% (most stable) |

**Furnace → 2nd strike is the most variable** — robot pick and transfer from furnace is the least controlled step.

### Slow Pieces (> 90th percentile per matrix)
- Total slow: 16,549 (9.8% of all pieces)
- Most penalized segment: **furnace_to_2nd_strike** (78% of slow pieces)
- 90th percentile thresholds: 4974=59.0s, 5052=61.8s, 5090=62.9s, 5091=63.4s

### Drift Detection
Matrix 5052 shows progressive bath time increase over its active period (Nov 2025 – Feb 2026) — consistent with tooling wear.

**Git tag:** `task-05`

---

## Task 6: Feature Selection and Predictive Model

**Deliverable:** `notebooks/05_feature_selection_and_model.ipynb`

### The Prediction Problem
Predict `lifetime_bath_s` using only data available after the 2nd strike (~18s into the 58s process). Early enough to raise real-time alerts while piece is still on the line.

### Feature Selection
| Feature | Included? | Why |
|---------|-----------|-----|
| `die_matrix` | ✅ | Each matrix has different baseline times (4974=56s vs 5091=59s) |
| `lifetime_2nd_strike_s` | ✅ | Earliest timing signal, r=0.77 with target |
| `oee_cycle_time_s` | ✅ | Production rhythm context, nulls filled with median=13.81s |
| `lifetime_3rd_strike_s` | ❌ | Available too late (~25s) |
| `lifetime_4th_strike_s` | ❌ | Too late (~38s) + 16% missing |
| `lifetime_auxiliary_press_s` | ❌ | Too late (~55s), only 2s before bath |
| `lifetime_general_s` | ❌ | Equivalent to target — data leakage |

### Model: XGBoost Regressor
```python
XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)
```
80/20 train/test split stratified by die_matrix. Training: 134,188 pieces. Test: 33,548 pieces.

### Results
| Metric | Value |
|--------|-------|
| RMSE | 1.838s |
| MAE | 0.919s (1.6% of typical bath time) |
| R² | 0.687 |

### Feature Importance
| Feature | Importance |
|---------|-----------|
| lifetime_2nd_strike_s | 68.7% |
| oee_cycle_time_s | 24.4% |
| die_matrix | 6.9% |

**Artifacts saved:** `models/xgboost_bath_predictor.json`, `models/model_metadata.json`

**Git tag:** `task-06`

---

## Task 7: Build Inference Service

**Deliverable:** `src/vaultech_analysis/inference.py`

### The Predictor Class
Wraps the saved XGBoost model for reuse by the Streamlit app, CLI, and tests.

```python
predictor = Predictor()
result = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.3, oee_cycle_time_s=13.5)
# Returns: {"predicted_bath_time_s": 57.196, "model_metrics": {...}}
```

**Key design decisions:**
- Validates die_matrix — returns `{"error": "..."}` for unknown values instead of crashing
- Missing OEE defaults to median (13.81s) — handles 23% of pieces with NULL OEE
- Returns model metrics alongside prediction — app can show confidence
- `predict_batch(df)` — runs predictions on all 169k pieces at app startup

**CLI usage:**
```bash
uv run python -m vaultech_analysis.inference --die-matrix 5052 --strike2 18.3 --oee 13.5
```

**All 7 tests passing** in `tests/test_inference.py`

**Git tag:** `task-07`

---

## Task 8: Integrate into Streamlit App

**Deliverable:** `app/streamlit_app.py`

### Key Streamlit Concepts
- **`@st.cache_resource`**: caches the Predictor object — model loaded once, shared across all users
- **`@st.cache_data`**: caches the gold dataset + predictions — computed once per session, not on every interaction
- Streamlit reruns the entire script on every user interaction — caching is essential for performance

### App Features
1. **Sidebar filters**: die matrix, date range, slow pieces only (> 90th pct per matrix)
2. **Summary metrics**: total pieces, median bath time, median predicted, MAE
3. **Pieces table** with row selection (`on_select="rerun"`)
4. **Piece detail panel** (appears on row click):
   - Cumulative times vs reference with deviation
   - Partial times vs reference with 🟢 OK / 🔴 Slow status (slow = >2s above reference)
   - Bar chart: actual vs reference partial times

**Launch:** `uv run app` → `http://localhost:8501`

**Git tag:** `task-08`

---

## Task 9: Package App Locally (Docker)

**Deliverable:** `Dockerfile`

### The Dockerfile
```dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app/ src/ models/ data/gold/ ./
EXPOSE 8501
CMD [".venv/bin/streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

**Why each decision:**
- `python:3.13-slim` — minimal base image, no unnecessary tools
- Dependencies copied before app code — Docker layer caching means rebuilds are fast when only code changes
- `--no-install-project` — installs only dependencies, not the package itself (app adds `src/` to path at runtime)
- Only copies `app/`, `src/`, `models/`, `data/gold/` — no notebooks, tests, docs, or raw data
- `--server.address=0.0.0.0` — makes app accessible outside container

**Build and test:**
```bash
docker build -t vaultech-app .
docker run -d -p 8502:8501 vaultech-app
# Open http://localhost:8502
```

**Fix needed:** Docker mirror required on home network (Cloudflare blocked):
Docker Desktop → Settings → Docker Engine → add `"registry-mirrors": ["https://mirror.gcr.io"]`

**Git tag:** `task-09`

---

## Task 10: Deploy Model as SageMaker Endpoint

**Deliverables:** `deploy/deploy_sagemaker.py`, `deploy/README.md`, 7 tests passing

**Git tag:** `task-10`

### What this task does

Takes the trained XGBoost model out of the local `models/` folder and deploys it as a managed cloud service that can accept HTTP requests and return predictions from anywhere.

### The 5-Step Pipeline

The script (`uv run python deploy/deploy_sagemaker.py --bucket ... --endpoint-name ... --model-package-group ...`) runs 5 steps:

**Step 1: Package model**
```python
def package_model(model_path, output_dir):
    # Patch base_score JSON format, then:
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.addfile(info, io.BytesIO(patched_bytes))  # named "xgboost-model"
```
SageMaker's built-in XGBoost container expects a file named exactly `xgboost-model` at the root of a `.tar.gz` archive. We rename during archiving.

**Why the base_score patch?** XGBoost 3.2 stores `base_score` as an array `'[58.6]'` in the JSON. The SageMaker container (XGBoost 3.0) expects a scalar `'58.6'`. Without the patch, predictions came out ~58 seconds too low (near 0 instead of 57s). We fix it in-flight when creating the tar.gz.

**Step 2: Upload to S3**
SageMaker can't read local files — the model artifact must live in S3 so the container can pull it at startup.
- Bucket: `vaultech-models-999390550986`
- Key: `models/xgboost-bath-predictor/model.tar.gz`

**Step 3: Register in Model Registry**
```python
sm.create_model_package(
    ModelPackageGroupName="vaultech-bath-predictor-group",
    InferenceSpecification={"Containers": [{"Image": image_uri, "ModelDataUrl": s3_uri}], ...},
    CustomerMetadataProperties={"rmse": "1.838", "mae": "0.919", "r2": "0.687"},
    ModelApprovalStatus="Approved",
)
```
The **Model Registry** is a versioned catalog of all deployed models. Each registration creates a new version with the metrics attached. This gives you:
- Audit trail: who deployed what, when, with what performance
- Rollback: can redeploy any previous version
- Governance: `ModelApprovalStatus="Approved"` signals it's production-ready

Container image retrieved with: `sagemaker.image_uris.retrieve("xgboost", region="eu-west-1", version="3.0-5")`

**Step 4: Deploy endpoint**
Three separate SageMaker objects must be created in order:
1. **Model** — links the container image to the S3 artifact + IAM role
2. **EndpointConfig** — specifies instance type (`ml.t2.medium`) and how many instances
3. **Endpoint** — the live HTTPS URL that accepts prediction requests

Then wait up to 15 minutes for the endpoint to reach `InService` status (container is pulled, model is loaded).

**Step 5: Test endpoint**
```python
runtime.invoke_endpoint(
    EndpointName="vaultech-bath-predictor",
    ContentType="text/csv",
    Body="5052,18.3,13.5",  # die_matrix, lifetime_2nd_strike_s, oee_cycle_time_s
)
# Returns: "57.196"
```
5 test cases checked: all predictions in 40–80s range, slow pieces predict higher.

### AWS Resources Created
| Resource | Name/Value |
|----------|-----------|
| S3 bucket | vaultech-models-999390550986 |
| Model Package Group | vaultech-bath-predictor-group |
| Endpoint | vaultech-bath-predictor |
| Instance type | ml.t2.medium (eu-west-1) |
| IAM role | VaultTechSageMakerExecutionRole |

### IAM Role Needed
SageMaker needs its own **execution role** — an IAM role with:
- Trust policy: `"Service": "sagemaker.amazonaws.com"` → allows SageMaker to assume this role
- Policies: `AmazonSageMakerFullAccess` + `AmazonS3FullAccess` → can read model from S3 and manage endpoint

### Test Results
All 7 pytest tests passed (`uv run pytest tests/test_sagemaker.py -v`):
- Model Package Group exists ✅
- At least one version registered ✅
- Latest version has metrics (MAE, RMSE, R²) ✅
- Endpoint is InService ✅
- Sample prediction in 40–80s range ✅
- Different matrices produce different predictions ✅
- Slow 2nd strike predicts higher bath time ✅

---

## Task 11: Wire App to SageMaker + Deploy to ECS/Fargate

**Deliverables:** updated `inference.py`, `streamlit_app.py`, `Dockerfile`

**Git tag:** `task-11`

**Public URL:** http://3.253.10.36:8501

### What this task does

Connects the Streamlit app to the live SageMaker endpoint (instead of loading the model locally), containerizes the updated app, and deploys it to AWS so it's publicly accessible from a browser.

### Step 1: SageMakerPredictor class

Added to `src/vaultech_analysis/inference.py`:

```python
class SageMakerPredictor:
    def __init__(self, endpoint_name, region="eu-west-1"):
        self._runtime = boto3.client("sagemaker-runtime", region_name=region)
        self.endpoint_name = endpoint_name

    def predict(self, die_matrix, lifetime_2nd_strike_s, oee_cycle_time_s=None):
        payload = f"{die_matrix},{lifetime_2nd_strike_s},{oee_value}"
        t0 = time.perf_counter()
        response = self._runtime.invoke_endpoint(
            EndpointName=self.endpoint_name, ContentType="text/csv", Body=payload
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "predicted_bath_time_s": float(response["Body"].read()),
            "_debug": {"payload": payload, "latency_ms": latency_ms, ...}
        }

    def predict_batch(self, df):
        # Sends up to 5000 rows per request (chunked to stay under 5 MB)
        for chunk in chunks(df, 5000):
            csv_payload = chunk.to_csv(index=False, header=False)
            response = self._runtime.invoke_endpoint(...)
```

**Key design:** Same `predict()` and `predict_batch()` interface as the local `Predictor` — the Streamlit app doesn't need to know which backend is being used.

**Batch chunking:** Sending 169k rows one-by-one would call the endpoint 169,000 times (hours of latency + cost). Instead we batch 5,000 rows per CSV request. SageMaker's synchronous limit is 5 MB — 5,000 rows × ~20 chars ≈ 100 KB, well within limits.

### Step 2: Streamlit backend selection

```python
@st.cache_resource
def load_predictor():
    endpoint = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if endpoint:
        return SageMakerPredictor(endpoint_name=endpoint, region=os.environ.get("AWS_DEFAULT_REGION", "eu-west-1"))
    return Predictor()  # local fallback if no env var
```

When `SAGEMAKER_ENDPOINT_NAME` is set (injected by ECS at runtime), the app routes all predictions through the SageMaker endpoint. Locally (no env var), it falls back to the local XGBoost model.

### Step 3: Inference debug panel

When you click a piece in the table, the app now shows:
- **Endpoint name** — confirms which SageMaker endpoint is serving predictions
- **Predicted bath time** — the result from the endpoint
- **Latency** — round-trip time from app to SageMaker and back (typically 30–80ms)
- **Raw request** — the exact CSV payload sent: `"5052,18.3,13.5"`
- **Raw response** — the exact float string returned: `"57.196"`

This proves the prediction is coming from SageMaker, not local code — required for the demo video.

### Step 4: Dockerfile update

Removed `COPY models/ ./models/` — the model now lives in SageMaker, not in the container:
```dockerfile
# models/ excluded — inference is delegated to the SageMaker endpoint
COPY app/ ./app/
COPY src/ ./src/
COPY data/gold/ ./data/gold/   # gold parquet still needed (piece history)
```
The gold parquet stays because the app still needs it to show the historical pieces table and reference profiles.

### Step 5: Push to ECR

**ECR = Amazon Elastic Container Registry** — AWS's private Docker Hub. You push your image there so ECS can pull it securely without internet.

```bash
aws ecr create-repository --repository-name vaultech-app --region eu-west-1
aws ecr get-login-password | docker login --username AWS --password-stdin 999390550986.dkr.ecr.eu-west-1.amazonaws.com
docker build -t vaultech-app:task11 .
docker tag vaultech-app:task11 999390550986.dkr.ecr.eu-west-1.amazonaws.com/vaultech-app:latest
docker push 999390550986.dkr.ecr.eu-west-1.amazonaws.com/vaultech-app:latest
```

### Step 6: Deploy on ECS/Fargate

**ECS (Elastic Container Service)** manages the lifecycle of Docker containers in the cloud. **Fargate** is the serverless mode — no EC2 instances to manage. AWS handles the underlying server automatically.

Three objects needed:

**1. ECS Cluster** — logical grouping (like a namespace):
```bash
aws ecs create-cluster --cluster-name vaultech-cluster
```

**2. Task Definition** — blueprint for the container:
```json
{
  "family": "vaultech-app",
  "networkMode": "awsvpc",      ← each task gets its own network interface
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512", "memory": "1024",
  "executionRoleArn": "...ecsTaskExecutionRole",
  "containerDefinitions": [{
    "image": "999390550986.dkr.ecr.eu-west-1.amazonaws.com/vaultech-app:latest",
    "portMappings": [{"containerPort": 8501}],
    "environment": [
      {"name": "SAGEMAKER_ENDPOINT_NAME", "value": "vaultech-bath-predictor"},
      {"name": "AWS_DEFAULT_REGION", "value": "eu-west-1"}
    ]
  }]
}
```

**3. Service** — keeps the task running (restarts it if it crashes):
```bash
aws ecs create-service \
  --cluster vaultech-cluster \
  --service-name vaultech-app-service \
  --task-definition vaultech-app:1 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[...],securityGroups=[...],assignPublicIp=ENABLED}"
```

### IAM for Fargate (ecsTaskExecutionRole)

The `ecsTaskExecutionRole` has two responsibilities:
1. **Pull image from ECR** — `AmazonECSTaskExecutionRolePolicy` (standard, managed by AWS)
2. **Call SageMaker endpoint** — inline policy allowing `sagemaker:InvokeEndpoint`

Without #2, the Fargate container starts but every prediction call fails with an access denied error.

### Networking (VPC + Security Group)

Fargate tasks need a VPC with public subnets. Created the **default VPC** in eu-west-1 (172.31.0.0/16) with 3 public subnets (one per AZ).

Security group `vaultech-app-sg`: allows inbound TCP on port 8501 from anywhere (0.0.0.0/0). Everything else is blocked by default.

`assignPublicIp=ENABLED` → Fargate assigns a public IP to the task's network interface (ENI). This is the URL you browse to.

### AWS Resources Created
| Resource | Name/Value |
|----------|-----------|
| ECR repository | 999390550986.dkr.ecr.eu-west-1.amazonaws.com/vaultech-app |
| ECS cluster | vaultech-cluster |
| ECS service | vaultech-app-service |
| Task definition | vaultech-app:1 (Fargate, 0.5 vCPU, 1 GB RAM) |
| VPC | vpc-0694c7ccca45389ec (172.31.0.0/16) |
| Security group | sg-00a242a3f3dfcf79b (port 8501 open) |
| IAM role | ecsTaskExecutionRole |
| Public URL | http://3.253.10.36:8501 |

### End-to-End Data Flow (what happens when you select a piece)
1. Browser sends HTTP request to `3.253.10.36:8501`
2. Fargate container (Streamlit) receives it, looks up piece from gold parquet
3. App calls `SageMakerPredictor.predict()` → `boto3.invoke_endpoint()` → HTTPS to SageMaker endpoint in same region
4. SageMaker container (XGBoost 3.0-5) loads model, runs inference, returns float
5. App receives response, displays prediction + debug info in browser

Round-trip browser → Fargate → SageMaker → Fargate → browser: ~50–150ms total

---

## Task 12: Architecture Diagram + Demo Video

**Deliverables:** `solutions/architecture_diagram.png`, `solutions/demo_video.mp4`

**Git tag:** `task-12`

### Architecture Diagram

Generated with `solutions/generate_diagram.py` (matplotlib). Shows all required components:

| Component | AWS Service | Detail |
|-----------|-------------|--------|
| User browser | — | http://3.253.10.36:8501 |
| App container | ECS / Fargate | vaultech-app-service, 0.5 vCPU, 1 GB RAM |
| Docker image | Amazon ECR | 999390550986.dkr.ecr.eu-west-1.amazonaws.com/vaultech-app |
| Prediction API | SageMaker Endpoint | vaultech-bath-predictor, ml.t2.medium, XGBoost 3.0-5 |
| Model artifact | Amazon S3 | vaultech-models-999390550986/models/.../model.tar.gz |
| Model versions | Model Registry | vaultech-bath-predictor-group, MAE=0.92s, R²=0.69 |
| Historical data | Gold Parquet (in image) | data/gold/pieces.parquet, 169k pieces |

**Data flow arrows:** User → ECS → SageMaker → S3 (model load) → prediction back to user. Model Registry ↔ SageMaker. ECR → ECS (image pull).

### Demo Video Outline (5 minutes)

**Part 1 — Architecture (1 min):** Open the diagram, explain each component and how they connect.

**Part 2 — Live App (3 min):**
- Open http://3.253.10.36:8501 in browser
- Show filters (die matrix, date range, slow pieces toggle)
- Select a piece → show detail panel (cumulative times, partial times, bar chart)
- Point to inference debug panel: CSV payload sent, float response, latency (~40 ms)
- This proves prediction is coming from SageMaker, not local code

**Part 3 — Data Flow Summary (1 min):** Step-by-step: browser → Fargate (Streamlit) → invoke_endpoint (boto3) → SageMaker XGBoost container → float response → displayed in app.

### Key things to say in the video

- "The model predicts bath time after only the 2nd strike — about 18 seconds into the 58-second journey — early enough to raise a real-time alert while the piece is still on the line"
- "ECS Fargate is serverless — no EC2 instances to manage, AWS handles the underlying server"
- "The `SAGEMAKER_ENDPOINT_NAME` environment variable injected by the ECS task definition switches the app from local inference to cloud inference"
- "The debug panel shows the exact CSV payload sent and the raw float response — proving the round-trip to SageMaker"

---

## Environment Reference

| Command | What it does |
|---------|-------------|
| `uv sync` | Install/update dependencies |
| `uv run lab` | Launch JupyterLab (run in separate WSL terminal) |
| `uv run app` | Launch Streamlit dashboard |
| `docker compose up -d postgres flyway` | Start database (from `infra/` folder) |
| `sudo chmod 666 /var/run/docker.sock` | Fix Docker permissions in WSL (if needed) |

**Database credentials:**
- Host: localhost:5432
- DB: vaultech
- User: vaultech
- Password: vaultech_dev
