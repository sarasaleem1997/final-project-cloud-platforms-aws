"""
Forging Line — Piece Travel Time Dashboard

Displays processed pieces with predicted bath time and per-stage
timing detail.

Usage:
    uv run streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vaultech_analysis.inference import Predictor

GOLD_FILE = PROJECT_ROOT / "data" / "gold" / "pieces.parquet"

# Column definitions — process order
PARTIAL_COLS = [
    "partial_furnace_to_2nd_strike_s",
    "partial_2nd_to_3rd_strike_s",
    "partial_3rd_to_4th_strike_s",
    "partial_4th_strike_to_auxiliary_press_s",
    "partial_auxiliary_press_to_bath_s",
]
PARTIAL_LABELS = [
    "Furnace → 2nd strike",
    "2nd strike → 3rd strike",
    "3rd strike → 4th strike",
    "4th strike → Aux. press",
    "Aux. press → Bath",
]
CUMULATIVE_COLS = [
    "lifetime_2nd_strike_s",
    "lifetime_3rd_strike_s",
    "lifetime_4th_strike_s",
    "lifetime_auxiliary_press_s",
    "lifetime_bath_s",
]
CUMULATIVE_LABELS = [
    "2nd strike (1st op)",
    "3rd strike (2nd op)",
    "4th strike (drill)",
    "Auxiliary press",
    "Bath",
]


@st.cache_resource
def load_predictor():
    return Predictor()


@st.cache_data
def load_data():
    predictor = load_predictor()
    df = pd.read_parquet(GOLD_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["predicted_bath_s"] = predictor.predict_batch(df)
    df["prediction_error_s"] = df["predicted_bath_s"] - df["lifetime_bath_s"]
    return df


@st.cache_data
def get_reference(_df):
    return _df.groupby("die_matrix")[PARTIAL_COLS + CUMULATIVE_COLS].median()


st.set_page_config(page_title="Forging Line Dashboard", layout="wide")
st.title("Forging Line — Piece Travel Time Dashboard")

df = load_data()
reference = get_reference(df)

# --- Sidebar filters ---
st.sidebar.header("Filters")

matrices = sorted(df["die_matrix"].unique())
selected_matrix = st.sidebar.selectbox("Die matrix", ["All"] + [str(m) for m in matrices])

min_date = df["timestamp"].dt.date.min()
max_date = df["timestamp"].dt.date.max()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

slow_only = st.sidebar.checkbox("Show slow pieces only (bath > 90th pct)")

# Apply filters
filtered = df.copy()

if selected_matrix != "All":
    filtered = filtered[filtered["die_matrix"] == int(selected_matrix)]

if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["timestamp"].dt.date >= start) &
        (filtered["timestamp"].dt.date <= end)
    ]

if slow_only:
    p90 = df.groupby("die_matrix")["lifetime_bath_s"].quantile(0.90)
    filtered = filtered[filtered["lifetime_bath_s"] > filtered["die_matrix"].map(p90)]

# --- Summary metrics ---
st.subheader("Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total pieces", f"{len(filtered):,}")
col2.metric("Median bath time", f"{filtered['lifetime_bath_s'].median():.1f}s")
col3.metric("Median predicted", f"{filtered['predicted_bath_s'].median():.1f}s")
col4.metric("MAE", f"{filtered['prediction_error_s'].abs().mean():.2f}s")

# --- Pieces table ---
st.subheader("Pieces")

table_cols = ["timestamp", "piece_id", "die_matrix", "lifetime_bath_s", "predicted_bath_s", "prediction_error_s", "oee_cycle_time_s"]
table_df = filtered[table_cols].copy()
table_df["timestamp"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

event = st.dataframe(
    table_df.reset_index(drop=True),
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
)

# --- Piece detail panel ---
selected_rows = event.selection.rows if event.selection else []

if selected_rows:
    idx = filtered.iloc[selected_rows[0]].name
    piece = df.loc[idx]
    ref = reference.loc[piece["die_matrix"]]

    st.subheader(f"Piece detail — {piece['piece_id']} (Matrix {piece['die_matrix']})")

    # Cumulative times vs reference
    st.markdown("**Cumulative travel times vs reference**")
    cum_data = []
    for col, label in zip(CUMULATIVE_COLS, CUMULATIVE_LABELS):
        actual = piece[col]
        ref_val = ref[col]
        deviation = actual - ref_val if pd.notna(actual) and pd.notna(ref_val) else None
        cum_data.append({
            "Stage": label,
            "Actual (s)": round(actual, 2) if pd.notna(actual) else None,
            "Reference (s)": round(ref_val, 2),
            "Deviation (s)": round(deviation, 2) if deviation is not None else None,
        })
    st.dataframe(pd.DataFrame(cum_data), use_container_width=True, hide_index=True)

    # Partial times vs reference
    st.markdown("**Partial times between stages vs reference**")
    partial_data = []
    for col, label in zip(PARTIAL_COLS, PARTIAL_LABELS):
        actual = piece[col]
        ref_val = ref[col]
        deviation = actual - ref_val if pd.notna(actual) and pd.notna(ref_val) else None
        status = "🔴 Slow" if deviation is not None and deviation > 2 else "🟢 OK"
        partial_data.append({
            "Segment": label,
            "Actual (s)": round(actual, 2) if pd.notna(actual) else None,
            "Reference (s)": round(ref_val, 2),
            "Deviation (s)": round(deviation, 2) if deviation is not None else None,
            "Status": status,
        })
    st.dataframe(pd.DataFrame(partial_data), use_container_width=True, hide_index=True)

    # Synoptic bar chart
    st.markdown("**Actual vs reference partial times (process synoptic)**")
    import matplotlib.pyplot as plt
    import numpy as np

    actual_vals = [piece[c] for c in PARTIAL_COLS]
    ref_vals = [ref[c] for c in PARTIAL_COLS]
    x = np.arange(len(PARTIAL_LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, actual_vals, width, label="Actual", color="steelblue")
    ax.bar(x + width/2, ref_vals, width, label="Reference", color="lightgray")
    ax.set_xticks(x)
    ax.set_xticklabels(PARTIAL_LABELS, rotation=20, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Partial times: Actual vs Reference")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Select a piece from the table above to see its per-stage timing detail.")
