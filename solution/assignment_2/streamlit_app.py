from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "operational"


st.set_page_config(page_title="Assignment 2 Forecast Viewer", layout="wide")
st.title("Assignment 2 Forecast Viewer")

forecast_files = sorted(DEFAULT_OUTPUT_ROOT.glob("*/forecast_results.csv"))
if not forecast_files:
    st.info("No operational forecast outputs found yet. Run `python run_pipeline.py --config configs/next_day_lightgbm.yaml` first.")
    st.stop()

selected = st.selectbox("Forecast output", forecast_files, format_func=lambda p: str(p.relative_to(ROOT)))
df = pd.read_csv(selected)

st.subheader("Forecast Preview")
st.dataframe(df.head(100), use_container_width=True)

metric_cols = [col for col in ["actual_pd", "predicted_pd"] if col in df.columns]
if len(metric_cols) == 2:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["abs_error"] = (df["predicted_pd"] - df["actual_pd"]).abs()
    wmape = df["abs_error"].sum() / df["actual_pd"].abs().sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("MAE", f"{df['abs_error'].mean():.3f}")
    c3.metric("WMAPE", f"{wmape:.3f}")

    st.subheader("Actual vs Predicted")
    bus_options = sorted(df["bus_id"].dropna().unique()) if "bus_id" in df.columns else []
    if bus_options:
        bus_id = st.selectbox("Bus", bus_options)
        plot_df = df[df["bus_id"] == bus_id].sort_values("timestamp")
    else:
        plot_df = df.sort_values("timestamp")
    st.line_chart(plot_df.set_index("timestamp")[["actual_pd", "predicted_pd"]])

    st.subheader("Error By Zone")
    if "zone_name" in df.columns:
        zone_error = df.groupby("zone_name", as_index=False)["abs_error"].mean().sort_values("abs_error", ascending=False)
        st.dataframe(zone_error, use_container_width=True)
else:
    st.warning("Forecast output does not include both actual and predicted values, so metrics are not shown.")
