"""
Observability dashboard reading straight from logs.db.
This is the piece most portfolio projects skip — worth highlighting in
interviews.

Run: streamlit run frontend/dashboard.py
"""
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).parent.parent / "logs.db"

st.set_page_config(page_title="Triage Observability", page_icon="📊")
st.title("📊 Support Triage — Observability Dashboard")

if not DB_PATH.exists():
    st.info("No requests logged yet. Run the app and triage a few tickets first.")
    st.stop()

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM requests ORDER BY timestamp DESC", conn)
conn.close()

if df.empty:
    st.info("No requests logged yet.")
    st.stop()

total_requests = len(df)
success_rate = df["success"].mean() * 100
avg_latency = df["latency_ms"].mean()
avg_cost = df["cost_usd"].mean()
total_cost = df["cost_usd"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total requests", total_requests)
col2.metric("Success rate", f"{success_rate:.1f}%")
col3.metric("Avg latency", f"{avg_latency:.0f} ms")
col4.metric("Avg cost/ticket", f"${avg_cost:.5f}")

st.caption(f"Total spend logged: ${total_cost:.4f}")

st.subheader("Category breakdown")
if df["success"].sum() > 0:
    st.bar_chart(df[df["success"] == 1]["category"].value_counts())

st.subheader("Latency over time")
df_sorted = df.sort_values("timestamp")
st.line_chart(df_sorted.set_index("timestamp")["latency_ms"])

st.subheader("Recent requests")
st.dataframe(
    df[["timestamp", "category", "urgency", "success", "latency_ms", "cost_usd", "error"]].head(20),
    use_container_width=True,
)
