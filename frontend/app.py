"""
Simple Streamlit frontend. Calls the FastAPI backend.

Run: streamlit run frontend/app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000") + "/triage"

st.set_page_config(page_title="Support Ticket Triage", page_icon="🎫", layout="wide")
st.title("🎫 Support Ticket Triage & Response Assistant")
st.caption("Paste a customer ticket below to classify it and draft a grounded reply.")
st.caption("Note: backend runs on a free tier — the first request after inactivity may take up to a minute to wake up.")

ticket_text = st.text_area("Ticket text", height=150, placeholder="e.g. My order hasn't arrived in 3 weeks...")

if st.button("Triage ticket", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket first.")
    else:
        with st.spinner("Classifying and drafting response... (may take longer on first request)"):
            try:
                resp = requests.post(API_URL, json={"text": ticket_text}, timeout=90)
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Check that the backend URL is correct and awake.")
                st.stop()

        if data.get("error"):
            st.error(data["error"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Category", data["category"])
            col2.metric("Urgency", data["urgency"])
            col3.metric("Latency", f"{data['latency_ms']:.0f} ms")

            st.subheader("Draft response")
            st.write(data["draft_response"])

            if data["retrieved_sources"]:
                st.caption(f"Grounded in: {', '.join(data['retrieved_sources'])}")