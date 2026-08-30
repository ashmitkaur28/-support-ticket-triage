"""
Simple Streamlit frontend. Calls the FastAPI backend, which must be
running separately (uvicorn api.main:app --reload).

Run: streamlit run frontend/app.py
"""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/triage"

st.set_page_config(page_title="Support Ticket Triage", page_icon="🎫")
st.title("🎫 Support Ticket Triage & Response Assistant")
st.caption("Paste a customer ticket below to classify it and draft a grounded reply.")

ticket_text = st.text_area("Ticket text", height=150, placeholder="e.g. My order hasn't arrived in 3 weeks...")

if st.button("Triage ticket", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket first.")
    else:
        with st.spinner("Classifying and drafting response..."):
            try:
                resp = requests.post(API_URL, json={"text": ticket_text}, timeout=30)
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is it running? (`uvicorn api.main:app --reload`)")
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
