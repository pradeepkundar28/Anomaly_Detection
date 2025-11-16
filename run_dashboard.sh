#!/bin/bash
# Convenience script to run Streamlit dashboard

cd "$(dirname "$0")"
streamlit run src/dashboard/app_streamlit.py \
    --server.port=8501 \
    --server.address=0.0.0.0
