"""Streamlit demo: Retail Sales Forecasting.

Run locally:  streamlit run app.py
Or on Streamlit Cloud from this repo (Python 3.11+, see requirements.txt).

Pick a product category and a horizon. The app trains the LightGBM
model in memory on launch from the bundled Sample Superstore data
(9,994 orders, 2019-2022) — no model binary is stored in the repo —
then forecasts the next weeks recursively: each predicted week becomes
the lag input for the following week.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from forecast import (  # noqa: E402
    CATEGORIES,
    load_weekly,
    recursive_forecast,
    train_lightgbm,
    add_features,
    FEATURES,
)

st.set_page_config(page_title="Retail Sales Forecast", layout="wide")
st.title("Retail Sales Forecast — Sample Superstore")
# Sliders must stay left-to-right even in RTL browsers (e.g. Persian):
# without this fix, dragging the thumb right decreases the value.
st.markdown(
    """<style>
    [data-testid="stSlider"] { direction: ltr !important; }
    </style>""",
    unsafe_allow_html=True,
)
st.write(
    "Weekly sales per product category, forecast with LightGBM "
    "(lag + rolling + calendar features). Test-set results: "
    "LightGBM RMSE $4,101 vs seasonal-naive baseline $5,529."
)

category = st.selectbox("Category", CATEGORIES)
horizon = st.slider("Forecast horizon (weeks)", 4, 12, 8)


@st.cache_resource
def load_model():
    weekly = load_weekly()
    feat = add_features(weekly).dropna(subset=FEATURES).reset_index(drop=True)
    model = train_lightgbm(feat)
    return model, weekly


model, weekly = load_model()
hist = weekly[weekly["category"] == category].copy()
forecast = recursive_forecast(model, weekly, category, horizon=horizon)

combined = pd.DataFrame(
    {
        "Actual": hist.set_index("week")["sales"],
        "Forecast": forecast.set_index("week")["forecast"],
    }
)
st.line_chart(combined)

st.subheader(f"Next {horizon} weeks — {category}")
out = forecast.copy()
out["week"] = out["week"].dt.date
out["forecast"] = out["forecast"].round(0).astype(int)
st.dataframe(out, use_container_width=True)
st.caption(
    "Forecasts are recursive: each predicted week feeds the next week's "
    "lag features, so uncertainty grows with horizon. The underlying data "
    "is a fictional/demo retail sample, not a real store's sales."
)
