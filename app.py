"""Interactive explorer for Kaggle's Global Shark Attacks dataset."""

from pathlib import Path

import kagglehub
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Global Shark Attacks Explorer", page_icon="🦈", layout="wide"
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Download the latest Kaggle dataset release and load its largest CSV."""
    dataset_path = Path(kagglehub.dataset_download("teajay/global-shark-attacks"))
    csv_files = list(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file was found in the downloaded dataset.")
    data_file = max(csv_files, key=lambda file: file.stat().st_size)
    # This Kaggle CSV is Windows-1252 encoded (not UTF-8).
    return pd.read_csv(data_file, encoding="cp1252", low_memory=False)


def download_frame(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


st.title("🦈 Global Shark Attacks Explorer")
st.caption(
    "Explore reported shark attacks worldwide using the latest Global Shark Attacks dataset."
)
try:
    with st.spinner("Downloading the latest dataset from Kaggle…"):
        data = load_data()
except Exception as error:
    st.error(f"I couldn't load the Kaggle dataset: {error}")
    st.info("Check your internet connection and Kaggle access, then refresh the page.")
    st.stop()

if data.empty:
    st.warning("This file has no rows to explore.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    filtered = data.copy()
    filter_column = st.selectbox("Filter column", ["None", *data.columns.tolist()])
    if filter_column != "None":
        series = data[filter_column]
        if pd.api.types.is_numeric_dtype(series):
            low, high = float(series.min()), float(series.max())
            selected_range = st.slider(
                f"{filter_column} range", low, high, (low, high)
            )
            filtered = filtered[filtered[filter_column].between(*selected_range)]
        else:
            values = sorted(series.dropna().astype(str).unique().tolist())
            selected_values = st.multiselect(
                f"{filter_column} values", values, default=values
            )
            filtered = filtered[filtered[filter_column].astype(str).isin(selected_values)]

numeric_columns = filtered.select_dtypes(include="number").columns.tolist()
missing_values = int(filtered.isna().sum().sum())
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(filtered):,}")
col2.metric("Columns", len(filtered.columns))
col3.metric("Missing values", f"{missing_values:,}")

left, right = st.columns([3, 2])
with left:
    st.subheader("Data preview")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

with right:
    st.subheader("Quick chart")
    if numeric_columns:
        y_column = st.selectbox("Numeric measure", numeric_columns)
        x_column = st.selectbox("Category or index", ["Row index", *filtered.columns])
        chart_data = filtered[[y_column]].copy()
        if x_column != "Row index":
            chart_data.index = filtered[x_column].astype(str)
        st.bar_chart(chart_data)
    else:
        st.info("Add at least one numeric column to build a chart.")

with st.expander("Column details"):
    details = pd.DataFrame(
        {
            "column": filtered.columns,
            "type": filtered.dtypes.astype(str).values,
            "non-null": filtered.notna().sum().values,
            "unique": filtered.nunique(dropna=True).values,
        }
    )
    st.dataframe(details, use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered shark-attack data",
    data=download_frame(filtered),
    file_name="filtered_shark_attacks.csv",
    mime="text/csv",
)
