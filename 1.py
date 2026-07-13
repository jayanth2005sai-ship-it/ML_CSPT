import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Universal Data Visualizer",
                   layout="wide")

st.title("📊 Universal Dataset Visualization Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload any CSV dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

##############################################
# Convert date columns automatically
##############################################

for col in df.columns:
    try:
        df[col] = pd.to_datetime(df[col])
    except:
        pass

original_df = df.copy()

##############################################
# Sidebar Filters
##############################################

st.sidebar.header("Filters")

# Global Search

search = st.sidebar.text_input("Search")

if search:
    mask = np.column_stack([
        df[col].astype(str).str.contains(search, case=False, na=False)
        for col in df.columns
    ])
    df = df.loc[mask.any(axis=1)]

##############################################
# Numeric Filters
##############################################

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:

    min_val = float(df[col].min())
    max_val = float(df[col].max())

    selected = st.sidebar.slider(
        col,
        min_val,
        max_val,
        (min_val, max_val)
    )

    df = df[
        (df[col] >= selected[0]) &
        (df[col] <= selected[1])
    ]

##############################################
# Category Filters
##############################################

cat_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns

for col in cat_cols:

    options = st.sidebar.multiselect(
        col,
        df[col].dropna().unique(),
        default=df[col].dropna().unique()
    )

    df = df[df[col].isin(options)]

##############################################
# Date Filters
##############################################

date_cols = df.select_dtypes(include="datetime").columns

for col in date_cols:

    start = df[col].min()
    end = df[col].max()

    dates = st.sidebar.date_input(
        col,
        [start, end]
    )

    if len(dates) == 2:

        df = df[
            (df[col] >= pd.Timestamp(dates[0])) &
            (df[col] <= pd.Timestamp(dates[1]))
        ]

##############################################
# Dataset Summary
##############################################

st.subheader("Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("Missing Values", int(df.isna().sum().sum()))

st.dataframe(df, use_container_width=True)

##############################################
# Missing Values
##############################################

with st.expander("Missing Value Summary"):

    miss = df.isna().sum()

    st.dataframe(
        miss.reset_index().rename(
            columns={
                "index":"Column",
                0:"Missing"
            }
        )
    )

##############################################
# Download
##############################################

csv = df.to_csv(index=False)

st.download_button(
    "Download Filtered Dataset",
    csv,
    "filtered.csv",
    "text/csv"
)

##############################################
# Visualization
##############################################

st.header("Visualization")

chart = st.selectbox(
    "Chart Type",
    [
        "Scatter",
        "Line",
        "Bar",
        "Histogram",
        "Box",
        "Pie"
    ]
)

x = st.selectbox("X-axis", df.columns)

y = st.selectbox(
    "Y-axis",
    ["None"] + list(df.columns)
)

color = st.selectbox(
    "Color",
    ["None"] + list(df.columns)
)

facet = st.selectbox(
    "Facet",
    ["None"] + list(df.columns)
)

color = None if color == "None" else color
facet = None if facet == "None" else facet

fig = None

if chart == "Scatter":

    fig = px.scatter(
        df,
        x=x,
        y=None if y=="None" else y,
        color=color,
        facet_col=facet,
        hover_data=df.columns
    )

elif chart == "Line":

    fig = px.line(
        df,
        x=x,
        y=None if y=="None" else y,
        color=color
    )

elif chart == "Bar":

    fig = px.bar(
        df,
        x=x,
        y=None if y=="None" else y,
        color=color
    )

elif chart == "Histogram":

    fig = px.histogram(
        df,
        x=x,
        color=color
    )

elif chart == "Box":

    fig = px.box(
        df,
        x=x,
        y=None if y=="None" else y,
        color=color
    )

elif chart == "Pie":

    fig = px.pie(
        df,
        names=x,
        values=None if y=="None" else y
    )

st.plotly_chart(fig, use_container_width=True)

##############################################
# Correlation Matrix
##############################################

if len(numeric_cols) > 1:

    st.header("Correlation Heatmap")

    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig, use_container_width=True)

##############################################
# Distribution
##############################################

st.header("Distribution")

num = st.selectbox(
    "Numeric Column",
    numeric_cols
)

fig = px.histogram(
    df,
    x=num,
    marginal="box"
)

st.plotly_chart(fig, use_container_width=True)

##############################################
# Pair Scatter
##############################################

if len(numeric_cols) >= 2:

    st.header("Scatter Relationship")

    c1, c2 = st.columns(2)

    xcol = c1.selectbox(
        "X",
        numeric_cols,
        key="sx"
    )

    ycol = c2.selectbox(
        "Y",
        numeric_cols,
        key="sy"
    )

    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        color=color
    )

    st.plotly_chart(fig, use_container_width=True)

##############################################
# Statistics
##############################################

st.header("Statistics")

st.dataframe(df.describe(include="all").T)