import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Retail Sales Dashboard",
                   page_icon="📊",
                   layout="wide")

st.title("📈 Retail Sales Analysis Dashboard")

uploaded_file = st.file_uploader(
    "Upload Retail Sales CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.sidebar.header("Filters")

    # Date Conversion
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    # Sidebar Filters
    region = st.sidebar.multiselect(
        "Region",
        df["Region"].unique(),
        default=df["Region"].unique()
    )

    category = st.sidebar.multiselect(
        "Category",
        df["Category"].unique(),
        default=df["Category"].unique()
    )

    filtered_df = df[
        (df.Region.isin(region)) &
        (df.Category.isin(category))
    ]

    st.subheader("Dataset")

    st.dataframe(filtered_df)

    # KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales",
                f"${filtered_df['Sales'].sum():,.0f}")

    col2.metric("Total Profit",
                f"${filtered_df['Profit'].sum():,.0f}")

    col3.metric("Orders",
                filtered_df.shape[0])

    st.divider()

    # Sales by Category
    st.subheader("Sales by Category")

    fig = px.bar(
        filtered_df.groupby("Category")["Sales"].sum().reset_index(),
        x="Category",
        y="Sales",
        color="Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Top Selling Products

    st.subheader("Top 10 Products")

    product = filtered_df.groupby("Product")["Sales"] \
                         .sum() \
                         .sort_values(ascending=False) \
                         .head(10)

    fig = px.bar(
        product,
        orientation='h',
        title="Top Selling Products"
    )

    st.plotly_chart(fig,
                    use_container_width=True)

    # Monthly Sales

    filtered_df["Month"] = filtered_df["Order Date"].dt.month_name()

    monthly = filtered_df.groupby("Month")["Sales"].sum().reset_index()

    st.subheader("Monthly Sales Trend")

    fig = px.line(
        monthly,
        x="Month",
        y="Sales",
        markers=True
    )

    st.plotly_chart(fig,
                    use_container_width=True)

    # Region Analysis

    st.subheader("Sales by Region")

    fig = px.pie(
        filtered_df,
        names="Region",
        values="Sales"
    )

    st.plotly_chart(fig,
                    use_container_width=True)

    # Profit vs Discount

    st.subheader("Profit vs Discount")

    fig = px.scatter(
        filtered_df,
        x="Discount",
        y="Profit",
        color="Category",
        size="Sales",
        hover_data=["Product"]
    )

    st.plotly_chart(fig,
                    use_container_width=True)

    # Correlation Heatmap

    st.subheader("Correlation Matrix")

    corr = filtered_df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig,
                    use_container_width=True)

    # Business Insights

    st.subheader("Business Insights")

    best = filtered_df.groupby("Product")["Sales"] \
                      .sum() \
                      .idxmax()

    best_month = filtered_df.groupby("Month")["Sales"] \
                            .sum() \
                            .idxmax()

    best_region = filtered_df.groupby("Region")["Sales"] \
                             .sum() \
                             .idxmax()

    st.success(f"🏆 Best Selling Product : {best}")
    st.success(f"📅 Highest Sales Month : {best_month}")
    st.success(f"🌍 Best Performing Region : {best_region}")

else:
    st.info("Upload a CSV file to begin analysis.")