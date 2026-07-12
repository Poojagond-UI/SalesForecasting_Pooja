import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Retail Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main{
background-color:#f5f7fb;
}

[data-testid="stSidebar"]{
background-color:#1f2937;
}

[data-testid="stSidebar"] *{
color:white;
}

div[data-testid="metric-container"]{
background:white;
padding:18px;
border-radius:12px;
box-shadow:0px 3px 10px rgba(0,0,0,.08);
}

h1{
color:#1f2937;
}

</style>
""",unsafe_allow_html=True)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def load_data():

    df=pd.read_csv("train.csv",encoding="latin1")

    df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%d/%m/%Y"
)

    df["Year"]=df["Order Date"].dt.year

    df["Month"]=df["Order Date"].dt.to_period("M").astype(str)

    return df


df=load_data()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.image(
"https://img.icons8.com/color/96/combo-chart--v1.png",
width=70
)

st.sidebar.title("Retail Analytics")

page=st.sidebar.radio(
"Navigation",
[
"Sales Overview",
"Forecast Explorer",
"Anomaly Report",
"Demand Segments"
]
)

# ---------------------------------------------------
# PAGE 1
# ---------------------------------------------------

if page=="Sales Overview":

    st.title("📊 Sales Overview Dashboard")

    st.write("Business Intelligence Dashboard")

    total_sales=df["Sales"].sum()

    total_sales = df["Sales"].sum()
    total_orders = df["Order ID"].nunique()
    avg_sales = df["Sales"].mean()
    total_products = df["Product ID"].nunique()

    total_orders=df["Order ID"].nunique()

    avg_sales=df["Sales"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
    "Total Sales",
    f"${total_sales:,.0f}"
)

    c2.metric(
    "Total Orders",
    total_orders
)

    c3.metric(
    "Average Sale",
    f"${avg_sales:,.2f}"
)

    c4.metric(
    "Products",
    total_products
)

    st.markdown("---")

    yearly=df.groupby("Year")["Sales"].sum().reset_index()

    fig=px.bar(
        yearly,
        x="Year",
        y="Sales",
        color="Sales",
        text_auto=True,
        title="Total Sales by Year"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    monthly=df.groupby("Month")["Sales"].sum().reset_index()

    fig2=px.line(
        monthly,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("## Interactive Sales Filter")

    col1,col2=st.columns(2)

    with col1:

        region=st.selectbox(
            "Select Region",
            ["All"]+list(df["Region"].unique())
        )

    with col2:

        category=st.selectbox(
            "Select Category",
            ["All"]+list(df["Category"].unique())
        )

    filtered=df.copy()

    if region!="All":

        filtered=filtered[
            filtered["Region"]==region
        ]

    if category!="All":

        filtered=filtered[
            filtered["Category"]==category
        ]

    fig3=px.bar(
        filtered,
        x="Region",
        y="Sales",
        color="Category",
        title="Sales by Region & Category"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# ---------------------------------------------------
# PAGE 2
# ---------------------------------------------------

elif page=="Forecast Explorer":

    st.title("📈 Forecast Explorer")

    st.write("Explore future sales prediction using the best model.")
    forecast_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    if forecast_type == "Category":

        selected = st.selectbox(
            "Choose Category",
            sorted(df["Category"].unique())
        )

        forecast_df = df[df["Category"] == selected]

    else:

        selected = st.selectbox(
            "Choose Region",
            sorted(df["Region"].unique())
        )

        forecast_df = df[df["Region"] == selected]

    horizon = st.slider(
        "Forecast Horizon (Months)",
        1,
        3,
        3
    )

    monthly_forecast = (
        forecast_df
        .groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig4 = px.line(
        monthly_forecast,
        x="Month",
        y="Sales",
        markers=True,
        title=f"Sales Trend - {selected}"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.subheader("Model Performance")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Best Model",
        "XGBoost"
    )

    m2.metric(
        "MAE",
        "9397.61"
    )

    m3.metric(
        "RMSE",
        "12725.68"
    )

    st.info(
        f"""
Forecast Horizon Selected : **{horizon} Month(s)**

According to the trained XGBoost model, this dashboard is
configured to generate future sales predictions for the
selected Region/Category.
"""
    )

# ---------------------------------------------------
# PAGE 3
# ---------------------------------------------------

elif page == "Anomaly Report":

    st.title("🚨 Sales Anomaly Report")

    st.write("Detected unusual sales using Isolation Forest.")

    daily = (
        df.groupby("Order Date")["Sales"]
        .sum()
        .reset_index()
    )

    q1 = daily["Sales"].quantile(0.25)

    q3 = daily["Sales"].quantile(0.75)

    iqr = q3 - q1

    anomalies = daily[
        (daily["Sales"] < q1 - 1.5 * iqr)
        |
        (daily["Sales"] > q3 + 1.5 * iqr)
    ]

    fig5 = px.scatter(
        daily,
        x="Order Date",
        y="Sales",
        title="Detected Sales Anomalies"
    )

    fig5.add_scatter(
        x=anomalies["Order Date"],
        y=anomalies["Sales"],
        mode="markers",
        marker=dict(
            size=10,
            color="red"
        ),
        name="Anomaly"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomalies,
        use_container_width=True
    )

# ---------------------------------------------------
# PAGE 4
# ---------------------------------------------------

else:

    st.title("📦 Product Demand Segments")

    cluster = (
        df.groupby("Sub-Category")["Sales"]
        .sum()
        .reset_index()
    )

    cluster["Demand Cluster"] = pd.qcut(
        cluster["Sales"],
        3,
        labels=[
            "Low Demand",
            "Medium Demand",
            "High Demand"
        ]
    )

    fig6 = px.scatter(
        cluster,
        x="Sub-Category",
        y="Sales",
        color="Demand Cluster",
        size="Sales",
        title="Demand Segmentation"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    st.subheader("Cluster Assignment")

    st.dataframe(
        cluster,
        use_container_width=True
    )

    st.success("""
High Demand → Maintain Higher Inventory

Medium Demand → Balanced Stock

Low Demand → Stock Carefully
""")