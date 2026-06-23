import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Career Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================

@st.cache_data
def load_data():
    df = pd.read_csv("Palo_Alto_Networks.csv")

    df["YearsAtCompany"] = df["YearsAtCompany"].replace(0, 1)

    df["PromotionGapRatio"] = (
        df["YearsSinceLastPromotion"] /
        df["YearsAtCompany"]
    )

    df["RoleStagnationIndex"] = (
        df["YearsInCurrentRole"] /
        df["YearsAtCompany"]
    )

    df["TrainingIntensityScore"] = (
        df["TrainingTimesLastYear"] /
        df["YearsAtCompany"]
    )

    conditions = [
        df["YearsSinceLastPromotion"] >= 5,
        df["YearsSinceLastPromotion"] >= 3
    ]

    df["PromotionGapScore"] = np.select(
        conditions,
        ["High", "Medium"],
        default="Low"
    )

    return df


df = load_data()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("Filters")

department = st.sidebar.multiselect(
    "Department",
    df["Department"].unique(),
    default=df["Department"].unique()
)

df = df[df["Department"].isin(department)]

clusters = st.sidebar.slider(
    "Number of Clusters",
    3,
    7,
    5
)

# ==========================
# CLUSTERING
# ==========================

features = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TrainingTimesLastYear",
    "MonthlyIncome",
    "JobLevel"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=clusters,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)

components = pca.fit_transform(X_scaled)

df["PC1"] = components[:, 0]
df["PC2"] = components[:, 1]

# ==========================
# HEADER
# ==========================

st.title("📊 Career Intelligence Dashboard")

st.markdown(
"""
Analyze promotion gaps, career stagnation,
retention opportunities and employee clusters.
"""
)

# ==========================
# KPI SECTION
# ==========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Employees",
        len(df)
    )

with col2:
    st.metric(
        "Attrition Count",
        int(df["Attrition"].sum())
    )

with col3:
    st.metric(
        "High Promotion Gap",
        int((df["PromotionGapScore"] == "High").sum())
    )

with col4:
    st.metric(
        "Avg Promotion Gap",
        round(df["YearsSinceLastPromotion"].mean(), 2)
    )

# ==========================
# CLUSTER VISUALIZATION
# ==========================

st.subheader("Career Path Clustering")

fig_cluster = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_data=[
        "Department",
        "JobRole",
        "YearsAtCompany"
    ]
)

st.plotly_chart(
    fig_cluster,
    use_container_width=True
)

# ==========================
# PROMOTION GAP ANALYSIS
# ==========================

st.subheader("Promotion Gap Analysis")

fig_gap = px.histogram(
    df,
    x="YearsSinceLastPromotion",
    nbins=15,
    color="PromotionGapScore"
)

st.plotly_chart(
    fig_gap,
    use_container_width=True
)

# ==========================
# DEPARTMENT ANALYSIS
# ==========================

st.subheader("Department Wise Promotion Gap")

dept_gap = (
    df.groupby("Department")
    ["YearsSinceLastPromotion"]
    .mean()
    .reset_index()
)

fig_dept = px.bar(
    dept_gap,
    x="Department",
    y="YearsSinceLastPromotion",
    color="YearsSinceLastPromotion"
)

st.plotly_chart(
    fig_dept,
    use_container_width=True
)

# ==========================
# RETENTION OPPORTUNITIES
# ==========================

st.subheader("Retention Opportunity Employees")

retention_df = df[
    (df["Attrition"] == 0) &
    (df["YearsSinceLastPromotion"] >= 3)
]

st.dataframe(
    retention_df[
        [
            "Department",
            "JobRole",
            "JobLevel",
            "YearsAtCompany",
            "YearsSinceLastPromotion",
            "MonthlyIncome"
        ]
    ],
    use_container_width=True
)

# ==========================
# CLUSTER SUMMARY
# ==========================

st.subheader("Cluster Summary")

summary = (
    df.groupby("Cluster")
    .agg({
        "YearsAtCompany":"mean",
        "YearsSinceLastPromotion":"mean",
        "MonthlyIncome":"mean",
        "Attrition":"mean"
    })
    .round(2)
)

st.dataframe(
    summary,
    use_container_width=True
)

# ==========================
# DOWNLOAD REPORT
# ==========================

csv = df.to_csv(index=False)

st.download_button(
    label="Download Processed Data",
    data=csv,
    file_name="career_analysis.csv",
    mime="text/csv"
)