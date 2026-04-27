import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cluster Analysis", page_icon="📊")

st.title("📊 Netflix Cluster Analysis Dashboard")

# -----------------------------
# Step 1: Dataset Loading
# -----------------------------
st.write("### Step 1: Dataset Loading")

data_option = st.radio(
    "Select how you want to load your dataset:",
    ("Auto-load from project data folder", "Upload CSV manually")
)

df = None

# --- Option 1: Auto-load from project data folder ---
if data_option == "Auto-load from project data folder":
    file_choice = st.selectbox(
        "Choose dataset:",
        ("netflix_cleaned.csv", "netflix_titles.csv")
    )

    try:
        df = pd.read_csv(f"data/{file_choice}")
        st.success(f"✅ Successfully loaded `{file_choice}` from project data folder.")
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

# --- Option 2: Manual upload ---
elif data_option == "Upload CSV manually":
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully from upload.")

# --- Proceed only if data is loaded ---
if df is not None:
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Step 2: Auto Feature Engineering
    # -----------------------------
    st.write("### Step 2: Auto Feature Engineering")

    df_processed = df.copy()

    if 'release_year' in df.columns or 'duration' in df.columns:
        st.info("🎬 Netflix-like dataset detected — generating smart numeric features...")

        # Duration: extract minutes + flag for series
        if 'duration' in df.columns:
            df_processed['duration_minutes'] = df['duration'].str.extract('(\d+)').astype(float)
            df_processed['is_series'] = df['duration'].str.contains('Season', case=False, na=False).astype(int)

        # Genre count
        if 'listed_in' in df.columns:
            df_processed['num_genres'] = df['listed_in'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

        # Clean release year
        if 'release_year' in df.columns:
            df_processed['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

        # Description length
        if 'description' in df.columns:
            df_processed['desc_length'] = df['description'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
    else:
        st.warning("Generic dataset detected — using only numeric columns.")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        df_processed = df.copy()

    # -----------------------------
    # Step 3: Clustering
    # -----------------------------
    if len(numeric_cols) >= 2:
        st.write("### Step 3: Run Clustering")

        features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:2])
        k = st.slider("Select number of clusters (K)", 2, 10, 3)

        X = df_processed[features].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df_processed['Cluster'] = kmeans.fit_predict(X_scaled)

        # -----------------------------
        # Step 4: Cluster Summary
        # -----------------------------
        st.write("### Step 4: Cluster Summary")
        cluster_summary = (
            df_processed['Cluster']
            .value_counts()
            .reset_index()
            .rename(columns={'index': 'Cluster', 'Cluster': 'Count'})
        )
        st.dataframe(cluster_summary)

        # -----------------------------
        # Step 5: Visualization
        # -----------------------------
        st.write("### Step 5: Cluster Visualization")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[features[0]], X[features[1]], c=df_processed['Cluster'], cmap='plasma', s=50)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title("K-Means Clusters")
        st.pyplot(fig)

        # -----------------------------
        # Step 6: Cluster Centroids
        # -----------------------------
        st.write("### Step 6: Cluster Centroids")
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)
        st.dataframe(centroids)
    else:
        st.warning("Not enough numeric features found for clustering.")
