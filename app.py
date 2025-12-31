import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

st.set_page_config(page_title="Wheat Seeds Clustering", layout="wide")
st.title("🌾 Wheat Seeds Clustering Application")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Rename columns
    data = data.rename(columns={
        'Kernel.Length': 'K_length',
        'Kernel.Width': 'K_width',
        'Asymmetry.Coeff': 'A_coeff',
        'Kernel.Groove': 'K_groove'
    })
    
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    
    # Feature selection
    x = data[['Area', 'K_length']]
    
    # Scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    
    # Sidebar for clustering parameters
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 1, 10, 3)
    
    # KMeans fit
    k_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    k_model.fit(x_scaled)
    k_labels = k_model.labels_
    k_centers = k_model.cluster_centers_
    
    data['Clusters'] = k_labels
    
    # PCA and Kaiser's rule
    pca = PCA()
    pca.fit(x_scaled)
    eigen_values = pca.explained_variance_
    kaiser_clusters = np.sum(eigen_values > 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Kaiser's Rule Suggests", f"{kaiser_clusters} clusters")
        st.metric("Inertia", f"{k_model.inertia_:.2f}")
    
    with col2:
        st.metric("Number of Samples", len(data))
        st.metric("Features Used", len(x.columns))
    
    # Plot eigen values
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, len(eigen_values) + 1), eigen_values, marker='o')
    ax1.axhline(y=1, color="r", linestyle="--", label="Kaiser Threshold")
    ax1.set_title("Eigenvalues (PCA)")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.legend()
    st.pyplot(fig1)
    
    # Elbow method
    inertia = []
    k_values = range(1, 11)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(x_scaled)
        inertia.append(km.inertia_)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(list(k_values), inertia, marker='o')
    ax2.set_title("Elbow Method")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Inertia")
    st.pyplot(fig2)
    
    # Cluster visualization
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    scatter = ax3.scatter(x_scaled.Area, x_scaled.K_length, c=k_labels, cmap="viridis", s=50)
    ax3.scatter(k_centers[:, 0], k_centers[:, 1], marker="X", color="red", s=200, edgecolors="black")
    ax3.set_xlabel("Area (scaled)")
    ax3.set_ylabel("K_length (scaled)")
    ax3.set_title("Clusters with Centers")
    plt.colorbar(scatter, ax=ax3, label="Cluster")
    st.pyplot(fig3)
    
    # Display clustered data
    st.subheader("Clustered Data")
    st.dataframe(data[['Area', 'K_length', 'Clusters']])
    
    # Download results
    csv = data.to_csv(index=False)
    st.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")