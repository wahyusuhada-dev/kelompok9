# === Import Library ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.model_selection import train_test_split

# === Konfigurasi Halaman ===
st.set_page_config(layout="wide", page_title="Prediksi dan Segmentasi Kelulusan Mahasiswa")

# === STREAMLIT APP ===
st.title("🎓 Prediksi dan Segmentasi Kelulusan Mahasiswa")
st.sidebar.title("📂 Menu Dashboard")

# === Upload Dataset ===
uploaded_file = st.sidebar.file_uploader("Unggah file cleaned_data.csv", type=["csv"])
data_processed = False

# === Load & Preprocess Data ===
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File berhasil diunggah!")

        # Label Encode target jika perlu
        if df['class'].dtype == 'object':
            df['class'] = df['class'].map({'Graduate': 1, 'Not Graduate': 0})

        # Fitur dan target
        X = df.drop('class', axis=1)
        y = df['class']

        # Standardisasi fitur
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Klasifikasi
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, cluster_labels)

        # PCA untuk visualisasi
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]
        df['Cluster'] = cluster_labels

        data_processed = True

    except Exception as e:
        st.error(f"Terjadi error saat membaca data: {e}")
        data_processed = False

# === Menu Navigasi ===
menu = st.sidebar.radio("Navigasi", [
    "🏠 Overview",
    "📊 Eksplorasi Data",
    "📈 Klasifikasi",
    "🧩 Segmentasi Mahasiswa"
])

if menu == "🏠 Overview":
    st.header("📘 Studi Kasus: Prediksi dan Segmentasi Kelulusan Mahasiswa")
    st.markdown("""
    **Judul Proyek:** Prediksi dan Segmentasi Kelulusan Mahasiswa Menggunakan Data Science Pipeline  
    **Sumber Dataset:** Students Grading Dataset dari Kaggle  
    
    **Tujuan:**
    - Memprediksi kelulusan mahasiswa berdasarkan nilai dan atribut lainnya
    - Mengelompokkan mahasiswa menjadi beberapa segmen berdasarkan performa
    
    **Metodologi:**
    - Preprocessing dan EDA
    - Klasifikasi menggunakan Logistic Regression
    - Segmentasi menggunakan KMeans + PCA
    """)

elif menu == "📊 Eksplorasi Data":
    st.header("📊 Eksplorasi Data Mahasiswa")
    if data_processed:
        st.subheader("📝 Statistik Deskriptif")
        st.write(df.describe())
        st.subheader("🔍 Korelasi")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.drop(['PCA1', 'PCA2', 'Cluster'], axis=1, errors='ignore').corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Silakan unggah file cleaned_data.csv terlebih dahulu.")

elif menu == "📈 Klasifikasi":
    st.header("📈 Prediksi Kelulusan Mahasiswa")
    if data_processed:
        st.metric("Akurasi Model", f"{acc*100:.2f}%")
        st.subheader("📋 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())
    else:
        st.info("Silakan unggah file cleaned_data.csv terlebih dahulu.")

elif menu == "🧩 Segmentasi Mahasiswa":
    st.header("🧩 Segmentasi Mahasiswa berdasarkan PCA")
    if data_processed:
        st.metric("Silhouette Score", f"{sil_score:.4f}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax)
        ax.set_title("Visualisasi Clustering Mahasiswa")
        st.pyplot(fig)
    else:
        st.info("Silakan unggah file cleaned_data.csv terlebih dahulu.")
