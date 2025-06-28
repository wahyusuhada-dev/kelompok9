# 🎓 Prediksi dan Segmentasi Kelulusan Mahasiswa

> Studi Kasus: Student Grading Dataset (Kaggle)

## Kelompok 09 
 
1.	Difa Ramadhan 	220102023
2.	Nabila Tsari Aulia Mahmudah	220102064
3.	Siti Arfi Mutoharoh 	220102082

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi kelulusan mahasiswa serta mengelompokkan mereka ke dalam segmen berdasarkan karakteristik akademik menggunakan pendekatan Data Science Pipeline. Dataset diambil dari Kaggle dan telah melalui tahap pembersihan dan pra-pemrosesan data.

### ✅ Tujuan:
- Memprediksi apakah mahasiswa akan **lulus atau tidak** menggunakan model klasifikasi.
- Melakukan **segmentasi mahasiswa** menggunakan teknik clustering (unsupervised learning).
- Menyajikan hasil analisis dalam bentuk **dashboard interaktif menggunakan Streamlit**.

---

## 🧩 Dataset

- Sumber: Kaggle (Students Grading Dataset)
- File: `cleaned_data.csv`
- Fitur utama:
  - `age`, `gender`, `study_hours`, `attendance`, `GPA`, dll.
  - Target (label): `class` → 1 = Lulus, 0 = Tidak Lulus

---

## 🛠️ Tools & Library

- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Streamlit (untuk dashboard)
- Jupyter Notebook

---

## 🧪 Tahapan Analisis

1. **EDA & Preprocessing** → `eda_preprocessing.ipynb`
2. **Clustering** (KMeans) → `clustering.ipynb`
3. **Classification** (Logistic Regression, Decision Tree, dll) → `classification_model.ipynb`
4. **Dashboard** Streamlit → `dashboard_app.py`

---

## 🚀 Cara Menjalankan Dashboard

### 1. Clone repository ini:

```bash
git clone https://github.com/namakamu/kelulusan-dashboard.git
cd kelulusan-dashboard
