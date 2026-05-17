# ♻️ EcoSort

Sistem klasifikasi sampah organik dan anorganik berbasis Computer Vision dan Machine Learning (SVM-RBF dengan fitur visual).

> **Computer Vision Project · Kelas LG01 · BINUS University · TA 2025/2026**

## 🌐 Live Demo

Aplikasi dapat diakses di: **[ecosort-ai.streamlit.app](https://ecosort-ai.streamlit.app)**

## 👥 Anggota Kelompok

- Clarencia Augustine (2802445311)
- Jason Hermawan (2802441875)
- Kevin Kagusman (2802429434)
- Nelson Filbert Erlanda Sakti (2802429270)
- Vania Alicia Yocelin (2802442612)

## 🔬 Metodologi

| Tahap            | Teknik                                                    |
| Preprocessing    | Resize 128×128, konversi RGB+HSV, normalisasi, augmentasi |
| Ekstraksi Fitur  | Color Histogram (HSV) + LBP + HOG → 2116-dim              |
| Reduksi Dimensi  | Principal Component Analysis (PCA)                        |
| Classifier Utama | SVM kernel RBF (binary: organik vs anorganik)             |
| Sub-Classifier   | SVM-RBF multiclass (6 sub-kategori anorganik)             |
| Tuning           | RandomizedSearchCV untuk parameter C dan gamma            |
| Evaluasi         | Accuracy, Precision, Recall, F1-Score, 5-Fold CV          |

## 📁 Struktur Project

```
ecosort-ai/
├── app.py                       # Streamlit web app
├── requirements.txt             # Python dependencies
├── packages.txt                 # System deps (libgl untuk OpenCV)
├── README.md
├── .gitignore
├── saved_model/                 # Model hasil training
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── svm_rbf_best.pkl
│   ├── scaler_subcat.pkl
│   ├── pca_subcat.pkl
│   ├── svm_subcat.pkl
│   └── subcat_classes.pkl
└── test/                        # Gambar test cases
    ├── sayur.jpg
    └── ...
```

## 🚀 Cara Menjalankan Secara Lokal

```bash
git clone https://github.com/USERNAME/ecosort-ai.git
cd ecosort-ai
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Dataset

[Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) dari Kaggle, dikelompokkan ulang menjadi:
- **Organik**: biological
- **Anorganik**: plastic, metal, glass, paper, cardboard, battery
