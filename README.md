# Mushroom Classification MLOps Pipeline

## Project Overview
Proyek ini merupakan implementasi pipeline Machine Learning Operations (MLOps)
secara end-to-end untuk menyelesaikan permasalahan klasifikasi jamur berdasarkan
deskripsi karakteristiknya. Model machine learning dikembangkan untuk
mengklasifikasikan jamur ke dalam dua kelas utama, yaitu **edible** dan
**poisonous**.

Pipeline ini dirancang dengan pendekatan terstruktur yang mencakup versioning
data, pelacakan eksperimen, deployment model, serta monitoring performa model
pada lingkungan produksi.

---

## Problem Statement
Kesalahan dalam mengidentifikasi jamur yang dapat dikonsumsi (edible) dan yang
beracun (poisonous) dapat menimbulkan risiko kesehatan yang serius. Oleh karena
itu, diperlukan sebuah sistem klasifikasi otomatis yang mampu memprediksi
kategori jamur secara akurat berdasarkan deskripsi karakteristik morfologinya.

Permasalahan utama yang diangkat dalam proyek ini adalah:
> *Bagaimana membangun pipeline MLOps yang mampu melakukan klasifikasi jamur
berdasarkan deskripsi fitur kategorikal secara end-to-end, mulai dari pengelolaan
data hingga monitoring model di lingkungan produksi.*

---

## Objectives
Tujuan dari proyek ini adalah:
1. Mengembangkan model machine learning untuk klasifikasi jamur berdasarkan
   deskripsi karakteristiknya.
2. Mengimplementasikan pipeline MLOps yang terstruktur dan reproducible.
3. Mengelola versioning data menggunakan **DVC**.
4. Melakukan pelacakan eksperimen dan parameter model menggunakan **MLflow**.
5. Mendeploy model ke dalam REST API menggunakan **FastAPI** dan **Docker**.
6. Menerapkan monitoring model untuk mendeteksi potensi *data drift* dan
   *prediction drift*.

---

## Dataset Description
Dataset yang digunakan berisi deskripsi karakteristik jamur yang bersifat
kategorikal, seperti:
- Bentuk dan warna tudung jamur
- Warna dan bentuk insang
- Bau (odor)
- Habitat jamur
- Karakteristik morfologi lainnya

Setiap observasi jamur memiliki label kelas berupa:
- **edible** : jamur yang aman dikonsumsi
- **poisonous** : jamur yang beracun

Dataset dikelola dan dilakukan versioning menggunakan **DVC** untuk memastikan
konsistensi dan reprodusibilitas eksperimen.

---

## MLOps Workflow
Pipeline MLOps pada proyek ini mencakup tahapan berikut:

1. **Data Versioning**
   - Dataset dikelola menggunakan DVC
   - Setiap perubahan data tercatat dan dapat direproduksi

2. **Data Processing & Feature Engineering**
   - Pembersihan data
   - Encoding fitur kategorikal
   - Pembagian data latih dan data uji

3. **Model Training**
   - Pelatihan model klasifikasi
   - Penyesuaian parameter model

4. **Experiment Tracking**
   - Pelacakan parameter, metrik, dan artefak model menggunakan MLflow

5. **Model Deployment**
   - Deployment model sebagai REST API menggunakan FastAPI
   - Containerization menggunakan Docker

6. **Model Monitoring**
   - Monitoring performa model
   - Deteksi *data drift* dan *prediction drift*

---

## Project Structure
Struktur direktori utama pada proyek ini adalah sebagai berikut:
├── data/ # Dataset dan artefak data
├── docs/ # Dokumentasi proyek
├── models/ # Model hasil training
├── steps/ # Tahapan pipeline MLOps
├── tests/ # Unit testing
├── app.py # FastAPI application
├── main.py # Entry point pipeline
├── dataset.py # Data loading dan preprocessing
├── config.yml # Konfigurasi pipeline dan eksperimen
├── dockerfile # Konfigurasi Docker
├── requirements.txt # Dependensi proyek
├── data.dvc # DVC tracking file
└── README.md # Dokumentasi proyek


---

## Tools & Technologies
Teknologi yang digunakan dalam proyek ini meliputi:
- **Python**
- **Scikit-learn**
- **DVC (Data Version Control)**
- **MLflow**
- **FastAPI**
- **Docker**
- **Evidently AI** (monitoring model)

---

## Expected Output
Output dari sistem ini berupa prediksi kelas jamur:
- **edible**
- **poisonous**

Prediksi dapat diakses melalui REST API yang telah dideploy, serta dimonitor
secara berkelanjutan untuk menjaga performa model di lingkungan produksi.

---

## License
Proyek ini menggunakan lisensi Apache 2.0.

