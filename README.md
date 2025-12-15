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
## Deployment

Bagian ini menjelaskan cara men-deploy model Mushroom Classification sebagai aplikasi **REST API** menggunakan **FastAPI** dan **Docker**. Panduan ini disesuaikan untuk dijalankan pada lingkungan **Play with Docker**.

### Tahapan Deployment

### 1. Persiapan dan Mengkloning Repositori
Langkah pertama dalam proses deployment di lingkungan **Play with Docker** adalah mengunduh *source code* proyek. Perintah ini akan menyalin repositori dari GitHub ke dalam *instance* Docker dan mengarahkan terminal ke direktori proyek yang sesuai.

Jalankan perintah berikut di terminal:

```bash
git clone https://github.com/sains-data/MLOPS_11_RA.git
cd MLOPS_11_RA
```
### 2. Membangun Docker Image
Setelah masuk ke direktori proyek, langkah selanjutnya adalah membangun **Docker image**. Proses ini akan membaca instruksi di dalam file `Dockerfile`, menyusun lingkungan aplikasi, dan menginstal seluruh dependensi yang terdaftar di `requirements.txt`.

Jalankan perintah berikut di terminal:

```bash
docker build -t mushroom-classification .
```

### 3. Menjalankan Docker Container
Langkah selanjutnya adalah menjalankan image yang telah dibangun menjadi sebuah container. Kita akan menggunakan opsi `-d` (*detached mode*) agar aplikasi berjalan di latar belakang dan tidak memblokir terminal, serta memetakan port internal container (80) ke port akses (80).

Jalankan perintah berikut:

```bash
docker run -d -p 80:80 mushroom-classification
```
Penjelasan:

-d: Menjalankan container di background. Terminal akan tetap aktif untuk perintah selanjutnya.

-p 80:80: Menghubungkan port 80 pada lingkungan Play with Docker dengan port 80 di aplikasi FastAPI.

### 4. Verifikasi dan Akses Aplikasi
Karena container berjalan di latar belakang, kita perlu memastikan bahwa container sudah aktif sebelum membukanya di browser.
a. Cek Status Container Ketik perintah berikut untuk melihat daftar container yang sedang berjalan:
```bash
docker ps
```
Pastikan Anda melihat output yang menampilkan Container ID dan status "Up". Jika statusnya Exited, berarti terjadi kesalahan (gunakan docker logs <container_id> untuk mengecek).

b. Membuka Port di Browser Di lingkungan Play with Docker, akses ke aplikasi dilakukan melalui tautan port yang tersedia:

Cek Badge Port: Lihat di bagian atas layar terminal (di sebelah informasi IP Address). Biasanya akan muncul tautan biru bertuliskan 80 secara otomatis.

Jika Badge Tidak Muncul:

Klik tombol "OPEN PORT" di bagian atas.

Masukkan angka 80 pada kolom yang muncul.

Klik OK.

Akses Aplikasi: Klik tautan 80 tersebut. Browser akan membuka tab baru menuju aplikasi Anda.

Tips: Setelah tab baru terbuka, tambahkan /docs di akhir URL (misalnya: ...play-with-docker.com/docs) untuk masuk ke halaman interaktif Swagger UI dan mulai menguji prediksi jamur.


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

