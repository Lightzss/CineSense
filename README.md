# 🎬 CineSense - AI Movie Analyst

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)

**CineSense** merupakan aplikasi *Artificial Intelligence* (AI) yang dirancang untuk melakukan analisis sentimen terhadap komentar dari suatu film. Proyek ini dibuat untuk memenuhi Ujian Akhir Semester (UAS) pada mata kuliah Kecerdasan Buatan dan Sains Data.

Aplikasi ini menggunakan metode **klasifikasi** dalam *Machine Learning* (ML) dengan algoritma ***Logistic Regression*** untuk memprediksi apakah sebuah komentar memiliki sentimen **Positif** atau **Negatif**. Untuk mengubah komentar menjadi nilai numerik, aplikasi ini menggunakan teknik vektorisasi ***Term Frequency–Inverse Document Frequency* (TF-IDF)**.

## 👥 Anggota Kelompok
* **36240010** - Levingga Mettaliani
* **36240011** - Calsen Arlu
* **36240019** - Samuel Lie
* **36240021** - Cynthia Tipani Tio

## 📊 Tentang Proyek
Proyek ini dirancang untuk menyelesaikan permasalahan sulitnya mengetahui kualitas film secara cepat tanpa harus membaca ribuan ulasan satu per satu.
* **Dataset**: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Metode**: *Machine Learning* (NLP)
* **Algoritma**: *Logistic Regression*
* **Akurasi Model**: 88%

## 🚀 Cara Menjalankan Aplikasi (Lokal)

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi di komputer Anda menggunakan ***Virtual Environment***. Cara ini disarankan agar *library* proyek tidak tercampur dengan sistem utama komputer Anda.

### 1. *Clone* Repositori
Buka terminal (Git Bash / CMD / PowerShell), lalu arahkan ke folder tempat project ingin diunduh.
```bash
cd Downloads
```

Lakukan *clone* repositori dan tunggu hingga prosesnya berakhir.
```bash
git clone https://github.com/Lightzss/CineSense.git
```

Selanjutnya Anda perlu masuk ke folder repository.
```bash
cd CineSense
```

### 2. Setup *Virtual Environment* (Venv)
Buat lingkungan virtual Python (*Virtual Environment*) baru.

**Untuk Pengguna Windows:**
```bash
python -m venv venv
```

Aktifkan *virtual environment* yang sudah dibuat.
```bash
venv\Scripts\activate
```

**Untuk Pengguna macOS / Linux:**
```bash
python3 -m venv venv
```

Aktifkan *virtual environment* yang sudah dibuat.
```bash
source venv/bin/activate
```

Jika berhasil, akan muncul tanda (*venv*) di awal baris terminal Anda

### 3. *Install Library*
Setelah *virtual environment* aktif, *install* semua kebutuhan *library* yang ada di `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 4. Jalankan *Streamlit*
Jalankan aplikasi melalui perintah dibawah ini.
```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser Anda pada alamat http://localhost:8501.

## 📂 Struktur File
* `dataset/`: Data yang digunakan.
* `source/`: Berisi model (`.pkl`) yang sudah di-*training*.
* `app.py`: File aplikasi Streamlit.
* `movie-review-sentiment-analysis.ipynb`: File analisis data dan pembuatan model.
* `requirements.txt`: Berisi library yang digunakan untuk melakukan analisis, pembuatan model, dan menjalankan aplikasi.