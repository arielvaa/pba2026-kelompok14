# 🧠 Fine-Grained Emotion Classification (20 Classes)

## 🚀 Live Demo
**Try the deployed model:** [Emotion Classification App on Hugging Face Spaces](https://huggingface.co/spaces/Aryamudas/pba2026-kelompok14)

---

## 📌 Deskripsi Proyek
Proyek ini bertujuan membangun sistem klasifikasi teks yang mampu mengidentifikasi emosi dalam kalimat berbahasa Inggris secara lebih detail. Dataset yang digunakan mencakup 20 kategori emosi, sehingga model tidak hanya membedakan sentimen positif dan negatif, tetapi juga mengenali emosi spesifik seperti marah, sedih, cemas, dan lainnya.

Selain itu, proyek ini membandingkan dua pendekatan utama dalam Natural Language Processing (NLP), yaitu Machine Learning (ML) dan Deep Learning (DL), untuk menentukan metode yang paling efektif dalam memahami emosi pada teks.

---

## 👥 Anggota Kelompok 14

| Nama                        | NIM        | GitHub                          |
|-----------------------------|-----------|----------------------------------|
| Arya Muda Siregar          | 123450063 | @Aryamuda                        |
| Arielva Simon Siahaan      | 123450105 | @arielvaa                        |
| Haikal Fransisko Simbolon  | 123450106 | @Haikal-Fransisko-Simbolon       |

---

## 📊 Dataset
Dataset yang digunakan adalah **20-Emotion Text Classification Dataset** yang berisi 79.595 kalimat dengan 20 label emosi yang berbeda.

- Total data: 79,595
- Jumlah kelas: 20 emosi
- Bahasa: Inggris
- Task: Multi-class text classification

Contoh label emosi:
- happiness, anger, sadness, fear, love, surprise, dan lainnya.

---

## ⚙️ Metodologi

### 1. Machine Learning (ML)
Pendekatan ML menggunakan scikit-learn dengan representasi fitur berbasis TF-IDF. Model yang telah dibenchmark:
- **Logistic Regression** - Baseline model
- **Naive Bayes (MultinomialNB)** - Specialized for text classification
- **Support Vector Machine (LinearSVC)** -  Best model (selected)

Model terbaik telah dipilih berdasarkan F1-Score dan di-deploy ke Hugging Face Spaces.

### 2. Deep Learning (DL)
Pendekatan DL akan menggunakan PyTorch dengan arsitektur seperti:
- LSTM
- Transformer sederhana

Model akan dilatih untuk memahami pola emosional dalam teks dengan tetap memperhatikan batasan jumlah parameter.

---

## 🎯 Tujuan
Tujuan utama dari proyek ini adalah:
- Membandingkan performa Machine Learning dan Deep Learning dalam klasifikasi emosi
- Mengidentifikasi model terbaik untuk menangani klasifikasi emosi dengan banyak kelas
- Memberikan insight terhadap efektivitas masing-masing pendekatan dalam memahami teks
