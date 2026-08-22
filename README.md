# 📊 Sistem Analisis Sentimen - Panduan Penggunaan

## 🚀 Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

---

## 📋 Struktur Menu

| No | Menu | Fungsi |
|----|------|--------|
| 1 | 🏠 Beranda | Tampilan awal dan penjelasan alur sistem |
| 2 | 📂 Unggah Dataset | Upload CSV dataset + leksikon opsional |
| 3 | 🔍 Filter & Analisis | Pilih keyword/sumber → Mulai Analisis |

---

## 📂 Format Dataset (CSV)

Kolom **wajib**:
- `text` — Teks komentar
- `source` — Sumber data (youtube, twitter, portal berita, dll)
- `keyword` — Kata kunci

Kolom **opsional**:
- `author` — Nama pengguna
- `date` — Tanggal komentar

---

## ⚙️ Alur Preprocessing

1. **Case Folding** — Mengubah teks ke huruf kecil
2. **Cleaning** — Hapus URL, mention, hashtag, angka, karakter khusus
3. **Tokenisasi** — Memisahkan teks menjadi token kata (NLTK)
4. **Stopword Removal** — Hapus kata tidak bermakna (NLTK + manual)
5. **Normalisasi** — Perbaiki kata alay & singkatan
6. **Stemming** — Bentuk dasar kata (PySastrawi)
7. **Labeling** — Label sentimen via leksikon (positif/negatif/netral)
8. **SVM** — LinearSVC dengan TF-IDF Bigram, split 80:20

---

## 📚 Leksikon Opsional

Jika ingin menggunakan leksikon sendiri, upload file CSV dengan format:
```
word,score
bagus,3
jelek,-2
...
```
Jika tidak diupload, sistem menggunakan leksikon bawaan.

---

## 📊 Output Analisis

- **Tab Preprocessing** — Tabel setiap tahap preprocessing
- **Tab Labeling** — Data + label sentimen + skor
- **Tab Visualisasi** — Pie chart & bar chart distribusi sentimen per sumber
- **Tab WordCloud** — WordCloud semua + positif + negatif + netral
- **Tab Model SVM** — Akurasi, laporan klasifikasi, confusion matrix
