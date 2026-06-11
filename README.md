---
title: Grape Disease Classification
emoji: 🍇
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Grape Disease Classification

Web application for grape leaf disease classification using MobileViT.


# GrapeLeaf AI 🍇

Sistem Diagnosis Penyakit Daun Anggur Terpadu berbasis kecerdasan buatan (**Deep Learning**) menggunakan arsitektur **MobileViT-S** dengan evaluasi berlapis. 

Aplikasi web ini dapat mendeteksi secara *real-time* 4 kategori utama citra daun anggur:
*   **Black Rot** (Busuk Hitam)
*   **ESCA** (Black Measles / Mekaran Hitam)
*   **Leaf Blight** (Isariopsis Leaf Spot / Hawar Daun)
*   **Healthy** (Daun Sehat)

---

## 🌟 Fitur Unggulan & Arsitektur

1.  **Arsitektur MobileViT-S**: Mengintegrasikan kekuatan lokal CNN dengan kemampuan atensi global Vision Transformer (ViT) dalam ukuran yang sangat ringkas dan efisien.
2.  **Stratified 5-Fold Cross-Validation**: Model dilatih menggunakan pembagian k-fold terlapis guna menjamin kestabilan akurasi di setiap subset data. Setiap fold digunakan bergantian sebagai data validasi, sementara 4 fold lain menjadi data pelatihan, sehingga hasil evaluasi lebih tidak bias dan lebih representatif.
3.  **Ensemble Soft-Voting**: Menggabungkan hasil probabilitas (`softmax`) dari seluruh checkpoint model (Fold 1 s.d. 5) secara dinamis. Hasil rata-rata voting memberikan keputusan diagnosis yang jauh lebih stabil dibandingkan model tunggal.
4.  **Grad-CAM++ Explainable AI (XAI)**: Menyediakan visualisasi interpretasi keputusan model dengan cara merender peta panas (*heatmap*) daerah atensi di atas citra daun asli. Pengguna dapat melihat dengan tepat bagian mana dari daun yang memicu infeksi penyakit menurut pandangan kecerdasan buatan.
5.  **Pendeteksi Citra Asing (Out-of-Distribution Filter)**:
    *   Jika tingkat keyakinan model di bawah **60%**, sistem secara otomatis mengklasifikasikan input sebagai **`Citra Tidak Dikenal`** dan menampilkan peringatan keras.
    *   Jika keyakinan berada di kisaran **60% - 70%**, model memberikan peringatan keyakinan rendah agar pengguna memperhatikan kualitas gambar atau sudut foto.
    *   Implementasi OOD kini dipantau langsung pada endpoint `/predict` dan hasilnya dikembalikan sebagai bagian dari respons JSON.

---

## � Kerangka Evaluasi Model

Model ini dievaluasi menggunakan pendekatan berbasis **stratified 5-fold cross-validation**, yaitu:

- Data dibagi menjadi 5 subset (fold) dengan distribusi kelas yang seimbang.
- Pada setiap iterasi, 4 fold dipakai untuk training dan 1 fold untuk validation.
- Proses ini diulang 5 kali sehingga setiap fold pernah menjadi data validasi.
- Hasil akhir dievaluasi secara rata-rata untuk mendapatkan estimasi performa model yang lebih stabil.

Setelah proses pelatihan, hasil dari kelima checkpoint tersebut digabungkan menggunakan **ensemble soft-voting**. Artinya, probabilitas prediksi dari setiap fold dirata-ratakan sebelum keputusan akhir dibuat. Pendekatan ini membantu mengurangi variansi hasil prediksi antar fold dan meningkatkan keandalan model pada data baru.

---

## �💻 Tampilan Antarmuka (UI/UX)
Antarmuka web dirancang dengan gaya **Slate & Indigo Light Theme** yang bersih, modern, dan profesional layaknya dashboard SaaS standar industri:
*   **Panel Kontrol Kiri**: Tempat pengguna mengunggah gambar daun anggur melalui area drag-and-drop. Panel ini juga menyediakan tombol hapus/reset agar proses pengujian dapat diulang dengan cepat tanpa reload halaman.
*   **Panel Analisis Kanan**: Dashboard diagnostik terpadu yang menampilkan:
    *   Kategori diagnosis utama lengkap dengan lencana fold kontributor.
    *   Skor kepercayaan berupa progress bar animasi untuk menunjukkan tingkat keyakinan model.
    *   Visualisasi letak fokus deteksi Grad-CAM++ sebagai penjelasan area yang paling berpengaruh pada prediksi.
    *   Grafik batang distribusi probabilitas Chart.js yang menampilkan perbandingan peluang tiap kelas penyakit.

Secara singkat, panel kiri berfungsi untuk input gambar, sedangkan panel kanan berfungsi untuk menampilkan hasil prediksi dan penjelasan model secara visual.

---

## 🚀 Panduan Memulai

### Prasyarat
Pastikan komputer Anda sudah terinstal Python versi 3.10 ke atas.

### 1. Kloning Repository
```bash
git clone https://github.com/Fluxzz/deep-learning-image-classification-grape-desease.git
cd deep-learning-image-classification-grape-desease
```

### 2. Buat Virtual Environment (Opsional namun Direkomendasikan)
Menggunakan virtual env bawaan Python:
```bash
python -m venv venv
venv\Scripts\activate  # Untuk Windows
source venv/bin/activate  # Untuk Linux/macOS
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```
Dependensi sudah ditulis dengan versi minimal yang kompatibel untuk Flask, NumPy, PyTorch, TorchVision, timm, dan `requests` untuk uji endpoint.

### 4. Taruh File Weights Model
Pastikan berkas weights model `.pth` hasil training diletakkan di root folder project dengan nama:
- `best_mobilevit_fold1.pth`
- `best_mobilevit_fold2.pth`
- `best_mobilevit_fold3.pth`
- `best_mobilevit_fold4.pth`
- `best_mobilevit_fold5.pth`

*(Jika file model tidak diletakkan di sana, aplikasi otomatis beralih ke **Simulation Mode (Demo)** dengan aman).*

### 5. Jalankan Aplikasi

#### Cara A: Menggunakan Python Lokal
```bash
python app.py
```

#### Cara B: Menggunakan Docker
1.  **Build Docker Image**:
    ```bash
    docker build -t grapeleaf-ai-web .
    ```
2.  **Jalankan Docker Container**:
    ```bash
    docker run -p 7860:7860 grapeleaf-ai-web
    ```

Buka peramban/browser Anda dan akses alamat:
[http://127.0.0.1:7860](http://127.0.0.1:7860)

Untuk menguji endpoint prediksi dari terminal, jalankan:
```bash
python test_predict.py sample_test.png
```

---

## 📂 Struktur Project
```text
deep-learning-image-classification-grape-desease/
├── .venv/                       # Virtual environment Python (opsional)
├── templates/
│   └── index.html               # Halaman antarmuka dashboard web
├── app.py                       # Server backend Flask (inference, Grad-CAM++, dll.)
├── best_mobilevit_fold1.pth     # Weight model fold 1
├── best_mobilevit_fold2.pth     # Weight model fold 2
├── best_mobilevit_fold3.pth     # Weight model fold 3
├── best_mobilevit_fold4.pth     # Weight model fold 4
├── best_mobilevit_fold5.pth     # Weight model fold 5
├── Dockerfile                   # Konfigurasi container Docker
├── LICENSE                      # Lisensi proyek
├── README.md                    # Dokumentasi utama project
├── requirements.txt             # Daftar dependensi Python
├── test_predict.py              # Skrip uji prediksi / integrasi backend
├── sample_test.png              # Contoh gambar uji (opsional)
└── __notebook_source__.ipynb    # Notebook sumber eksplorasi/modeling
```


