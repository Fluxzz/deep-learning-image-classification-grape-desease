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
2.  **Stratified 5-Fold Cross-Validation**: Model dilatih menggunakan pembagian k-fold terlapis guna menjamin kestabilan akurasi di setiap subset data.
3.  **Ensemble Soft-Voting**: Menggabungkan hasil probabilitas (`softmax`) dari seluruh checkpoint model (Fold 1 s.d. 5) secara dinamis. Hasil rata-rata voting memberikan keputusan diagnosis yang jauh lebih stabil dibandingkan model tunggal.
4.  **Grad-CAM++ Explainable AI (XAI)**: Menyediakan visualisasi interpretasi keputusan model dengan cara merender peta panas (*heatmap*) daerah atensi di atas citra daun asli. Pengguna dapat melihat dengan tepat bagian mana dari daun yang memicu infeksi penyakit menurut pandangan kecerdasan buatan.
5.  **Pendeteksi Citra Asing (Out-of-Distribution Filter)**:
    *   Jika tingkat keyakinan model di bawah **60%**, sistem secara otomatis mengklasifikasikan input sebagai **`Citra Tidak Dikenal`** dan menampilkan peringatan keras (sangat berguna untuk menyaring gambar non-daun anggur, seperti foto orang memancing, benda mati, dll.).
    *   Jika keyakinan berada di kisaran **60% - 70%**, model memberikan peringatan keyakinan rendah agar pengguna memperhatikan kembali kualitas gambar atau sudut foto.

---

## 💻 Tampilan Antarmuka (UI/UX)
Antarmuka web dirancang dengan gaya **Slate & Indigo Light Theme** yang bersih, modern, dan profesional layaknya dashboard SaaS standar industri:
*   **Panel Kontrol Kiri**: Box drag-and-drop file interaktif dengan fitur hapus/reset instan tanpa memuat ulang halaman.
*   **Panel Analisis Kanan**: Dashboard diagnostik terpadu yang menampilkan:
    *   Kategori diagnosis utama lengkap dengan lencana fold kontributor.
    *   Skor kepercayaan berupa progress bar animasi.
    *   Visualisasi letak fokus deteksi Grad-CAM++.
    *   Grafik batang distribusi probabilitas Chart.js ter-kustomisasi warna per kategori penyakit.

---

## 🚀 Panduan Memulai

### Prasyarat
Pastikan komputer Anda sudah terinstal Python versi 3.8 ke atas.

### 1. Kloning Repository
```bash
git clone https://github.com/USERNAME_ANDA/REPOSTORY_ANDA.git
cd grapeleaf-ai-web
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
    docker run -p 5000:5000 grapeleaf-ai-web
    ```

Buka peramban/browser Anda dan akses alamat:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📂 Struktur Project
```text
my_dl_web/
├── templates/
│   └── index.html               # Halaman antarmuka dashboard
├── app.py                       # Server backend Flask (Inference & Grad-CAM++)
├── requirements.txt             # Daftar dependensi modul python
├── .gitignore                   # Konfigurasi pengabaian file Git
├── README.md                    # Dokumentasi utama project
├── best_mobilevit_fold1.pth     # Weights model fold 1 (dst.)
└── test_predict.py              # Skrip integrasi testing backend
```
