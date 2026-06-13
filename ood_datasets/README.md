# 📂 Folder Dataset OOD (Out-of-Distribution)

Letakkan gambar atau folder berisi gambar **yang bukan daun anggur** di sini untuk menguji kemampuan OOD detection pada model GrapeLeaf AI.

## Cara Penggunaan

### 1. Masukkan Dataset OOD

Anda bisa memasukkan gambar dengan cara apapun:

```text
ood_datasets/
├── README.md              ← File ini
├── kucing/                ← Folder berisi gambar kucing
│   ├── cat1.jpg
│   └── cat2.png
├── bunga/                 ← Folder berisi gambar bunga
│   ├── flower1.jpg
│   └── flower2.jpg
├── random_image.jpg       ← Atau file gambar langsung
└── mobil.png              ← Gambar apapun yang bukan daun anggur
```

**Format yang didukung**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### 2. Jalankan Server Flask

```bash
python app.py
```

### 3. Jalankan Test OOD

```bash
python test_ood.py
```

### 4. Lihat Hasil

- Hasil ditampilkan di terminal
- Laporan detail disimpan di `ood_test_report.txt`

## Tips

- Gunakan gambar yang **beragam** (hewan, objek, pemandangan, dll.) untuk menguji robustness OOD filter
- Gambar daun dari tanaman **selain anggur** juga bagus untuk menguji apakah model bisa membedakannya
- Semakin banyak gambar, semakin representatif hasilnya
