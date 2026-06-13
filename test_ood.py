"""
test_ood.py — Script Pengujian Out-of-Distribution (OOD) Detection
=================================================================

Menguji apakah model GrapeLeaf AI dapat mendeteksi gambar yang BUKAN daun anggur
(Out-of-Distribution) dengan benar.

Mendukung dua format dataset OOD:
  A. Folder berisi gambar (.jpg, .png, dll.) — langsung atau dalam subfolder
  B. CIFAR-100 pickle format (.zip berisi file pickle train/test/meta)

Cara Penggunaan:
    1. Letakkan gambar/folder/zip OOD di folder `ood_datasets/`
       ATAU taruh file CIFAR-100 .zip di root project
    2. Jalankan server Flask: `python app.py`
    3. Jalankan script ini: `python test_ood.py`

Opsi CLI:
    --folder <path>     Folder dataset OOD (default: ood_datasets/)
    --zip <path>        Path langsung ke file ZIP CIFAR-100
    --url <api_url>     URL endpoint predict (default: http://127.0.0.1:7860/predict)
    --threshold         Tampilkan informasi threshold yang digunakan di app.py
    --report <path>     Path file laporan (default: ood_test_report.txt)
    --max <N>           Maksimum gambar yang diuji (default: semua)
    --split <name>      Pilih split dataset: train, test, atau both (default: test)
"""

import os
import sys
import argparse
import time
import zipfile
import pickle
import io
from datetime import datetime

# Fix encoding untuk Windows console (agar emoji bisa ditampilkan)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import requests
except ImportError:
    print("Error: Module 'requests' belum terinstal.")
    print("Jalankan: pip install requests")
    sys.exit(1)

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Error: Module 'numpy' atau 'Pillow' belum terinstal.")
    print("Jalankan: pip install numpy Pillow")
    sys.exit(1)


# ============================================================
# Konfigurasi Default
# ============================================================
DEFAULT_OOD_FOLDER = "ood_datasets"
DEFAULT_API_URL = "http://127.0.0.1:7860/predict"
DEFAULT_REPORT_PATH = "ood_test_report.txt"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# CIFAR-100 Loader
# ============================================================

class CIFAR100Loader:
    """Memuat dan mengonversi dataset CIFAR-100 dari format pickle ZIP."""

    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.fine_label_names = []
        self.coarse_label_names = []
        self._loaded = False

    def is_cifar100_zip(self):
        """Cek apakah ZIP ini berisi dataset CIFAR-100."""
        try:
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                names = set(zf.namelist())
                # CIFAR-100 ZIP contains: meta, test, train, file.txt
                # At minimum needs meta and either train or test
                has_meta = "meta" in names
                has_data = "train" in names or "test" in names
                return has_meta and has_data
        except (zipfile.BadZipFile, Exception):
            return False

    def load_meta(self):
        """Muat metadata (label names)."""
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            meta_data = zf.read("meta")
            meta = pickle.loads(meta_data)
            self.fine_label_names = meta.get("fine_label_names", [])
            self.coarse_label_names = meta.get("coarse_label_names", [])
            self._loaded = True

    def load_split(self, split_name):
        """
        Muat data dari split tertentu (train atau test).
        Returns list of dict: {image: PIL.Image, fine_label: str, coarse_label: str, filename: str}
        """
        if not self._loaded:
            self.load_meta()

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            raw = zf.read(split_name)

        data = pickle.loads(raw, encoding="latin1")
        images_data = data["data"]  # shape (N, 3072), dtype uint8
        fine_labels = data["fine_labels"]
        coarse_labels = data["coarse_labels"]
        filenames = data["filenames"]

        n = len(fine_labels)
        results = []

        for i in range(n):
            # CIFAR-100: 3072 bytes = 3 channels × 32 × 32
            # Stored as [R...1024, G...1024, B...1024]
            img_flat = images_data[i]
            img_r = img_flat[0:1024].reshape(32, 32)
            img_g = img_flat[1024:2048].reshape(32, 32)
            img_b = img_flat[2048:3072].reshape(32, 32)
            img_np = np.stack([img_r, img_g, img_b], axis=-1).astype(np.uint8)

            pil_image = Image.fromarray(img_np, mode="RGB")

            fine_idx = fine_labels[i]
            coarse_idx = coarse_labels[i]
            fine_name = self.fine_label_names[fine_idx] if fine_idx < len(self.fine_label_names) else f"class_{fine_idx}"
            coarse_name = self.coarse_label_names[coarse_idx] if coarse_idx < len(self.coarse_label_names) else f"group_{coarse_idx}"

            results.append({
                "image": pil_image,
                "fine_label": fine_name,
                "coarse_label": coarse_name,
                "filename": filenames[i],
                "index": i,
            })

        return results

    def get_info(self):
        """Return ringkasan dataset."""
        if not self._loaded:
            self.load_meta()

        info = {
            "fine_labels": self.fine_label_names,
            "coarse_labels": self.coarse_label_names,
            "splits": [],
        }

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            for name in ["train", "test"]:
                if name in zf.namelist():
                    info["splits"].append(name)

        return info


# ============================================================
# Utilitas Gambar Folder
# ============================================================

def find_images(folder_path):
    """Scan folder secara rekursif dan temukan semua file gambar."""
    images = []
    for root, _dirs, files in os.walk(folder_path):
        rel_root = os.path.relpath(root, folder_path)
        if "__MACOSX" in rel_root or any(
            part.startswith(".") for part in rel_root.split(os.sep) if part != "."
        ):
            continue
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                images.append(os.path.join(root, fname))
    return images


def extract_zip_files(folder_path):
    """Cari dan ekstrak file .zip biasa (berisi gambar) di folder."""
    extracted = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".zip"):
            zip_path = os.path.join(folder_path, fname)

            # Skip jika ini CIFAR-100 format
            loader = CIFAR100Loader(zip_path)
            if loader.is_cifar100_zip():
                continue

            extract_dir = os.path.join(folder_path, os.path.splitext(fname)[0])
            if os.path.isdir(extract_dir):
                print(f"  📂 Folder '{os.path.splitext(fname)[0]}/' sudah ada, skip ekstraksi {fname}")
                extracted.append(zip_path)
                continue

            print(f"  📦 Mengekstrak {fname} → {os.path.splitext(fname)[0]}/")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
                extracted.append(zip_path)
                images_count = len(find_images(extract_dir))
                print(f"     ✅ Berhasil: {images_count} gambar ditemukan")
            except zipfile.BadZipFile:
                print(f"     ❌ Error: {fname} bukan file ZIP yang valid")
            except Exception as e:
                print(f"     ❌ Error mengekstrak {fname}: {e}")
    return extracted


def get_subfolder(image_path, base_folder):
    """Dapatkan nama subfolder (kelas OOD) dari path gambar."""
    rel = os.path.relpath(image_path, base_folder)
    parts = rel.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "(root)"


def send_predict_file(image_path, api_url):
    """Kirim file gambar ke endpoint /predict."""
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(api_url, files=files, timeout=120)
    return response.status_code, response.json()


def send_predict_pil(pil_image, filename, api_url):
    """Kirim PIL Image ke endpoint /predict."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": (filename, buf, "image/png")}
    response = requests.post(api_url, files=files, timeout=120)
    return response.status_code, response.json()


def format_duration(seconds):
    """Format durasi dalam format yang mudah dibaca."""
    if seconds < 60:
        return f"{seconds:.1f} detik"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes} menit {secs:.1f} detik"


# ============================================================
# Kelas Hasil
# ============================================================

class OODTestResult:
    """Menyimpan hasil pengujian per-gambar."""

    def __init__(self, identifier, status_code, response_data, category="(root)", error=None):
        self.identifier = identifier  # filename or path
        self.category = category      # subfolder or coarse_label
        self.status_code = status_code
        self.error = error

        if error:
            self.success = False
            self.is_ood = False
            self.is_low_confidence = False
            self.confidence_score = 0.0
            self.confidence = "N/A"
            self.predicted_class = "ERROR"
        else:
            self.success = response_data.get("success", False)
            self.is_ood = response_data.get("is_ood", False)
            self.is_low_confidence = response_data.get("is_low_confidence", False)
            self.confidence_score = response_data.get("confidence_score", 0.0)
            self.confidence = response_data.get("confidence", "N/A")
            self.predicted_class = response_data.get("class", "N/A")

    @property
    def detected_as_ood(self):
        return self.is_ood

    @property
    def flagged(self):
        return self.is_ood or self.is_low_confidence

    @property
    def status_icon(self):
        if self.error:
            return "💥"
        if self.is_ood:
            return "✅"
        if self.is_low_confidence:
            return "⚠️"
        return "❌"

    @property
    def status_label(self):
        if self.error:
            return "ERROR"
        if self.is_ood:
            return "OOD TERDETEKSI"
        if self.is_low_confidence:
            return "LOW CONFIDENCE"
        return "TIDAK TERDETEKSI"


# ============================================================
# Runner Utama
# ============================================================

class OODTestRunner:
    """Runner utama untuk batch testing OOD."""

    def __init__(self, api_url, report_path):
        self.api_url = api_url
        self.report_path = report_path
        self.results = []
        self.start_time = None
        self.end_time = None
        self.source_description = ""

    def check_server(self):
        """Cek koneksi ke server Flask."""
        try:
            status_url = self.api_url.replace("/predict", "/status")
            resp = requests.get(status_url, timeout=10)
            if resp.status_code != 200:
                raise ConnectionError(f"Status code: {resp.status_code}")
            status_data = resp.json()
            model_loaded = status_data.get("model_loaded", False)
            device = status_data.get("device", "unknown")
            folds = status_data.get("total_folds_loaded", 0)

            print(f"  🔗 Server terhubung: {self.api_url}")
            if model_loaded:
                print(f"     ✅ Model aktif: {folds} fold(s) pada {device.upper()}")
            else:
                print(f"     ⚠️  Mode Simulasi (model tidak dimuat)")
            return True

        except requests.exceptions.ConnectionError:
            print(f"\n  ❌ Gagal terhubung ke server: {self.api_url}")
            print(f"     Pastikan server Flask berjalan: python app.py")
            return False
        except Exception as e:
            print(f"\n  ❌ Error saat cek status server: {e}")
            return False

    def run_from_folder(self, folder_path):
        """Jalankan tes dari folder berisi gambar."""
        # Auto-extract ZIP biasa (non-CIFAR)
        zip_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".zip")]
        if zip_files:
            print(f"\n  📦 Ditemukan {len(zip_files)} file ZIP, mengekstrak otomatis...")
            extract_zip_files(folder_path)

        images = find_images(folder_path)
        if not images:
            print(f"\n  ❌ Tidak ada gambar ditemukan di '{folder_path}'!")
            print(f"     Format yang didukung: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            return False

        subfolders = set(get_subfolder(img, folder_path) for img in images)
        self.source_description = f"Folder: {os.path.abspath(folder_path)}"

        self._print_header(len(images), f"{len(subfolders)} subfolder ({', '.join(sorted(subfolders))})")

        self.start_time = time.time()
        self.results = []

        current_sub = None
        for idx, img_path in enumerate(images, 1):
            subfolder = get_subfolder(img_path, folder_path)
            rel_path = os.path.relpath(img_path, folder_path)

            if subfolder != current_sub:
                current_sub = subfolder
                print(f"\n  ── 📁 {subfolder} ──")

            try:
                status_code, data = send_predict_file(img_path, self.api_url)
                result = OODTestResult(rel_path, status_code, data, category=subfolder)
            except Exception as e:
                result = OODTestResult(rel_path, 0, {}, category=subfolder, error=str(e))

            self.results.append(result)
            self._print_progress(idx, len(images), result)

        self.end_time = time.time()
        return True

    def run_from_cifar100(self, zip_path, split="test", max_images=None):
        """Jalankan tes dari CIFAR-100 pickle ZIP."""
        loader = CIFAR100Loader(zip_path)

        if not loader.is_cifar100_zip():
            print(f"\n  ❌ '{zip_path}' bukan file CIFAR-100 yang valid!")
            return False

        info = loader.get_info()
        splits_to_test = []
        if split == "both":
            splits_to_test = [s for s in ["test", "train"] if s in info["splits"]]
        else:
            if split in info["splits"]:
                splits_to_test = [split]
            else:
                print(f"\n  ❌ Split '{split}' tidak ditemukan. Tersedia: {info['splits']}")
                return False

        self.source_description = f"CIFAR-100: {os.path.abspath(zip_path)}"

        # Load semua data yang akan diuji
        all_items = []
        for s in splits_to_test:
            print(f"  📂 Memuat split '{s}'...")
            items = loader.load_split(s)
            for item in items:
                item["split"] = s
            all_items.extend(items)
            print(f"     ✅ {len(items)} gambar dimuat")

        if max_images and max_images < len(all_items):
            # Sample secara merata dari semua coarse labels
            import random
            random.seed(42)
            random.shuffle(all_items)
            all_items = all_items[:max_images]
            print(f"\n  🎯 Dibatasi ke {max_images} gambar (dari {len(all_items)} total)")

        total = len(all_items)
        categories = set(item["coarse_label"] for item in all_items)
        cat_desc = f"{len(categories)} kategori coarse ({len(info['fine_labels'])} fine labels)"

        self._print_header(total, cat_desc)

        self.start_time = time.time()
        self.results = []

        # Sort by coarse_label agar tampilan terkelompok
        all_items.sort(key=lambda x: (x["coarse_label"], x["fine_label"]))

        current_cat = None
        for idx, item in enumerate(all_items, 1):
            coarse = item["coarse_label"]
            fine = item["fine_label"]
            fname = item["filename"]
            identifier = f"[{fine}] {fname}"

            if coarse != current_cat:
                current_cat = coarse
                print(f"\n  ── 📁 {coarse} ──")

            try:
                status_code, data = send_predict_pil(item["image"], fname, self.api_url)
                result = OODTestResult(identifier, status_code, data, category=coarse)
            except Exception as e:
                result = OODTestResult(identifier, 0, {}, category=coarse, error=str(e))

            self.results.append(result)
            self._print_progress(idx, total, result)

        self.end_time = time.time()
        return True

    def _print_header(self, total_images, categories_desc):
        print(f"\n{'='*65}")
        print(f"  🧪 PENGUJIAN OOD DETECTION — GrapeLeaf AI")
        print(f"{'='*65}")
        print(f"  📂 Sumber    : {self.source_description}")
        print(f"  🖼️  Gambar    : {total_images} file")
        print(f"  📁 Kategori  : {categories_desc}")
        print(f"  🌐 Endpoint  : {self.api_url}")
        print(f"  📝 Laporan   : {self.report_path}")
        print(f"{'='*65}")

    def _print_progress(self, idx, total, result):
        progress = f"[{idx}/{total}]"
        icon = result.status_icon
        conf = result.confidence if result.confidence != "N/A" else "-"
        print(f"  {progress} {icon} {result.identifier}")
        print(f"         Kelas: {result.predicted_class} | Confidence: {conf} | {result.status_label}")

    # ── Ringkasan & Laporan ──

    def get_summary(self):
        total = len(self.results)
        ood_detected = sum(1 for r in self.results if r.detected_as_ood)
        low_conf = sum(1 for r in self.results if r.is_low_confidence and not r.is_ood)
        not_detected = sum(1 for r in self.results if not r.flagged and not r.error)
        errors = sum(1 for r in self.results if r.error)
        flagged = sum(1 for r in self.results if r.flagged)
        valid = total - errors
        ood_rate = (ood_detected / valid * 100) if valid > 0 else 0.0
        flagged_rate = (flagged / valid * 100) if valid > 0 else 0.0
        duration = (self.end_time - self.start_time) if self.end_time else 0
        return {
            "total": total, "valid": valid, "ood_detected": ood_detected,
            "low_confidence": low_conf, "not_detected": not_detected,
            "errors": errors, "flagged": flagged,
            "ood_rate": ood_rate, "flagged_rate": flagged_rate,
            "duration": duration,
        }

    def get_category_summary(self):
        groups = {}
        for r in self.results:
            cat = r.category
            if cat not in groups:
                groups[cat] = {"total": 0, "ood": 0, "low_conf": 0, "not_detected": 0, "errors": 0}
            groups[cat]["total"] += 1
            if r.error:
                groups[cat]["errors"] += 1
            elif r.detected_as_ood:
                groups[cat]["ood"] += 1
            elif r.is_low_confidence:
                groups[cat]["low_conf"] += 1
            else:
                groups[cat]["not_detected"] += 1
        return groups

    def print_summary(self):
        s = self.get_summary()
        print(f"\n{'='*65}")
        print(f"  📊 RINGKASAN HASIL PENGUJIAN OOD")
        print(f"{'='*65}")
        print(f"  Total gambar diuji     : {s['total']}")
        print(f"  Gambar valid diproses  : {s['valid']}")
        print(f"  Error                  : {s['errors']}")
        print(f"{'─'*65}")
        print(f"  ✅ OOD Terdeteksi       : {s['ood_detected']} ({s['ood_rate']:.1f}%)")
        print(f"  ⚠️  Low Confidence       : {s['low_confidence']}")
        print(f"  ❌ Tidak Terdeteksi     : {s['not_detected']}")
        print(f"{'─'*65}")
        print(f"  📈 OOD Detection Rate  : {s['ood_rate']:.1f}%")
        print(f"  📈 Total Flagged Rate  : {s['flagged_rate']:.1f}% (OOD + Low Conf)")
        print(f"  ⏱️  Durasi total        : {format_duration(s['duration'])}")
        print(f"{'='*65}")

        # Per-category summary
        cat_summary = self.get_category_summary()
        if len(cat_summary) > 1 or (len(cat_summary) == 1 and "(root)" not in cat_summary):
            print(f"\n{'─'*65}")
            print(f"  📁 RINGKASAN PER KATEGORI OOD")
            print(f"{'─'*65}")
            print(f"  {'Kategori':<30} {'Tot':<5} {'OOD':<5} {'Low':<5} {'Miss':<5} {'Rate'}")
            print(f"  {'─'*30} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*6}")

            for name in sorted(cat_summary.keys()):
                c = cat_summary[name]
                c_valid = c["total"] - c["errors"]
                c_rate = (c["ood"] / c_valid * 100) if c_valid > 0 else 0.0
                display = name[:28] + ".." if len(name) > 30 else name
                print(f"  {display:<30} {c['total']:<5} {c['ood']:<5} {c['low_conf']:<5} {c['not_detected']:<5} {c_rate:.0f}%")
            print(f"{'─'*65}")

        # Penilaian
        if s['valid'] == 0:
            print(f"\n  ⚠️  Tidak ada gambar valid yang diproses.")
        elif s['ood_rate'] >= 90:
            print(f"\n  🎉 EXCELLENT! OOD filter bekerja sangat baik.")
        elif s['ood_rate'] >= 70:
            print(f"\n  👍 GOOD. OOD filter bekerja cukup baik, namun beberapa gambar lolos.")
        elif s['ood_rate'] >= 50:
            print(f"\n  ⚠️  FAIR. Pertimbangkan untuk menaikkan OOD_THRESHOLD di app.py.")
        else:
            print(f"\n  🚨 POOR. OOD filter kurang efektif. Periksa threshold dan model.")

        # False negatives
        not_detected = [r for r in self.results if not r.flagged and not r.error]
        if not_detected:
            print(f"\n  ⚠️  Gambar TIDAK terdeteksi sebagai OOD (False Negatives): {len(not_detected)} gambar")
            # Show max 20
            shown = not_detected[:20]
            for r in shown:
                print(f"     • [{r.category}] {r.identifier} → {r.predicted_class} ({r.confidence})")
            if len(not_detected) > 20:
                print(f"     ... dan {len(not_detected) - 20} lainnya (lihat laporan lengkap)")

        print()

    def save_report(self):
        s = self.get_summary()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append("=" * 75)
        lines.append("  LAPORAN PENGUJIAN OOD DETECTION — GrapeLeaf AI")
        lines.append("=" * 75)
        lines.append(f"  Waktu          : {timestamp}")
        lines.append(f"  Sumber         : {self.source_description}")
        lines.append(f"  Endpoint       : {self.api_url}")
        lines.append(f"  Durasi         : {format_duration(s['duration'])}")
        lines.append("")
        lines.append("-" * 75)
        lines.append("  RINGKASAN")
        lines.append("-" * 75)
        lines.append(f"  Total gambar           : {s['total']}")
        lines.append(f"  Valid diproses         : {s['valid']}")
        lines.append(f"  Error                  : {s['errors']}")
        lines.append(f"  OOD Terdeteksi         : {s['ood_detected']} ({s['ood_rate']:.1f}%)")
        lines.append(f"  Low Confidence         : {s['low_confidence']}")
        lines.append(f"  Tidak Terdeteksi       : {s['not_detected']}")
        lines.append(f"  OOD Detection Rate     : {s['ood_rate']:.1f}%")
        lines.append(f"  Total Flagged Rate     : {s['flagged_rate']:.1f}%")
        lines.append("")

        # Per-category
        cat_summary = self.get_category_summary()
        if len(cat_summary) > 1 or (len(cat_summary) == 1 and "(root)" not in cat_summary):
            lines.append("-" * 75)
            lines.append("  RINGKASAN PER KATEGORI OOD")
            lines.append("-" * 75)
            lines.append(f"  {'Kategori':<30} {'Tot':<5} {'OOD':<5} {'Low':<5} {'Miss':<5} {'Rate'}")
            lines.append(f"  {'─'*30} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*6}")

            for name in sorted(cat_summary.keys()):
                c = cat_summary[name]
                c_valid = c["total"] - c["errors"]
                c_rate = (c["ood"] / c_valid * 100) if c_valid > 0 else 0.0
                display = name[:28] + ".." if len(name) > 30 else name
                lines.append(f"  {display:<30} {c['total']:<5} {c['ood']:<5} {c['low_conf']:<5} {c['not_detected']:<5} {c_rate:.0f}%")
            lines.append("")

        # Detail per gambar
        lines.append("-" * 75)
        lines.append("  DETAIL PER-GAMBAR")
        lines.append("-" * 75)
        lines.append(f"  {'No.':<6} {'St':<4} {'Kategori':<20} {'File':<25} {'Kelas':<20} {'Conf':<8} {'Label'}")
        lines.append(f"  {'─'*6} {'─'*4} {'─'*20} {'─'*25} {'─'*20} {'─'*8} {'─'*18}")

        for idx, r in enumerate(self.results, 1):
            icon = r.status_icon
            cat = r.category[:18] + ".." if len(r.category) > 20 else r.category
            ident = r.identifier[:23] + ".." if len(r.identifier) > 25 else r.identifier
            cls = r.predicted_class[:18] + ".." if len(r.predicted_class) > 20 else r.predicted_class
            lines.append(
                f"  {idx:<6} {icon:<4} {cat:<20} {ident:<25} {cls:<20} {r.confidence:<8} {r.status_label}"
            )

        lines.append("")

        # False negatives
        not_detected = [r for r in self.results if not r.flagged and not r.error]
        if not_detected:
            lines.append("-" * 75)
            lines.append(f"  GAMBAR TIDAK TERDETEKSI — FALSE NEGATIVES ({len(not_detected)})")
            lines.append("-" * 75)
            for r in not_detected:
                lines.append(f"  • [{r.category}] {r.identifier}")
                lines.append(f"    Kelas: {r.predicted_class} | Confidence: {r.confidence} (score: {r.confidence_score:.4f})")
            lines.append("")

        lines.append("=" * 75)
        lines.append("  Akhir Laporan")
        lines.append("=" * 75)

        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  📄 Laporan disimpan ke: {os.path.abspath(self.report_path)}")


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pengujian OOD Detection untuk GrapeLeaf AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python test_ood.py                                   # Dari folder ood_datasets/
  python test_ood.py --zip ood_datasets.zip             # Dari CIFAR-100 ZIP langsung
  python test_ood.py --zip ood_datasets.zip --max 100   # Batasi 100 gambar
  python test_ood.py --zip ood_datasets.zip --split both --max 500
  python test_ood.py --threshold                       # Lihat threshold saat ini
        """,
    )
    parser.add_argument("--folder", type=str, default=DEFAULT_OOD_FOLDER,
                        help=f"Path ke folder dataset OOD (default: {DEFAULT_OOD_FOLDER})")
    parser.add_argument("--zip", type=str, default=None,
                        help="Path langsung ke file ZIP CIFAR-100")
    parser.add_argument("--url", type=str, default=DEFAULT_API_URL,
                        help=f"URL endpoint /predict (default: {DEFAULT_API_URL})")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_PATH,
                        help=f"Path file output laporan (default: {DEFAULT_REPORT_PATH})")
    parser.add_argument("--threshold", action="store_true",
                        help="Tampilkan informasi threshold OOD yang digunakan di app.py")
    parser.add_argument("--max", type=int, default=None, dest="max_images",
                        help="Maksimum gambar yang diuji (default: semua)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "both"],
                        help="Split dataset CIFAR-100 yang diuji (default: test)")
    return parser.parse_args()


def show_threshold_info():
    print(f"\n{'='*65}")
    print(f"  📐 KONFIGURASI THRESHOLD OOD (dari app.py)")
    print(f"{'='*65}")

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    ood_thresh = None
    low_conf_thresh = None

    if os.path.exists(app_path):
        with open(app_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("OOD_THRESHOLD"):
                    try:
                        ood_thresh = float(stripped.split("=")[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif stripped.startswith("LOW_CONFIDENCE_THRESHOLD"):
                    try:
                        low_conf_thresh = float(stripped.split("=")[1].strip())
                    except (ValueError, IndexError):
                        pass

    if ood_thresh is not None:
        print(f"  OOD_THRESHOLD            : {ood_thresh} ({ood_thresh*100:.0f}%)")
        print(f"    → Confidence < {ood_thresh*100:.0f}% = 'Citra Tidak Dikenal'")
    else:
        print(f"  OOD_THRESHOLD            : Tidak ditemukan di app.py")

    if low_conf_thresh is not None:
        ood_pct = ood_thresh * 100 if ood_thresh else 65
        print(f"  LOW_CONFIDENCE_THRESHOLD : {low_conf_thresh} ({low_conf_thresh*100:.0f}%)")
        print(f"    → Confidence {ood_pct:.0f}%-{low_conf_thresh*100:.0f}% = peringatan 'kepercayaan rendah'")
    else:
        print(f"  LOW_CONFIDENCE_THRESHOLD : Tidak ditemukan di app.py")

    print(f"\n  💡 Untuk mengubah threshold, edit variabel di app.py lalu restart server.")
    print(f"{'='*65}\n")


def detect_cifar100_zip(folder_path):
    """Cari file CIFAR-100 ZIP di folder atau parent directory."""
    # Cek di folder itu sendiri
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".zip"):
            zip_path = os.path.join(folder_path, fname)
            loader = CIFAR100Loader(zip_path)
            if loader.is_cifar100_zip():
                return zip_path

    # Cek di parent directory (root project)
    parent = os.path.dirname(os.path.abspath(folder_path))
    for fname in os.listdir(parent):
        if fname.lower().endswith(".zip") and "ood" in fname.lower():
            zip_path = os.path.join(parent, fname)
            loader = CIFAR100Loader(zip_path)
            if loader.is_cifar100_zip():
                return zip_path

    return None


def main():
    args = parse_args()

    print(f"\n  🍇 GrapeLeaf AI — OOD Detection Test Suite")
    print(f"  {'─'*45}")

    if args.threshold:
        show_threshold_info()

    runner = OODTestRunner(api_url=args.url, report_path=args.report)

    # Cek koneksi server dulu
    if not runner.check_server():
        sys.exit(1)

    # Tentukan mode: ZIP langsung atau folder
    if args.zip:
        # Mode: ZIP CIFAR-100 langsung
        if not os.path.isfile(args.zip):
            print(f"\n  ❌ File ZIP tidak ditemukan: {args.zip}")
            sys.exit(1)

        loader = CIFAR100Loader(args.zip)
        if loader.is_cifar100_zip():
            print(f"\n  📦 Terdeteksi: Dataset CIFAR-100 (pickle format)")
            info = loader.get_info()
            print(f"     Fine labels  : {len(info['fine_labels'])} kelas")
            print(f"     Coarse labels: {len(info['coarse_labels'])} kategori")
            print(f"     Splits       : {', '.join(info['splits'])}")

            success = runner.run_from_cifar100(args.zip, split=args.split, max_images=args.max_images)
        else:
            print(f"\n  ❌ File '{args.zip}' bukan CIFAR-100 format yang valid!")
            sys.exit(1)
    else:
        # Mode: Folder
        if not os.path.isdir(args.folder):
            print(f"\n  ❌ Folder '{args.folder}' tidak ditemukan!")
            sys.exit(1)

        # Auto-detect CIFAR-100 ZIP
        cifar_zip = detect_cifar100_zip(args.folder)
        if cifar_zip:
            print(f"\n  📦 Terdeteksi: Dataset CIFAR-100 di {os.path.basename(cifar_zip)}")
            loader = CIFAR100Loader(cifar_zip)
            info = loader.get_info()
            print(f"     Fine labels  : {len(info['fine_labels'])} kelas")
            print(f"     Coarse labels: {len(info['coarse_labels'])} kategori")
            print(f"     Splits       : {', '.join(info['splits'])}")

            success = runner.run_from_cifar100(cifar_zip, split=args.split, max_images=args.max_images)
        else:
            # Folder biasa berisi gambar
            images = find_images(args.folder)
            if not images:
                print(f"\n  ❌ Tidak ada gambar atau dataset CIFAR-100 ditemukan di '{args.folder}'!")
                print(f"     Format gambar: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
                print(f"     Atau gunakan --zip <path> untuk file CIFAR-100")
                sys.exit(1)

            success = runner.run_from_folder(args.folder)

    if success:
        runner.print_summary()
        runner.save_report()


if __name__ == "__main__":
    main()
