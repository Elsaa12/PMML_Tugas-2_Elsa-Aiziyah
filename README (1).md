# Single Layer Perceptron (SLP) - Klasifikasi Iris Setosa vs Iris Versicolor

Implementasi **Single Layer Perceptron** dari nol untuk mengklasifikasikan dua spesies Iris
(**Iris-setosa** vs **Iris-versicolor**) menggunakan dataset UCI Iris.

Kode dalam notebook ini adalah **replikasi presisi** dari perhitungan manual di file Excel
`PMM-TemplateSLP_FILLED.xlsx` (sheet `Data`, `SLP-Training`, `SLP-Validation`, dan `Grafik`).
Seluruh nilai MSE dan akurasi cocok hingga 6 digit desimal.

---

## Daftar File

| File | Keterangan |
|---|---|
| `slp_iris.ipynb` | Notebook utama -- training, validasi, visualisasi |
| `PMM-TemplateSLP_FILLED.xlsx` | Referensi perhitungan Excel (4 sheet) |
| `Laporan_SLP_PMM.docx` | Laporan praktikum lengkap |
| `grafik_python.png` | Output grafik dari notebook |
| `grafik_excel_mse.png` | Grafik MSE dari Excel |
| `grafik_excel_akurasi.png` | Grafik akurasi dari Excel |

---

## Konfigurasi Model

| Parameter | Nilai | Lokasi di Excel |
|---|---|---|
| Learning rate | 0.1 | Sel K2 |
| Fungsi aktivasi | Sigmoid: g(z) = 1/(1+exp(-z)) | Kolom O |
| Fungsi error | SSE = (g(z) - y)^2 | Kolom R |
| Strategi update | Online SGD (per baris) | Kolom I-M baris n+1 |
| Jumlah epoch | 5 | Sel AB8 |
| Bobot awal | bias = t1 = t2 = t3 = t4 = 0.5 | Sel I5:M5 |
| Threshold prediksi | g(z) > 0.5 -> kelas 1 | Kolom P |

**Pembagian data:**
- Training : baris 1-40 (Iris-setosa) + 51-90 (Iris-versicolor) = **80 baris**
- Validasi : baris 41-50 (Iris-setosa) + 91-100 (Iris-versicolor) = **20 baris**

---

## Cara Menjalankan

### Prasyarat

```bash
pip install numpy openpyxl matplotlib pandas
```

### Langkah

1. Pastikan file `PMM-TemplateSLP_FILLED.xlsx` ada di direktori yang sama dengan notebook.
2. Buka `slp_iris.ipynb` di Jupyter Notebook / JupyterLab.
3. Jalankan semua sel dari atas ke bawah (**Run All**).

Jika nama file Excel berbeda, ubah variabel di sel 7:

```python
EXCEL_PATH = "nama_file_excel_kamu.xlsx"
```

---

## Arsitektur SLP

```
Input Layer             Neuron Tunggal          Output
-----------             --------------          ------
x1 (Sepal Length) --+
x2 (Sepal Width)  --+                           y_pred in {0, 1}
x3 (Petal Length) --+-->  z = bias + sum(wi*xi) --> g(z) --> pred
x4 (Petal Width)  --+
```

**Forward pass:**

```
z    = bias + t1*x1 + t2*x2 + t3*x3 + t4*x4
g(z) = 1 / (1 + exp(-z))
pred = 1 jika g(z) > 0.5, else 0
```

**Backward pass (Online SGD):**

```
d_bias = 2 * (g(z) - y) * g(z) * (1 - g(z))
d_ti   = d_bias * xi
w_baru = w_lama - lr * d_w
```

Bobot diupdate **setiap baris** dan **tidak direset** antar-epoch.

---

## Hasil Akhir

| Epoch | MSE Training | Akurasi Training | MSE Validasi | Akurasi Validasi |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.449889 | 52.50% | 0.328951 | 50.00% |
| 2 | 0.037452 | 95.00% | 0.247289 | 50.00% |
| 3 | 0.024372 | 97.50% | 0.175892 | 50.00% |
| 4 | 0.017357 | 97.50% | 0.119381 | 85.00% |
| 5 | 0.012740 | **98.75%** | 0.081581 | **100.00%** |

**Bobot akhir setelah 5 epoch:**

| bias | t1 (Sepal L.) | t2 (Sepal W.) | t3 (Petal L.) | t4 (Petal W.) |
|:---:|:---:|:---:|:---:|:---:|
| +0.2577 | -0.2418 | -0.4169 | **+1.1222** | +0.8241 |

`t3` (Petal Length) bernilai paling besar -- **panjang kelopak adalah fitur paling
diskriminatif** untuk memisahkan Iris-setosa dan Iris-versicolor.

---

## Struktur Notebook

| Bagian | Isi |
|---|---|
| 0. Import | Library yang digunakan |
| 1. Konfigurasi | `SLPConfig` dataclass -- hyperparameter |
| 2. Load Data | Membaca Excel, membangun set training & validasi |
| 3. Fungsi Inti | `sigmoid`, `forward_pass`, `compute_gradient`, `update_weights` |
| 4. Training Loop | `train_slp` -- online SGD + validasi per-epoch |
| 5. Prediksi | `predict` untuk sekumpulan data |
| 6. Visualisasi | `plot_training_curves` -- 2 grafik, 2 garis masing-masing |
| 7. Eksekusi | Jalankan pipeline penuh, tampilkan ringkasan epoch |
| 8. DataFrame | Tabel ringkasan MSE & akurasi dengan pandas |

---

## Mata Kuliah

> **Pembelajaran Mesin Modern (PMM)**
> Program Studi Magister Kecerdasan Artifisial
> Universitas Gadjah Mada -- 2024/2025
