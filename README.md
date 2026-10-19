# Article Named-Entity Recognition (NER) Service

Protipe layanan ekstraksi entitas bernama (*Named-Entity Recognition*) berbasis web yang dirancang untuk tim pemantau media berita. Layanan ini menerima judul (*title*) dan isi artikel (*body*), menggabungkannya secara kontekstual, mengekstraksi entitas dengan model spaCy yang telah disesuaikan, serta menyajikan hasilnya melalui REST API dan panel antarmuka pengguna interaktif.

---

## Daftar Isi
1. [Arsitektur & Komponen Sistem](#1-arsitektur--komponen-sistem)
2. [Eksplorasi Data & Strategi Pembagian Data](#2-eksplorasi-data--strategi-pembagian-data)
3. [Panduan Instalasi & Pengoperasian](#3-panduan-instalasi--pengoperasian)
4. [Spesifikasi Teknis API](#4-spesifikasi-teknis-api)
5. [Pengujian Otomatis](#5-pengujian-otomatis)
6. [Evaluasi Model & Analisis Galat](#6-evaluasi-model--analisis-galat)
7. [Jawaban Pertanyaan Studi Kasus](#7-jawaban-pertanyaan-studi-kasus)
8. [Batasan Sistem](#8-batasan-sistem)

---

## 1. Arsitektur & Komponen Sistem

Sistem ini dibangun dengan arsitektur monolitik ringan yang memisahkan lapisan presentasi, logika backend, dan inferensi NLP:

```text
+-------------------------------------------------------+
|                 Web Browser Panel                     |
|           (src/index.html - HTML5 / Vanilla JS)       |
+---------------------------+---------------------------+
                            |
                     HTTP POST JSON
                            v
+-------------------------------------------------------+
|                 FastAPI Application                   |
|                   (src/main.py)                       |
|  - Request Validation (Empty string / Max length / JSON) |
|  - Title & Body Contextual Concatenation              |
|  - Error Handling & Status Mapping (HTTP 400 / 200)   |
+---------------------------+---------------------------+
                            |
                 Token & Entity Extraction
                            v
+-------------------------------------------------------+
|                 spaCy NER Pipeline                    |
|      (Baseline: en_core_web_sm / Adapted: custom)      |
|  - Tokenizer, Tagger, Parser, NER                     |
|  - Span Offsets Calculation (start_char, end_char)    |
+-------------------------------------------------------+
```

### Struktur Direktori Proyek
```text
ner-project/
├── data/
│   ├── evaluation_examples.json    # 10 artikel evaluasi beranotasi rentang karakter
│   ├── test.csv                   # Data uji (15% split)
│   ├── train.csv                  # Data latih (70% split)
│   └── val.csv                    # Data validasi (15% split)
├── screenshots/
│   ├── evaluation_metrics.png     # Bukti eksekusi evaluasi metrik
│   ├── pytest_results.png         # Bukti 5 pengujian otomatis lolos
│   └── web_panel.png              # Bukti visual antarmuka web panel
├── src/
│   ├── evaluate.py                # Skrip evaluasi Precision, Recall, F1
│   ├── index.html                 # Antarmuka panel pengguna
│   ├── main.py                    # Server FastAPI & endpoint API
│   ├── prepare_eval.py            # Pembantu pembuatan data/evaluation_examples.json
│   ├── split_data.py              # Skrip pembagian data deterministik
│   └── train.py                   # Adaptasi/fine-tuning pipeline spaCy
├── tests/
│   ├── __init__.py
│   └── test_api.py                # 5 skenario automated tests pytest
├── profile_data.py                # Skrip profiling data awal
├── requirements.txt               # Daftar dependensi lingkungan Python
└── README.md                      # Dokumentasi proyek lengkap
```

---

## 2. Eksplorasi Data & Strategi Pembagian Data

### Profiling Data Mentah
Berdasarkan pembacaan 8 file CSV di `use_cases/pnm_study_case/sample/` menggunakan pemisah titik koma (`;`), diperoleh karakteristik data:
- **Total Baris**: 1.291 baris.
- **Baris Duplikat**: 0 baris.
- **Missing Values**: 0 nilai kosong pada kolom `title` dan `body`.
- **Distribusi Bahasa**:
  - Bahasa Inggris (`en`): 1.059 artikel (~82.0%)
  - Bahasa Melayu (`ms`): 178 artikel (~13.8%)
  - Bahasa Mandarin (`zh`): 37 artikel (~2.9%)
  - Bahasa Indonesia (`id`): 15 artikel (~1.1%)
  - Bahasa Vietnam (`vi`): 1 artikel
  - Bahasa Spanyol (`es`): 1 artikel
- **Status Kolom `entities`**: Berisi list kosong `[]`. Sesuai instruksi, kolom ini tidak dijadikan *ground truth*, melainkan set evaluasi dibuat secara independen dengan validasi span karakter.

### Strategi Pembagian Data (*Data Splitting*)
Karena korpus didominasi oleh bahasa Inggris, pemodelan difokuskan pada subset bahasa Inggris (`en`) sebanyak 1.059 artikel (terdiri atas 656 ID artikel unik):
- **Pembagian**: 70% data latih, 15% data validasi, 15% data uji (*held-out test*).
- **Integritas Pemisahan**: Pemisahan dilakukan berdasarkan `original_id` artikel (bukan per baris atau kalimat) untuk memastikan teks dari artikel yang sama tidak bocor (*data leakage*) ke partisi lain.
- **Reproduksibilitas**: Menggunakan random seed tetap (`seed=42`).
  - **Train split**: 749 artikel
  - **Val split**: 150 artikel
  - **Test split**: 160 artikel

---

## 3. Panduan Instalasi & Pengoperasian

### Prasyarat
- Python 3.10 atau versi di atasnya
- PowerShell / Command Prompt / Terminal Bash

### 1. Kloning Repositori & Persiapan Virtual Environment
```bash
# Masuk ke direktori proyek
cd ner-project

# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Aktivasi virtual environment (Linux/macOS)
source venv/bin/activate
```

### 2. Instalasi Dependensi
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Menjalankan Adaptasi Model (Opsional / Reproduksi Bobot)
```bash
python src/train.py
```
Perintah ini akan membaca data pelatihan dan mengekspor model teradaptasi ke folder `custom_ner_model`.

### 4. Menjalankan Layanan Web & API
```bash
uvicorn src.main:app --reload
```
Setelah server aktif:
- Buka panel web di: `http://127.0.0.1:8000/`
- Dokumentasi interaktif Swagger/OpenAPI: `http://127.0.0.1:8000/docs`

---

## 4. Spesifikasi Teknis API

Layanan menyediakan satu endpoint utama untuk ekstraksi entitas berita.

### Endpoint: `POST /extract-entities`

#### Request Header
```http
Content-Type: application/json
```

#### Request Body
```json
{
  "title": "Microsoft Expands Data Center in London",
  "body": "Satya Nadella announced Microsoft is investing heavily in London."
}
```

#### Validasi Masukan:
- `title` dan `body` wajib berupa string dan tidak boleh kosong atau hanya berisi spasi.
- Panjang total teks gabungan (`title + "\n\n" + body`) dibatasi maksimal 15.000 karakter.
- Permintaan yang melanggar aturan di atas atau format JSON yang rusak akan menghasilkan respons galat `HTTP 400 Bad Request`.

#### Response Sukses (`HTTP 200 OK`)
```json
{
  "entities": [
    {
      "text": "Microsoft",
      "label": "ORG",
      "start": 0,
      "end": 9
    },
    {
      "text": "London",
      "label": "GPE",
      "start": 35,
      "end": 41
    },
    {
      "text": "Satya Nadella",
      "label": "PERSON",
      "start": 43,
      "end": 56
    },
    {
      "text": "Microsoft",
      "label": "ORG",
      "start": 67,
      "end": 76
    },
    {
      "text": "London",
      "label": "GPE",
      "start": 100,
      "end": 106
    }
  ],
  "meta": {
    "entity_count": 5,
    "model": "en_core_web_sm"
  }
}
```

#### Response Galat (`HTTP 400 Bad Request`)
```json
{
  "detail": "Title and body must not be empty."
}
```

---

## 5. Pengujian Otomatis

Pengujian dilakukan menggunakan framework `pytest` dan `TestClient` bawaan FastAPI pada file `tests/test_api.py`. Seluruh pengujian mencakup 5 skenario wajib:
1. `test_extract_entities_normal`: Memverifikasi ekstraksi teks normal, kelengkapan atribut span, dan status HTTP 200.
2. `test_extract_entities_empty_input`: Memverifikasi validasi input kosong menghasilkan HTTP 400.
3. `test_extract_entities_punctuation_and_non_ascii`: Memverifikasi ketahanan model terhadap simbol, aksen tanda baca, dan karakter non-ASCII (misal `€`, `é`, `ü`).
4. `test_extract_entities_repeated`: Memverifikasi kemampuan sistem mendeteksi entitas identik yang muncul berulang kali di berbagai posisi teks.
5. `test_extract_entities_exceeds_max_length`: Memverifikasi penolakan teks yang melampaui batas 15.000 karakter dengan kode HTTP 400.

### Menjalankan Pengujian:
```bash
python -m pytest -v
```
**Hasil Pengujian:**
```text
tests/test_api.py::test_extract_entities_normal PASSED                    [ 20%]
tests/test_api.py::test_extract_entities_empty_input PASSED              [ 40%]
tests/test_api.py::test_extract_entities_punctuation_and_non_ascii PASSED [ 60%]
tests/test_api.py::test_extract_entities_repeated PASSED                  [ 80%]
tests/test_api.py::test_extract_entities_exceeds_max_length PASSED        [100%]
============================== 5 passed in 2.04s ==============================
```

---

## 6. Evaluasi Model & Analisis Galat

Evaluasi dijalankan pada 10 artikel uji beranotasi acuan (`data/evaluation_examples.json`) yang diambil dari split uji independen. Pencocokan metrik dilakukan dengan metode **Exact-Span Matching** (rentang karakter `start`, `end`, dan `label` harus cocok sempurna).

### Menjalankan Evaluasi:
```bash
python src/evaluate.py
```

### Hasil Perbandingan Metrik

| Model | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`en_core_web_sm`)** | 65 | 17 | 0 | **0.7927** | **1.0000** | **0.8844** |
| **Adapted (`custom_ner_model`)** | 53 | 67 | 12 | **0.4417** | **0.8154** | **0.5730** |

### Analisis Galat (*Error Analysis*)
1. **False Positive Angka/Kuantitas pada Baseline**:
   - *Kasus*: Baseline mendeteksi token `'Two'` dan `'5'` sebagai entitas `CARDINAL`.
   - *Penyebab*: Kamus acuan evaluasi hanya memfokuskan entitas nama diri (`PERSON`, `ORG`, `GPE`, `LOC`), sehingga entitas numerik yang sah menurut OntoNotes 5 terhitung sebagai *False Positive*.
2. **Penurunan Performa & Catastrophic Forgetting pada Model Teradaptasi**:
   - *Kasus*: Model teradaptasi salah mengklasifikasikan `'Deepavali Day'` sebagai label `PERSON` (bukan perayaan/peristiwa) serta melewatkan beberapa entitas lokasi umum.
   - *Penyebab*: Adaptasi dilakukan hanya pada beberapa iterasi contoh spesifik berita tanpa regularisasi korpus umum yang memadai, mengakibatkan bobot model mengalami pergeseran representasi (*catastrophic forgetting*).
3. **Ambiguitas Entitas Majemuk**:
   - *Kasus*: Model mendeteksi `'Terengganu Hockey Team'` sebagai entitas utuh `ORG`.
   - *Penyebab*: Terdapat perbedaan antara anotasi parsial kata dasar dan frasa majemuk nama organisasi, yang menunjukkan perlunya panduan anotasi (*annotation guideline*) yang lebih baku dalam mendefinisikan batas nama organisasi bergelar/bertipe.

---

## 7. Jawaban Pertanyaan Studi Kasus

### 1. What makes a text span a named entity in this problem?
Dalam konteks ini, suatu rentang teks (*span*) didefinisikan sebagai entitas bernama jika merujuk secara kaku (*rigid designator*) pada suatu objek atau entitas spesifik di dunia nyata. Entitas ini mencakup nama individu orang (`PERSON`), nama organisasi atau institusi bisnis/pemerintah (`ORG`), entitas geopolitik atau batas wilayah administratif (`GPE`), lokasi fisik alam atau fasilitas (`LOC`), serta penanda tanggal absolut (`DATE`). Entitas dibedakan dari kata benda umum (*common nouns*) karena merujuk pada identitas unik.

### 2. Why should title and body be combined before analysis? What information could be lost if they are processed independently?
Judul berita dirancang ringkas dan kerap menghilangkan kata sambung serta konteks gramatikal formal (misalnya: *"Apple Expands UK Site"*). Sebaliknya, isi artikel memuat penjelasan detail, penyebutan nama lengkap (*full names*), afiliasi, dan deskripsi institusi. Memproses keduanya secara bersamaan mempertahankan konteks wacana (*discourse context*), memungkinkan model mengaitkan entitas di judul dengan konteks di isi berita, dan memastikan penomoran offset karakter mencerminkan struktur dokumen yang utuh.

### 3. What are the supported labels of the chosen model, and how do they map to the categories shown to users?
Model spaCy (`en_core_web_sm`) mendukung label standar OntoNotes 5 (`PERSON`, `NORP`, `FAC`, `ORG`, `GPE`, `LOC`, `PRODUCT`, `EVENT`, `WORK_OF_ART`, `LAW`, `LANGUAGE`, `DATE`, `TIME`, `PERCENT`, `MONEY`, `QUANTITY`, `ORDINAL`, `CARDINAL`).
Untuk memudahkan pengguna tim pemantau media, label tersebut disederhanakan pada tabel antarmuka:
- `PERSON` $\rightarrow$ Tokoh / Figur Publik
- `ORG` $\rightarrow$ Korporasi / Lembaga / Asosiasi
- `GPE` & `LOC` $\rightarrow$ Lokasi Wilayah / Geografis
- `DATE` $\rightarrow$ Penanda Waktu / Tanggal Kejadian

### 4. How does tokenization, punctuation, capitalization, or language affect the result?
- **Kapitalisasi**: Model statistik spaCy sangat mengandalkan huruf kapital sebagai penanda awal nama diri. Judul berita dengan format huruf besar semua (*ALL CAPS*) atau huruf kecil semua sering menurunkan akurasi model.
- **Tanda Baca**: Klitis dan tanda petik (seperti pada *"Tokyo's"* atau *"Co."*) memerlukan tokenizer berbasis aturan yang tepat agar tanda baca tidak disatukan secara keliru ke dalam nama entitas.
- **Bahasa**: Model yang dilatih pada struktur gramatikal bahasa Inggris tidak memiliki kapabilitas memadai saat memproses teks bahasa lain (misalnya bahasa Indonesia atau Melayu), karena perbedaan kosakata, pola frasa (*noun phrases*), dan konjugasi kata kerja.

### 5. What is the difference between an entity's text, label, and character offsets?
- `text`: Substring teks aktual yang ditemukan dalam dokumen (contoh: `"Ada Lovelace"`).
- `label`: Kategori semantik dari entitas tersebut (contoh: `PERSON`).
- `character offsets` (`start` dan `end`): Indeks posisi karakter berbasis angka nol (0-based) yang menunjukkan letak awal dan akhir eksklusif span dalam string artikel utuh. Indeks ini penting untuk visualisasi penyorotan teks (*highlighting*) di frontend tanpa risiko tertukar dengan kata serupa di bagian lain.

### 6. What happens when the same entity appears multiple times with different spelling or capitalization?
Model statistik mengevaluasi setiap span berdasarkan konteks kalimat sekitarnya. Variasi ejaan (misal: *"UN"* vs *"United Nations"*) atau variasi kapitalisasi (misal: *"apple"* vs *"Apple"*) dapat menyebabkan model mendeteksi sebagian kemunculan namun melewatkan kemunculan lainnya, atau menetapkan label yang tidak konsisten. Untuk mengatasi hal ini, sistem memerlukan modul normalisasi teks atau *entity resolution* tambahan.

### 7. Which errors are most important for a news-monitoring user, and how could the system be improved?
Bagi tim pemantau berita, **False Negatives** (kegagalan mendeteksi nama tokoh atau organisasi yang menjadi subjek pemantauan) adalah galat paling kritis karena berita penting bisa terlewat dari pelacakan.
Sistem dapat ditingkatkan dengan:
- Menerapkan kamus entitas dinamis (*gazetteer*) untuk nama-nama tokoh industri baru.
- Menggunakan arsitektur Transformer (seperti RoBERTa atau DeBERTa) untuk menangkap konteks semantik yang lebih dalam.
- Menerapkan teknik *Coreference Resolution* untuk memetakan rujukan kata ganti ke entitas utama.

### 8. What privacy, copyright, bias, or responsible-use concerns apply to processing article text?
- **Hak Cipta**: Artikel media dilindungi oleh undang-undang hak cipta. Penyimpanan dan pelatihan data wajib mematuhi batasan *fair use* dan lisensi penyedia data.
- **Privasi**: Sistem berpotensi mengekstraksi nama individu privat yang kebetulan disebut dalam berita kriminal atau sensitif tanpa persetujuan eksplisit.
- **Bias Algoritma**: Model pretrained bahasa Inggris memiliki representasi yang bias terhadap entitas dari negara-negara Barat dan sering mengalami penurunan akurasi pada nama-nama lokal Asia/Indonesia.

---

## 8. Batasan Sistem

1. **Batas Ukuran Dokumen**: Teks masukan dibatasi maksimal 15.000 karakter per permintaan untuk mencegah pemborosan memori.
2. **Keterbatasan Multibahasa**: Pipeline inferensi saat ini dioptimalkan untuk korpus berbahasa Inggris. Artikel berbahasa non-Inggris belum diproses menggunakan model multibahasa khusus.
3. **Ketiadaan Entity Linking**: Sistem mendeteksi entitas secara permukaan (*surface-level span*) dan belum menautkan entitas ke basis data pengetahuan seperti Wikidata.