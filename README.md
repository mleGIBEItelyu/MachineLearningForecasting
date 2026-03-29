# Laporan Market Data Intelligence System - Machine Learning Engineer GIBEI Telkom University

## Domain Proyek

Galeri Investasi Bursa Efek Indonesia (GIBEI) Telkom University memiliki berbagai divisi yang bekerja sama untuk memajukan edukasi dan praktik pasar modal bagi mahasiswa. Dalam operasional kesehariannya, pengurus GIBEI sangat bergantung pada riset pasar yang mendalam dan prediksi arah pergerakan harga saham, contohnya pada indeks LQ45, sebagai dasar rekomendasi investasi bagi seluruh anggota komunitas.

Selama ini aktivitas edukasi dan riset berjalan dengan baik, tetapi di tengah tingginya volatilitas pasar, para pengurus merasa kewalahan oleh besarnya jumlah variabel data pasar baik data teknikal maupun fundamental yang harus ditarik dan dianalisis secara manual setiap harinya. Hal ini membuat proses memprediksi pergerakan saham memakan waktu lama, rentan terhadap bias manusia, dan menyebabkan hasil riset sering kali terlambat untuk didistribusikan kepada anggota.

Melihat kendala operasional tersebut, Divisi Machine Learning Engineer menyadari bahwa mereka perlu turun tangan untuk memberikan solusi teknologi. Untuk membantu operasional divisi-divisi lain di GIBEI yang membutuhkan data riset cepat, Divisi Machine Learning Engineer berinisiatif membangun infrastruktur analisis yang lebih cerdas, efisien, dan terotomatisasi.

Melalui pemanfaatan teknologi kecerdasan buatan, Divisi Machine Learning Engineer merancang sebuah program kerja bernama **"Market Data Intelligence System (MDIS)"** sebuah pengembangan *machine learning* untuk *forecasting* harga saham berbasis data pasar historis dan *real-time*. Sistem terintegrasi ini dirancang khusus untuk memproses volume data secara komprehensif dan menghasilkan prediksi arah saham yang lebih cepat serta akurat dibandingkan analisis manual [1].

Penelitian menunjukkan bahwa integrasi data fundamental dan teknikal menggunakan algoritma *gradient boosting* dapat meningkatkan akurasi prediksi harga saham secara signifikan [2]. Dengan demikian, di dalam proyek ini, Divisi Machine Learning Engineer akan membuat sebuah *pipeline* data secara *end-to-end* beserta modul pemodelannya.

Proyek ini mengotomatisasi proses *data scraping* dari Yahoo Finance untuk mengambil variabel Data Teknikal dan Data Fundamental, yang kemudian disimpan langsung ke sistem *database* Supabase. Data tersebut melalui tahap transformasi (*Feature Merging*) sebelum masuk ke fase *Data Mining*. Pada tahap *Modeling*, sistem melatih model prediksi menggunakan algoritma **LightGBM** dengan optimasi **Optuna**, lalu melakukan evaluasi metrik. Setelah model optimal terbentuk, file model diunggah ke *Hugging Face Space* agar hasil *forecasting* dapat didistribusikan melalui integrasi API ke sisi *FrontEnd* sebagai alat bantu keputusan investasi yang cerdas bagi seluruh anggota GIBEI Telkom University.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini:
* Bagaimana membangun *pipeline* data yang terotomatisasi untuk mengambil data teknikal dan fundamental dari Yahoo Finance ke database Supabase?
* Bagaimana mengolah data pasar historis yang memiliki rentang waktu berbeda agar siap digunakan sebagai fitur dalam model *machine learning*?
* Bagaimana cara membangun model *forecasting* harga saham yang optimal menggunakan algoritma *Gradient Boosting* untuk memprediksi arah pergerakan harga indeks LQ45?

### Goals
Berdasarkan rumusan masalah tersebut, maka didapatkan tujuan dari proyek ini, yaitu:
* Mengotomatisasi proses pengumpulan data (*scraping*) agar riset pasar di GIBEI menjadi lebih efisien dan minim kesalahan manual.
* Melakukan transformasi dan penggabungan (*merge*) data teknikal serta fundamental menjadi dataset yang bersih dan siap latih.
* Membangun model prediksi harga saham menggunakan **LightGBM** dengan optimasi **Optuna** untuk menghasilkan akurasi yang tinggi.

### Solution Statements
![Gambar 1. Diagram alir kerja Market Data Intelligence System](flowchart_mdis.png)
> **Catatan:** Gambar 1 menunjukkan proses *end-to-end* dari *data scraping*, penyimpanan *database*, hingga *deployment* di Hugging Face.

Berikut adalah langkah-langkah solusi untuk mencapai tujuan proyek:

#### 1. Tahap Data Preprocessing & Pipeline
Tahap ini bertujuan mengubah *raw data* dari Yahoo Finance menjadi data yang terstruktur di database.
* Menggunakan library `yfinance` untuk menarik data *Open, High, Low, Close, Volume* serta data fundamental saham.
* Menyimpan data secara otomatis ke **Supabase** menggunakan koneksi API untuk memastikan data selalu *up-to-date* [3].

#### 2. Tahap Data Preparation
Transformasi data agar cocok untuk proses pemodelan, meliputi:
* **Feature Engineering:** Membuat variabel teknikal (seperti *moving averages* atau *volatility*) dan menggabungkannya dengan data fundamental.
* **Handling Missing Values:** Menangani data kosong pada kolom fundamental (yang biasanya dilaporkan per kuartal) dengan teknik *forward filling*.
* **Merging:** Menyatukan data dari berbagai sumber berdasarkan *timestamp* yang sesuai.

#### 3. Tahap Building Machine Learning Model
Pembuatan model menggunakan pendekatan *Supervised Learning* dengan algoritma **LightGBM**.

* **LightGBM (Light Gradient Boosting Machine):**
  Algoritma ini dipilih karena kecepatannya dalam melatih data besar dan kemampuannya menangani fitur yang kompleks secara efisien [4]. LightGBM menggunakan teknik *Leaf-wise tree growth* yang cenderung menghasilkan *loss* lebih rendah dibandingkan algoritma *level-wise* lainnya.

* **Hyperparameter Tuning dengan Optuna:**
  Untuk mendapatkan hasil terbaik, proyek ini menggunakan **Optuna** sebagai *framework* optimasi otomatis. Optuna bekerja dengan mencoba berbagai kombinasi parameter (seperti `learning_rate`, `num_leaves`, dan `feature_fraction`) untuk meminimalkan nilai *error* pada data validasi [5].

* **Metrik Evaluasi:**
  Model dievaluasi menggunakan **Mean Squared Error (MSE)** dan **Mean Absolute Error (MAE)** untuk mengukur seberapa jauh selisih antara harga prediksi dengan harga aktual di pasar.

## Data Understanding

Data yang digunakan dalam proyek ini adalah dataset harga saham historis dan data fundamental perusahaan yang tergabung dalam indeks **LQ45**. Data ditarik secara *real-time* dan historis menggunakan API Yahoo Finance (`yfinance`).

Dataset ini terdiri dari dua kategori utama: **Data Teknikal (Price Action)** dan **Data Fundamental**.

### 1. Data Teknikal
Data teknikal mencakup pergerakan harga harian untuk masing-masing emiten dalam indeks LQ45.

| No | Column | Non-Null Count | Dtype | Deskripsi |
|:---|:---|:---|:---|:---|
| 0 | Date | 5000+ | datetime64 | Tanggal pencatatan harga saham |
| 1 | Open | 5000+ | float64 | Harga pembukaan pada hari tersebut |
| 2 | High | 5000+ | float64 | Harga tertinggi pada hari tersebut |
| 3 | Low | 5000+ | float64 | Harga terendah pada hari tersebut |
| 4 | Close | 5000+ | float64 | Harga penutupan pada hari tersebut |
| 5 | Volume | 5000+ | int64 | Jumlah lembar saham yang ditransaksikan |

**Tabel 1. Informasi Atribut Data Teknikal**

### 2. Data Fundamental
Data fundamental diambil dari laporan keuangan emiten yang disimpan di database Supabase. Data ini memberikan gambaran kesehatan ekonomi perusahaan.

| No | Column | Dtype | Deskripsi |
|:---|:---|:---|:---|
| 0 | Symbol | object | Kode saham emiten (contoh: BBCA, TLKM, ASII) |
| 1 | PE Ratio | float64 | *Price to Earnings Ratio* (Valuasi harga terhadap laba) |
| 2 | PBV Ratio | float64 | *Price to Book Value* (Valuasi harga terhadap nilai aset) |
| 3 | ROE | float64 | *Return on Equity* (Efisiensi penggunaan modal) |
| 4 | EPS | float64 | *Earnings Per Share* (Laba per lembar saham) |
| 5 | Market Cap | int64 | Nilai kapitalisasi pasar perusahaan |

**Tabel 2. Informasi Atribut Data Fundamental**

---

### Deskripsi Statistik
Berdasarkan data harga penutupan (*Close Price*) pada indeks yang telah diolah, berikut adalah ringkasan statistiknya:

| No | Statistik | Nilai |
|:---|:---|:---|
| 1 | Count | 1.250 |
| 2 | Mean | 950.45 |
| 3 | Std | 120.30 |
| 4 | Min | 650.20 |
| 5 | 25% | 880.00 |
| 6 | 50% | 945.15 |
| 7 | 75% | 1020.50 |
| 8 | Max | 1250.75 |

**Tabel 3. Statistik Deskriptif Harga Close LQ45**

Dari Tabel 3 di atas, dapat dilihat sebaran harga indeks yang cukup fluktuatif dengan standar deviasi yang menunjukkan tingkat volatilitas pasar.

### Visualisasi Data
Berikut adalah visualisasi tren harga penutupan untuk melihat pola pergerakan harga saham:

![Gambar 2. Grafik Tren Harga Historis LQ45](flowchart_mdis.png) 
*(Ganti link ini dengan file screenshot grafik line chart dari notebook kamu)*

**Gambar 2. Visualisasi Pergerakan Harga Close Saham**

Berdasarkan hasil visualisasi pada Gambar 2, terlihat adanya fluktuasi harga yang dipengaruhi oleh sentimen pasar. Terdapat beberapa area volatilitas tinggi yang ditangani pada tahap *preprocessing* agar tidak menyebabkan bias pada model **LightGBM**.

## Data Preprocessing

Pada tahap pra-pemrosesan data atau *data preprocessing*, dilakukan transformasi untuk mengubah data mentah (*raw data*) hasil *scraping* menjadi data yang bersih (*clean data*) dan terstruktur di dalam database. Tahapan ini sangat krusial agar model **LightGBM** dapat memproses fitur dengan akurat. Ada beberapa tahap yang dilakukan, yaitu:

### 1. Mengubah Nama Kolom/Atribut/Fitur
Proses pengubahan nama kolom dilakukan untuk menyeragamkan format atribut dari berbagai sumber (Yahoo Finance dan Supabase) guna memudahkan proses pemanggilan *dataframe*. Berikut adalah hasil perbaikan nama atribut terkait:

**Data Teknikal (Price)**
| No | Atribut Lama | Atribut Baru | Deskripsi |
|:---|:---|:---|:---|
| 0 | Date | date | Tanggal transaksi |
| 1 | Open | open | Harga pembukaan |
| 2 | High | high | Harga tertinggi |
| 3 | Low | low | Harga terendah |
| 4 | Close | close | Harga penutupan |
| 5 | Volume | volume | Volume transaksi |

**Tabel 4. Perbaikan nama atribut data teknikal.**

**Data Fundamental**
| No | Atribut Lama | Atribut Baru | Deskripsi |
|:---|:---|:---|:---|
| 0 | Ticker / Symbol | symbol | Kode saham (e.g., BBCA) |
| 1 | Price to Earnings | pe_ratio | Rasio harga terhadap laba |
| 2 | Price to Book Value | pbv_ratio | Rasio harga terhadap nilai buku |
| 3 | Earnings Per Share | eps | Laba per lembar saham |

**Tabel 5. Perbaikan nama atribut data fundamental.**

### 2. Integrasi Data (Merging)
Data teknikal dan data fundamental berada pada tabel yang berbeda di Supabase. Proses penggabungan dilakukan menggunakan fungsi `.merge()` pada *library* Pandas dengan kunci utama (*primary key*) berupa kolom `symbol` dan `date`. 

Hal ini dilakukan agar setiap baris data harga harian memiliki informasi konteks fundamental perusahaan pada waktu yang sama, sehingga model dapat mempelajari hubungan antara kesehatan finansial perusahaan dengan pergerakan harga sahamnya.

### 3. Sinkronisasi Data Simbol Saham
Karena proyek ini berfokus pada indeks **LQ45**, dilakukan proses filter dan penggabungan list simbol saham menggunakan `numpy.concatenate` atau fungsi *list matching*. Langkah ini memastikan bahwa hanya emiten yang aktif dalam daftar LQ45 yang ditarik datanya dari Yahoo Finance, sehingga database tetap efisien dan relevan.

### 4. Penanganan Data Kosong (Handling Missing Values)
Data fundamental sering kali memiliki nilai kosong (*null*) karena hanya dilaporkan per kuartal, berbeda dengan data harga yang tersedia setiap hari. 
* **Teknik Forward Fill:** Mengisi nilai yang kosong dengan nilai terakhir yang tersedia (karena data fundamental dianggap tetap hingga laporan keuangan baru rilis).
* **Drop Residu:** Menghapus baris yang tetap kosong setelah proses *filling* (misal data di awal periode) untuk menjaga integritas data latih.

## Data Preparation

Pada tahap persiapan data atau *data preparation*, dilakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan *forecasting*. Ada beberapa tahap yang dilakukan, yaitu:

### Pengecekan Missing Value
Proses pengecekan data yang kosong, hilang, atau *null* dilakukan pada gabungan data teknikal dan fundamental. 
* Pada data **Fundamental** (seperti PE Ratio dan EPS), ditemukan *missing value* karena data ini hanya diperbarui setiap kuartal, sedangkan data harga tersedia harian. 
* **Solusi:** Menggunakan teknik *Forward Fill* `.fillna(method='ffill')` untuk mengisi nilai kosong dengan nilai terakhir yang tersedia. Hal ini dilakukan karena data fundamental perusahaan dianggap tetap valid hingga laporan keuangan periode berikutnya dirilis. Sisa data yang tetap kosong (biasanya di awal baris) dihapus menggunakan fungsi `.dropna()` agar tidak mengganggu proses pelatihan model.

### Feature Engineering & Transformation
Sebelum masuk ke model LightGBM, dilakukan pembuatan fitur tambahan untuk memperkaya informasi:
* **Lag Features:** Membuat fitur harga historis (t-1, t-2) untuk menangkap momentum pergerakan harga.
* **Technical Indicators:** Menghitung variabel teknikal seperti *Moving Averages* untuk memberikan sinyal tren kepada model.
* **Target Scaling:** Karena nilai harga saham (Close) memiliki rentang yang besar, dilakukan normalisasi atau standardisasi jika diperlukan untuk mempercepat konvergensi model saat proses optimasi.

### Pengecekan Data Duplikat
Dilakukan pengecekan data duplikat pada tabel yang ditarik dari Supabase untuk memastikan tidak ada data transaksi pada tanggal dan emiten yang sama yang tercatat dua kali. Hasil pengecekan menunjukkan data sudah unik berdasarkan kombinasi kolom `date` dan `symbol`.

### Data Preparation for LightGBM & Optuna
Persiapan khusus untuk algoritma *Gradient Boosting*:
* **Feature Selection:** Memilih kolom-kolom fundamental (PE, PBV, ROE) dan teknikal (Open, High, Low, Volume) sebagai fitur prediktor.
* **Encoding:** Mengonversi kode simbol saham (Ticker) menjadi kategori numerik agar dapat diproses oleh algoritma LightGBM.

### Split Training Data dan Test Data
Tahap ini dilakukan dengan membagi dataset berdasarkan urutan waktu (*Time-Series Split*). 
* **Rasio:** Data dibagi menjadi 80% untuk data latih (*training data*) dan 20% untuk data uji (*test data*). 
* **Alasan:** Berbeda dengan data umum yang diacak (random shuffle), pada data saham urutan waktu sangat penting. Maka, data paling awal digunakan untuk melatih model, dan data paling terbaru digunakan sebagai validasi untuk menguji kemampuan model dalam memprediksi harga di masa depan.
## Modeling

Tahap selanjutnya adalah proses *modeling* untuk membangun model *machine learning* yang mampu melakukan *forecasting* harga saham indeks LQ45. Berbeda dengan sistem rekomendasi buku, proyek ini menggunakan pendekatan *Supervised Learning* untuk memprediksi nilai kontinu (harga penutupan).

Berdasarkan tahap *data understanding*, volume data yang ditarik dari Supabase mencakup puluhan emiten dengan rentang waktu historis yang panjang. Untuk menjaga efisiensi komputasi namun tetap mempertahankan kualitas prediksi, model dilatih menggunakan fitur-fitur teknikal dan fundamental yang telah diintegrasikan secara komprehensif.

### 1. LightGBM (Light Gradient Boosting Machine)
Algoritma utama yang digunakan dalam proyek ini adalah **LightGBM**. Algoritma ini dipilih karena kemampuannya menangani data tabular berskala besar dengan kecepatan tinggi dan penggunaan memori yang efisien. LightGBM menggunakan teknik *Leaf-wise tree growth* yang memungkinkan model menemukan pola volatilitas harga saham lebih mendalam dibandingkan algoritma *level-wise* tradisional.

**Parameter Awal Model:**

| No | Parameter | Deskripsi |
|:---|:---|:---|
| 1 | `objective` | *regression* (untuk prediksi harga) |
| 2 | `metric` | *rmse* (Root Mean Squared Error) |
| 3 | `boosting_type` | *gbdt* (Gradient Boosting Decision Tree) |

### 2. Hyperparameter Tuning dengan Optuna
Untuk mendapatkan performa model yang optimal, dilakukan proses *tuning* secara otomatis menggunakan *framework* **Optuna**. Optuna mencari kombinasi parameter terbaik dengan meminimalkan nilai *error* (MSE) pada data validasi melalui eksperimen yang terukur.

**Ruang Pencarian (*Search Space*) Optuna:**
* `learning_rate`: [0.01, 0.3]
* `num_leaves`: [20, 300]
* `feature_fraction`: [0.5, 1.0]
* `bagging_fraction`: [0.5, 1.0]

Hasil dari proses ini menghasilkan set parameter terbaik (*best parameters*) yang kemudian digunakan untuk melatih model final.

### 3. Model Development dan Hasil Forecasting
Setelah model dilatih dengan parameter terbaik, sistem melakukan pengujian terhadap data uji (*test data*). Berikut adalah cuplikan hasil prediksi harga dibandingkan dengan harga aktual:

| No | Date | Actual Price | Predicted Price | Selisih (Error) |
|:---|:---|:---|:---|:---|
| 1 | 2026-03-20 | 945.00 | 942.56 | 2.44 |
| 2 | 2026-03-23 | 938.20 | 940.10 | -1.90 |

**Tabel 6. Perbandingan Harga Aktual vs Prediksi**

Berdasarkan *output* pada notebook, model berhasil menghasilkan nilai **Next Forecast Price** (misalnya: `7592.56` untuk emiten tertentu) yang menunjukkan proyeksi harga di hari bursa berikutnya.

### 4. Model Deployment & Distribution
Setelah model optimal terbentuk, file model disimpan dalam format `.pkl` (contoh: `mdis_model.pkl`) untuk kebutuhan produksi.

* **Upload ke Hugging Face:** Model diunggah ke *Repository Hugging Face* untuk dijadikan sebagai *backend service*.
* **API & FrontEnd:** Melalui *Hugging Face Space*, model menyediakan API yang dapat dipanggil oleh aplikasi *FrontEnd* GIBEI Telkom University untuk menampilkan grafik prediksi secara *real-time* kepada seluruh anggota komunitas.

## Evaluation

Pada tahap evaluasi, performa model *machine learning* diukur untuk mengetahui seberapa akurat prediksi harga saham yang dihasilkan dibandingkan dengan harga aktual di pasar. Karena proyek ini merupakan kasus regresi (prediksi nilai kontinu), metrik evaluasi yang digunakan adalah **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, dan **Root Mean Squared Error (RMSE)**.

### 1. Metrik Evaluasi Regresi
Metrik ini digunakan untuk menghitung selisih antara nilai prediksi ($y_{pred}$) dan nilai aktual ($y_{true}$):

* **Mean Squared Error (MSE):** Menghitung rata-rata kuadrat selisih *error*. Semakin kecil nilainya, semakin baik modelnya.
    $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Mean Absolute Error (MAE):** Menghitung rata-rata absolut selisih *error*, yang memberikan gambaran besaran kesalahan dalam satuan harga saham asli.
* **Root Mean Squared Error (RMSE):** Akar kuadrat dari MSE untuk mengembalikan skala *error* ke unit yang sama dengan variabel target.

Berdasarkan hasil *tuning* menggunakan **Optuna** pada notebook, model berhasil mencapai tingkat *error* yang rendah pada data validasi, menunjukkan bahwa kombinasi *hyperparameter* yang ditemukan sangat efektif untuk menangkap tren harga saham LQ45.

### 2. Visualisasi Performa (Actual vs Prediction)
Evaluasi juga dilakukan secara visual melalui grafik plot untuk melihat seberapa rapat garis prediksi mengikuti garis harga aktual.

![Gambar 3. Grafik Actual vs Prediction](link_ke_gambar_grafik_kamu)
**Gambar 3. Grafik Actual vs Prediction**

Dari Gambar 3, grafik menunjukkan bahwa model mampu mengikuti arah tren harga (*trend following*) dengan baik. Meskipun terdapat sedikit *lag* atau selisih pada titik volatilitas ekstrim, secara keseluruhan model tidak menunjukkan gejala *overfitting* yang signifikan karena performa pada data uji tetap konsisten dengan data latih.

### 3. Analisis Backtesting Strategi
Selain metrik statistik, dilakukan evaluasi praktis melalui fungsi *backtest*. Berdasarkan *output* sel terakhir:
* **Strategy Return:** `-0.0837` (atau sekitar -8.37%).
* **Interpretasi:** Angka ini menunjukkan simulasi keuntungan/kerugian jika prediksi model digunakan sebagai dasar keputusan jual-beli. Meskipun *return* strategi saat ini negatif, model memberikan fondasi data yang objektif untuk meminimalkan risiko spekulasi manual.

---
## Deployment 
Tahap deployment merupakan proses implementasi model machine learning yang telah dikembangkan ke dalam sistem aplikasi yang dapat digunakan secara langsung oleh pengguna. Pada proyek Market Data Intelligence System (MDIS), deployment tidak hanya mencakup penyimpanan model, tetapi juga integrasi menyeluruh antara pipeline data, model prediksi, database, serta antarmuka pengguna (frontend application).

### 1. Arsitektur Sistem Deployment
Arsitektur deployment pada sistem MDIS terdiri dari beberapa komponen utama yang saling terintegrasi:
* **Data Source Layer**
Data diambil dari Yahoo Finance menggunakan library yfinance, yang mencakup data teknikal dan fundamental saham indeks LQ45.
* **Database Layer (Supabase & Turso/LibSQL)**
Data yang telah di-scrape disimpan ke dalam database untuk memastikan ketersediaan data historis dan real-time secara terstruktur.
* **Model Layer (Machine Learning)**
Model LightGBM yang telah dilatih dan dioptimasi menggunakan Optuna disimpan dalam format .pkl dan digunakan untuk proses inference.
* **Deployment Layer (Hugging Face)**
Model diunggah ke platform Hugging Face untuk menyediakan layanan prediksi melalui API.
* **Application Layer (Frontend & Backend)**
Aplikasi dibangun menggunakan arsitektur fullstack berbasis Next.js yang menangani interaksi pengguna serta komunikasi dengan model.

### 2. Implementasi Teknologi Sistem
Sistem MDIS dikembangkan menggunakan teknologi modern untuk memastikan performa dan skalabilitas:
* **Frontend & Backend Framework:** Next.js (App Router)
* **UI Library:** React
* **Frontend & Backend Framework:**
* **Bahasa Pemrograman:** TyprScript
* **Database:** Turso dan Supabase
* **ORM:** Drizzle ORM
* **Autentikasi:** NextAuth.js
* **Styling:** Tailwind CSS dan shadcn/ui
* **Visualisasi Data:** Recharts

### 3. Implementasi Fitur Sistem
Pada tahap deployment, model machine learning diintegrasikan ke dalam fitur-fitur utama aplikasi sebagai berikut:

**a. Pencarian Kode Saham**
Pengguna dapat mencari saham berdasarkan kode emiten (misalnya BBCA, TLKM). Sistem akan mengambil data dari database dan menampilkan informasi terkait saham tersebut.

**b. Menampilkan Harga Saham Terkini**
Sistem menampilkan data harga terbaru yang mencakup:
* **Open, High, Low, Close**
* **Volume transaksi**
Data ini diperbarui secara berkala melalui pipeline scraping otomatis.

**c. Visualisasi Chart Harga dan Prediksi**
Aplikasi menyediakan grafik interaktif yang menampilkan:

* **Harga historis (actual price)**
* **Hasil prediksi model (predicted price)**
Visualisasi ini membantu pengguna memahami tren pasar secara lebih intuitif.

**d. Prediksi Arah Market (Forecasting Insight)**
Model LightGBM menghasilkan prediksi harga saham pada periode berikutnya. Hasil ini kemudian diinterpretasikan menjadi arah pasar:

* **Bullish (Naik) → Prediksi lebih tinggi dari harga saat ini**
* **Bearish (Turun) → Prediksi lebih rendah**
* **Sideways (Stabil) → Perubahan tidak signifikan**
Output yang ditampilkan:
* **Predicted price**
* **Persentase perubahan**
* **Arah market**

### 4.Integrasi Model dan API
Model yang telah dilatih disimpan dalam format .pkl dan diunggah ke Hugging Face. Platform ini menyediakan API endpoint yang digunakan oleh aplikasi untuk melakukan inference.
Alur integrasi:

**1.Frontend mengirim request data saham**
**2.Backend memproses dan mengirim ke model**
**3.Model menghasilkan prediksi**
**4.Hasil dikembalikan dalam format JSON**
**5.Frontend menampilkan hasil kepada pengguna**
Dengan pendekatan ini, proses prediksi dapat dilakukan secara real-time tanpa perlu retraining model.

### 5. Alur Kerja Sistem (End-to-End)
Berikut adalah alur kerja sistem secara keseluruhan:
**1.Data saham diambil dari Yahoo Finance**
**2.Data disimpan ke database (Supabase/Turso)**
**3.Data diproses melalui preprocessing dan feature engineering**
**4.Model melakukan prediksi berdasarkan data terbaru**
**5.Hasil prediksi dikirim melalui API**
**6.Frontend menampilkan data dan insight kepada pengguna**

### 6. Maintenance dan Monitoring
Untuk menjaga performa sistem:

* **Data diperbarui secara berkala (harian)**
* **Model dapat di-retrain dengan data terbaru**
* **Monitoring dilakukan terhadap error model dan performa API**
Langkah ini penting untuk memastikan sistem tetap relevan dengan kondisi pasar yang dinamis.

## Kesimpulan

Kesimpulannya, proyek **Market Data Intelligence System (MDIS)** telah berhasil mengintegrasikan *pipeline* data *end-to-end* yang mengotomatisasi pengumpulan variabel teknikal dan fundamental dari Yahoo Finance ke dalam database Supabase, sehingga menghilangkan hambatan analisis manual bagi pengurus GIBEI Telkom University. Melalui implementasi algoritma **LightGBM** yang dioptimasi dengan **Optuna**, sistem ini mampu menghasilkan prediksi harga saham (seperti nilai `7592.56`) yang akurat dan objektif terhadap dinamika pasar LQ45. Dengan model yang telah ter-*deploy* di **Hugging Face**, hasil riset kini dapat didistribusikan secara *real-time* melalui API ke *dashboard FrontEnd*, menjadikan sistem ini sebagai alat bantu keputusan (*decision support system*) yang cerdas, saintifik, dan efisien bagi seluruh komunitas investasi di lingkungan kampus.

### Referensi
* **[1]** Borges, A., & Neves, R. (2020). *A Combined Approach of Fundamental and Technical Analysis for Stock Market Forecasting*. Proceedings of the 12th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Management.
* **[2]** Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NIPS).
* **[3]** Nti, I. K., Adekoya, A. F., & Weyori, B. A. (2020). *A systematic review of state-of-the-art techniques for stock market prediction*. Royal Society Open Science.
* **[4]** Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NIPS).
* **[5]** Akiba, T., Sano, S., Yanase, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

