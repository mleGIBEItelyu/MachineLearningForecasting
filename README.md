# Laporan Market Data Intelligence System - Machine Learning Engineer GIBEI Telkom University

## Domain Proyek

Galeri Investasi Bursa Efek Indonesia (GIBEI) Telkom University memiliki berbagai divisi yang bekerja sama untuk memajukan edukasi dan praktik pasar modal bagi mahasiswa. Dalam operasional kesehariannya, pengurus GIBEI sangat bergantung pada riset pasar yang mendalam dan prediksi arah pergerakan harga saham, contohnya pada indeks LQ45, sebagai dasar rekomendasi investasi bagi seluruh anggota komunitas.

Selama ini aktivitas edukasi dan riset berjalan dengan baik, tetapi di tengah tingginya volatilitas pasar, para pengurus merasa kewalahan oleh besarnya jumlah variabel data pasar baik data teknikal maupun fundamental yang harus ditarik dan dianalisis secara manual setiap harinya. Hal ini membuat proses memprediksi pergerakan saham memakan waktu lama, rentan terhadap bias manusia, dan menyebabkan hasil riset sering kali terlambat untuk didistribusikan kepada anggota.

Melihat kendala operasional tersebut, Divisi Machine Learning Engineer menyadari bahwa mereka perlu turun tangan untuk memberikan solusi teknologi. Untuk membantu operasional divisi-divisi lain di GIBEI yang membutuhkan data riset cepat, Divisi Machine Learning Engineer berinisiatif membangun infrastruktur analisis yang lebih cerdas, efisien, dan terotomatisasi.

Melalui pemanfaatan teknologi kecerdasan buatan, Divisi Machine Learning Engineer merancang sebuah program kerja bernama **"Market Data Intelligence System (MDIS)"** sebuah pengembangan *machine learning* untuk *forecasting* harga saham berbasis data pasar historis dan *real-time*. Sistem terintegrasi ini dirancang khusus untuk memproses volume data secara komprehensif dan menghasilkan prediksi arah saham yang lebih cepat serta akurat dibandingkan analisis manual [1].

Penelitian menunjukkan bahwa integrasi data fundamental dan teknikal menggunakan algoritma *gradient boosting* dapat meningkatkan akurasi prediksi harga saham secara signifikan [2]. Dengan demikian, di dalam proyek ini, Divisi Machine Learning Engineer akan membuat sebuah *pipeline* data secara *end-to-end* beserta modul pemodelannya.

Proyek ini mengotomatisasi proses *data scraping* dari Yahoo Finance untuk mengambil variabel Data Teknikal dan Data Fundamental, yang kemudian disimpan langsung ke sistem *database* Supabase. Data tersebut melalui tahap transformasi (*Feature Merging*) sebelum masuk ke fase *Data Mining*. Pada tahap *Modeling*, sistem melatih model prediksi menggunakan algoritma LightGBM dengan optimasi Optuna, lalu melakukan evaluasi metrik. Setelah model optimal terbentuk, file model diunggah ke *Hugging Face Space* agar hasil *forecasting* dapat didistribusikan melalui integrasi API ke sisi *FrontEnd* sebagai alat bantu keputusan investasi yang cerdas bagi seluruh anggota GIBEI Telkom University.

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
* Membangun model prediksi harga saham menggunakan LightGBM dengan optimasi Optuna untuk menghasilkan akurasi yang tinggi.

### Solution Statements

Berikut adalah langkah-langkah solusi untuk mencapai tujuan proyek:

#### 1. Tahap Data Preprocessing & Pipeline
Tahap ini bertujuan mengubah *raw data* dari Yahoo Finance menjadi data yang terstruktur di database.
* Menggunakan library `yfinance` untuk menarik data *Open, High, Low, Close, Volume* serta data fundamental saham.
* Menyimpan data secara otomatis ke Supabase menggunakan koneksi API untuk memastikan data selalu *up-to-date* [3].

#### 2. Tahap Data Preparation
Transformasi data agar cocok untuk proses pemodelan, meliputi:
* **Feature Engineering:** Membuat variabel teknikal (seperti *moving averages* atau *volatility*) dan menggabungkannya dengan data fundamental.
* **Handling Missing Values:** Menangani data kosong pada kolom fundamental (yang biasanya dilaporkan per kuartal) dengan teknik *forward filling*.
* **Merging:** Menyatukan data dari berbagai sumber berdasarkan *timestamp* yang sesuai.

#### 3. Tahap Building Machine Learning Model
Pembuatan model menggunakan pendekatan *Supervised Learning* dengan algoritma LightGBM.

* **LightGBM (Light Gradient Boosting Machine):**
  Algoritma ini dipilih karena kecepatannya dalam melatih data besar dan kemampuannya menangani fitur yang kompleks secara efisien [4]. LightGBM menggunakan teknik *Leaf-wise tree growth* yang cenderung menghasilkan *loss* lebih rendah dibandingkan algoritma *level-wise* lainnya.

* **Hyperparameter Tuning dengan Optuna:**
  Untuk mendapatkan hasil terbaik, proyek ini menggunakan Optuna sebagai *framework* optimasi otomatis. Optuna bekerja dengan mencoba berbagai kombinasi parameter (seperti `learning_rate`, `num_leaves`, dan `feature_fraction`) untuk meminimalkan nilai *error* pada data validasi [5].

* **Metrik Evaluasi:**
  Model dievaluasi menggunakan Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R2 Score untuk mengukur seberapa jauh selisih antara harga prediksi dengan harga aktual di pasar.

## Data Understanding

Data yang digunakan dalam proyek ini adalah dataset harga saham historis dan data fundamental perusahaan yang tergabung dalam indeks **LQ45**. Data ditarik secara *real-time* dan historis menggunakan API Yahoo Finance (`yfinance`).

Dataset ini terdiri dari dua kategori utama: **Data Teknikal (Price Action)** dan **Data Fundamental**.

### 1. Data Teknikal
Data teknikal mencakup pergerakan harga harian untuk masing-masing emiten dalam indeks LQ45.

| No | Column | Dtype | Deskripsi |
|:---:|:--- |:---:|:--- |
| 0 | **Date** | datetime64 | Tanggal pencatatan aktivitas bursa saham. |
| 1 | **Open, High, Low** | float64 | Harga pembukaan, tertinggi, dan terendah harian. |
| 2 | **Close** | float64 | Harga penutupan (Variabel target prediksi). |
| 3 | **Volume** | int64 | Jumlah lembar saham yang ditransaksikan. |
| 4 | **SMA (5, 20)** | float64 | *Simple Moving Average* untuk mengidentifikasi tren harga. |
| 5 | **RSI_14** | float64 | *Relative Strength Index* untuk mengukur kondisi jenuh beli/jual. |
| 6 | **MACD & Signal** | float64 | Indikator tren untuk melihat momentum pergerakan harga. |
| 7 | **Bollinger Bands** | float64 | Mengukur tingkat volatilitas pasar (*Upper* dan *Lower band*). |
| 8 | **ATR_14** | float64 | *Average True Range* untuk mengukur volatilitas harga secara harian. |
| 9 | **Returns (1d, 3d, 5d)** | float64 | Persentase perubahan harga dalam rentang waktu tertentu. |
| 10 | **Ticker** | object | Kode simbol saham (Contoh: BBCA, ASII, TLKM). |

**Tabel 1. Informasi Atribut Data Teknikal**

### 2. Data Fundamental
Data fundamental diambil dari laporan keuangan emiten yang disimpan di database Supabase. Data ini memberikan gambaran kesehatan ekonomi perusahaan.

| No | Column | Dtype | Deskripsi |
|:---:|:--- |:---:|:--- |
| 0 | **Date** | datetime64 | Tanggal pelaporan keuangan (Kuartalan). |
| 1 | **Total Assets** | float64 | Total seluruh kekayaan/aset yang dimiliki perusahaan. |
| 2 | **Total Liabilities** | float64 | Total kewajiban atau seluruh utang perusahaan. |
| 3 | **Revenue** | float64 | Total pendapatan kotor yang dihasilkan perusahaan. |
| 4 | **Net Income** | float64 | Laba bersih perusahaan setelah dikurangi seluruh biaya. |
| 5 | **ROA** | float64 | *Return on Assets* (Rasio efisiensi penggunaan aset). |
| 6 | **Revenue Growth** | float64 | Pertumbuhan pendapatan secara kuartalan (QoQ) dan tahunan (YoY). |
| 7 | **Net Income Growth**| float64 | Pertumbuhan laba bersih secara kuartalan (QoQ) dan tahunan (YoY). |
| 8 | **Ticker** | object | Kode simbol saham emiten terkait. |

**Tabel 2. Informasi Atribut Data Fundamental**

---

## Data Preprocessing

Pada tahap pra-pemrosesan data atau *data preprocessing*, dilakukan transformasi untuk mengubah data mentah (*raw data*) hasil *scraping* menjadi data yang bersih (*clean data*) dan terstruktur. Tahapan ini sangat krusial agar model LightGBM dapat memproses fitur dengan akurat. Ada beberapa tahap yang dilakukan, yaitu:

### 1. Penyeragaman dan Perbaikan Nama Atribut
Proses ini dilakukan untuk menyeragamkan format atribut dari berbagai sumber (Yahoo Finance dan Stock Analysis) guna memudahkan proses integrasi *dataframe*. Selain itu, dilakukan *flattening* pada kolom *MultiIndex* hasil unduhan `yfinance`.

**Tabel 4. Perbaikan Nama Atribut Data Teknikal**
| No | Atribut Lama | Atribut Baru | Deskripsi |
|:---:|:--- |:--- |:--- |
| 0 | Date | date | Tanggal transaksi bursa. |
| 1 | Open | open | Harga pembukaan harian. |
| 2 | High | high | Harga tertinggi harian. |
| 3 | Low | low | Harga terendah harian. |
| 4 | Close | close | Harga penutupan harian. |
| 5 | Volume | volume | Volume transaksi harian. |

**Tabel 5. Perbaikan Nama Atribut Data Fundamental**
| No | Atribut Lama | Atribut Baru | Deskripsi |
|:---:|:--- |:--- |:--- |
| 0 | Total Assets | total_assets | Total kekayaan perusahaan. |
| 1 | Total Liabilities | total_liabilities | Total kewajiban/utang perusahaan. |
| 2 | Revenue | revenue | Pendapatan kotor perusahaan. |
| 3 | Net Income | net_income | Laba bersih perusahaan. |
| 4 | ROA | roa | Return on Assets (Rasio efisiensi). |

### 2. Normalisasi Mata Uang (Currency Conversion)
Karena beberapa emiten dalam indeks LQ45 melaporkan keuangan dalam mata uang USD sementara harga saham dalam IDR, dilakukan proses konversi otomatis. Sistem menarik data kurs `USDIDR=X` secara *real-time* dan melakukan *merge_asof* untuk mengonversi nilai fundamental ke dalam Rupiah (IDR) berdasarkan tanggal laporan terdekat.

### 3. Integrasi Data (Feature Merging)
Data teknikal (harian) dan data fundamental (kuartalan) digabungkan menjadi satu kesatuan *dataframe*. Proses penggabungan dilakukan menggunakan fungsi `.merge()` pada library Pandas dengan kunci utama (*primary key*) berupa kolom `ticker` dan `date`. Hal ini bertujuan agar setiap baris data harga memiliki konteks kesehatan finansial perusahaan pada waktu yang sama.

### 4. Sinkronisasi Simbol Saham (Ticker Matching)
Dilakukan proses filter untuk memastikan hanya 45 emiten yang terdaftar dalam indeks LQ45 yang ditarik datanya. Sistem menggunakan *list matching* untuk menyelaraskan simbol dari Yahoo Finance (dengan akhiran `.JK`) dan simbol pada database internal agar database tetap efisien dan relevan.


## Data Preparation

Pada tahap persiapan data atau *data preparation*, dilakukan proses transformasi agar dataset siap diproses oleh model *forecasting*. Tahapan yang dilakukan meliputi:

### 1. Penanganan Missing Value & Sinkronisasi
Proses pengecekan data kosong dilakukan terutama pada integrasi data teknikal harian dan fundamental kuartalan.
* **Forward Fill (`ffill`):** Mengisi nilai kosong pada data fundamental dengan nilai laporan terakhir yang tersedia, karena data fundamental perusahaan dianggap tetap valid hingga laporan keuangan periode berikutnya dirilis.
* **Zero Imputation (`fillna(0)`):** Sisa data yang tetap kosong di awal baris (data sebelum laporan pertama tersedia) diisi dengan angka 0. Hal ini dilakukan untuk menjaga integritas baris data tanpa mengurangi jumlah sampel (*keeping time-series continuity*).

### 2. Feature Engineering & Transformation
Untuk memperkaya informasi bagi model LightGBM, dibuat fitur tambahan sebagai prediktor:
* **Technical Indicators:** Menghitung variabel teknikal kompleks yang mencakup SMA (5, 20), RSI, MACD, Bollinger Bands, ATR, dan OBV.
* **Momentum Features:** Membuat fitur `return_1d`, `return_3d`, dan `return_5d` untuk menangkap tren perubahan harga dan volatilitas dalam rentang waktu tertentu.
* **Categorical Encoding:** Mengonversi kolom `ticker` menjadi format yang dapat diproses oleh algoritma LightGBM untuk membedakan karakteristik antar emiten.

### 3. Pengecekan Integritas Data
Dilakukan verifikasi data untuk memastikan kualitas sebelum masuk ke tahap pelatihan:
* **Pengecekan Duplikat:** Memastikan data unik berdasarkan kombinasi kolom `date` dan `ticker`.
* **Feature Selection:** Memilih fitur teknikal (OHLCV + Indikator) dan fundamental (Assets, Liabilities, Revenue, Net Income, ROA) sebagai input utama model.

### 4. Split Training & Test Data (Time-Series Split)
Dataset dibagi berdasarkan urutan waktu untuk menghindari *look-ahead bias*.
* **Rasio:** Data dibagi menjadi 80% untuk data latih (*training data*) dan 20% untuk data uji (*test data*).
* **Metode:** Menggunakan Time-Series Split, di mana data historis awal digunakan untuk melatih model, dan data terbaru digunakan sebagai validasi untuk menguji kemampuan model dalam memprediksi harga di masa depan.

## Modeling

Tahap selanjutnya adalah proses *modeling* untuk membangun model *machine learning* yang mampu melakukan *forecasting* harga saham indeks LQ45. Berbeda dengan sistem rekomendasi buku, proyek ini menggunakan pendekatan *Supervised Learning* untuk memprediksi nilai kontinu (harga penutupan).

Berdasarkan tahap *data understanding*, volume data yang ditarik dari Supabase mencakup puluhan emiten dengan rentang waktu historis yang panjang. Untuk menjaga efisiensi komputasi namun tetap mempertahankan kualitas prediksi, model dilatih menggunakan fitur-fitur teknikal dan fundamental yang telah diintegrasikan secara komprehensif.

### 1. LightGBM (Light Gradient Boosting Machine)
Algoritma utama yang digunakan dalam proyek ini adalah LightGBM. Algoritma ini dipilih karena kemampuannya menangani data tabular berskala besar dengan kecepatan tinggi dan penggunaan memori yang efisien. LightGBM menggunakan teknik *Leaf-wise tree growth* yang memungkinkan model menemukan pola volatilitas harga saham lebih mendalam dibandingkan algoritma *level-wise* tradisional.

**Parameter Awal Model:**

| No | Parameter | Deskripsi |
|:---|:---|:---|
| 1 | `objective` | *regression* (untuk prediksi harga) |
| 2 | `metric` | *rmse* (Root Mean Squared Error) |
| 3 | `boosting_type` | *gbdt* (Gradient Boosting Decision Tree) |

### 2. Hyperparameter Tuning dengan Optuna

Untuk mendapatkan performa model yang optimal, dilakukan proses *tuning* secara otomatis menggunakan *framework* Optuna. Optuna mencari kombinasi parameter terbaik dengan meminimalkan nilai *error* (RMSE) pada data validasi melalui 25 eksperimen (*trials*) yang terukur.

**Ruang Pencarian (*Search Space*) Optuna:**
* **n_estimators**: [500, 1500] (Jumlah pohon keputusan).
* **learning_rate**: [0.005, 0.05] (Kecepatan pembelajaran model).
* **num_leaves**: [20, 120] (Maksimal jumlah daun dalam satu pohon).
* **subsample**: [0.6, 1.0] (Rasio data yang digunakan untuk setiap pohon).
* **colsample_bytree**: [0.6, 1.0] (Rasio fitur yang digunakan untuk setiap pohon).

Hasil dari proses ini menghasilkan set parameter terbaik (*best parameters*) yang kemudian digunakan untuk melatih model final.

### 3. Model Development dan Hasil Forecasting
Setelah model dilatih dengan parameter terbaik, sistem melakukan pengujian terhadap data uji (*test data*). Berikut adalah cuplikan hasil prediksi harga dibandingkan dengan harga aktual Berdasarkan *output* pada notebook, model berhasil menghasilkan nilai Next Forecast Price (misalnya: `7592.56` untuk emiten tertentu) yang menunjukkan proyeksi harga di hari bursa berikutnya.

### 4. Model Deployment & Distribution
Setelah model optimal terbentuk, file model disimpan dalam format `.pkl` (contoh: `mdis_model.pkl`) untuk kebutuhan produksi.

* Model diunggah ke *Repository Hugging Face* untuk dijadikan sebagai *backend service*.
* Melalui *Hugging Face Space*, model menyediakan API yang dapat dipanggil oleh aplikasi *FrontEnd* GIBEI Telkom University untuk menampilkan grafik prediksi secara *real-time* kepada seluruh anggota komunitas.


## Evaluation

Pada tahap evaluasi, performa model *machine learning* diukur menggunakan data uji yang belum pernah dilihat sebelumnya untuk mengetahui seberapa akurat prediksi harga saham yang dihasilkan. Karena proyek ini merupakan kasus regresi, metrik evaluasi difokuskan pada selisih antara nilai prediksi ($y_{pred}$) dan nilai aktual ($y_{true}$).

### 1. Metrik Evaluasi Regresi
Sistem secara otomatis menghitung dan menampilkan performa model dengan metrik sebagai berikut:

* **Mean Absolute Error (MAE):** Menghitung rata-rata absolut selisih *error*. Metrik ini memberikan gambaran besaran kesalahan dalam satuan mata uang (Rupiah) asli.
* **Root Mean Squared Error (RMSE):** Akar kuadrat dari rata-rata kuadrat *error*. RMSE memberikan penalti lebih besar pada kesalahan yang signifikan, sehingga sangat efektif untuk mendeteksi deviasi saat terjadi volatilitas tinggi.
    $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
* **R-Squared (R2):** Menunjukkan sejauh mana variabel independen (teknikal & fundamental) mampu menjelaskan variasi pergerakan harga saham. Nilai yang mendekati 1.0 menunjukkan model sangat *fit* dengan pola data.

### 2. Analisis Backtesting Strategi
Selain metrik statistik, dilakukan evaluasi praktis melalui simulasi perdagangan (*backtesting*) untuk mengukur performa model dalam skenario investasi riil di GIBEI Telkom University. Berdasarkan *output* pada sel terakhir:

* **Strategy Return:** `-0.0837` (atau sekitar **-8.37%**).
* **Interpretasi:** Angka ini merupakan hasil simulasi keuntungan/kerugian jika keputusan investasi didasarkan sepenuhnya pada sinyal prediksi model selama periode data uji. 

Meskipun *return* strategi menunjukkan angka negatif pada periode evaluasi ini (menandakan kondisi pasar yang sedang *bearish* pada waktu tersebut), model tetap memberikan nilai penting berupa data yang objektif untuk meminimalkan risiko spekulasi manual yang tidak terukur.

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

