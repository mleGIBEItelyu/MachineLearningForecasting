# Laporan Market Data Intelligence System - Machine Learning Engineer GIBEI Telkom University

## Domain Proyek

Galeri Investasi Bursa Efek Indonesia (GIBEI) Telkom University memiliki berbagai divisi yang bekerja sama untuk memajukan edukasi dan praktik pasar modal bagi mahasiswa. Dalam operasional kesehariannya, pengurus GIBEI sangat bergantung pada riset pasar yang mendalam dan prediksi arah pergerakan harga saham, contohnya pada indeks LQ45, sebagai dasar rekomendasi investasi bagi seluruh anggota komunitas.

Selama ini aktivitas edukasi dan riset berjalan dengan baik, tetapi di tengah tingginya volatilitas pasar, para pengurus merasa kewalahan oleh besarnya jumlah variabel data pasar baik data teknikal maupun fundamental yang harus ditarik dan dianalisis secara manual setiap harinya. Hal ini membuat proses memprediksi pergerakan saham memakan waktu lama, rentan terhadap bias manusia, dan menyebabkan hasil riset sering kali terlambat untuk didistribusikan kepada anggota.

Melihat kendala operasional tersebut, Divisi Machine Learning Engineer menyadari bahwa mereka perlu turun tangan untuk memberikan solusi teknologi. Untuk membantu operasional divisi-divisi lain di GIBEI yang membutuhkan data riset cepat, Divisi Machine Learning Engineer berinisiatif membangun infrastruktur analisis yang lebih cerdas, efisien, dan terotomatisasi.

Melalui pemanfaatan teknologi kecerdasan buatan, Divisi Machine Learning Engineer merancang sebuah program kerja bernama **"Market Data Intelligence System (MDIS)"**—sebuah pengembangan *machine learning* untuk *forecasting* harga saham berbasis data pasar historis dan *real-time*. Sistem terintegrasi ini dirancang khusus untuk memproses volume data secara komprehensif dan menghasilkan prediksi arah saham yang lebih cepat serta akurat dibandingkan analisis manual [1].

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

---

### Referensi
* **[1]** Borges, A., & Neves, R. (2020). *A Combined Approach of Fundamental and Technical Analysis for Stock Market Forecasting*. Proceedings of the 12th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Management.
* **[2]** Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NIPS).
* **[3]** Nti, I. K., Adekoya, A. F., & Weyori, B. A. (2020). *A systematic review of state-of-the-art techniques for stock market prediction*. Royal Society Open Science.
* **[4]** Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NIPS).
* **[5]** Akiba, T., Sano, S., Yanase, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

