# Laporan Proyek Machine Learning - Ratu Chairunisa

## Domain Proyek
Dalam dunia pendidikan, peningkatan performa akademik siswa telah lama menjadi fokus utama institusi dan pendidik. Salah satu pendekatan yang semakin populer untuk memahami faktor-faktor yang memengaruhi performa siswa adalah melalui educational data mining (EDM), yaitu eksplorasi data pendidikan menggunakan teknik analitik dan machine learning (ML).

Berdasarkan penelitian yang dilakukan oleh Aljaffer, Almadani, AlDughaither, dkk. (2024) dalam artikel berjudul "Dampak kebiasaan belajar dan faktor pribadi terhadap prestasi akademik mahasiswa kedokteran", teridentifikasi bahwa gaya hidup dan faktor sosial memainkan peran krusial dalam menentukan kinerja akademik mahasiswa. Lebih lanjut, studi Jafari et al. menunjukkan adanya perbedaan signifikan dalam kebiasaan belajar antara siswa asli dan siswa asrama, di mana siswa asli cenderung memiliki kebiasaan yang lebih superior. Temuan ini diperkuat oleh Jouhari et al. yang menyoroti skor lebih rendah pada siswa asrama dalam aspek sikap, strategi ujian, pemilihan gagasan utama, dan konsentrasi. Curcio G et al. juga menegaskan bahwa mahasiswa dengan pola tidur yang teratur dan adekuat cenderung mencapai Rata-rata Nilai (GPA) yang lebih tinggi. Sementara itu, faktor gaya hidup seperti menonton televisi dan mendengarkan musik dinilai memiliki dampak minimal terhadap nilai akademik, namun aplikasi media sosial, termasuk WhatsApp, Facebook, dan Twitter, terbukti menjadi distraktor yang signifikan selama proses belajar.

Masalah ini penting diselesaikan karena dengan mengetahui faktor-faktor gaya hidup yang paling memengaruhi performa akademik, institusi pendidikan dapat merancang intervensi yang lebih tepat sasaran untuk meningkatkan prestasi siswa.

**Referensi**
Aljaffer, M.A., Almadani, A.H., AlDughaither, A.S. et al. _The impact of study habits and personal factors on the academic achievement performances of medical students_. BMC Med Educ 24, 888 (2024) [https://doi.org/10.1186/s12909-024-05889-y](https://doi.org/10.1186/s12909-024-05889-y)

## Business Understanding
### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana kebiasaan siswa sehari-hari (seperti pola tidur, durasi belajar, dan penggunaan media sosial) memengaruhi performa akademik mereka?
- Fitur gaya hidup mana yang memiliki pengaruh paling signifikan terhadap nilai ujian akhir?

### Goals
- Mengidentifikasi pola kebiasaan hidup siswa yang berkorelasi kuat dengan performa akademik.
- Memprediksi nilai ujian akhir siswa berdasarkan data kebiasaan hidup mereka menggunakan algoritma machine learning. 

### Solution statements
Untuk mencapai tujuan ini, akan digunakan beberapa algoritma supervised learning seperti:
- Linear Regression sebagai baseline model.
- Random Forest Regressor dan XGBoost Regressor sebagai model yang lebih kompleks, dengan proses hyperparameter tuning untuk meningkatkan performa.

Model-model ini akan dievaluasi menggunakan metrik regresi yang terukur, seperti:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Data Understanding
Dataset ini terdiri dari 1.000 entri data siswa yang terdiri dari 16 fitur kebiasaan harian dan target berupa nilai ujian akhir. 
source dataset: [Kaggle Dataset Repository](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance).

### Variabel-variabel pada Students Habits vs Academic Performance kaggle dataset adalah sebagai berikut:
- student_id    			    : merupakan ID unik yang berbeda-beda yang dimiliki setiap siswa
- age	        			    : merupakan umur siswa yang terdata
- gender	        		    : merupakan gender siswa (female, male, others)
- study_hours_per_day         	: merupakan lamanya waktu belajar yang dihabiskan dalam satu hari    
- social_media_hours	    	: merupakan lamanya waktu menggunakan sosial media
- netflix_hours                 : merupakan durasi waktu yang dihabiskan untuk menonton dari app netflix
- part_time_job			        : merupakan fitur yang menyatakan status mahasiswa yang melakukan part time dan yang tidak melakukan part time
- attendance_percentage	    	: merupakan persentase performa kehadiran siswa
- sleep_hours		        	: merupakan durasi waktu yang digunakann siswa untuk tidur dan beristirahat
- diet_quality		        	: merupakan informasi kualitas diet yang dilakukan siswa yang terdiri dari tiga kategori yaitu Fair, Good, dan Poor.
- exercise_frequency		    : merupakan frekuensi latihan terkait materi yang dilakukan siswa 
- parental_education_level  	: merupakan tingkat pendidikan yang ditempuh orang tua siswa yang terdiri dari tiga kategori yaitu High School, Bachelor, Master.
- internet_quality		        : merupakan informasi kualitas internet yang digunakan yang terdiri dari tiga kategori yaitu Good, Average, dan Poor.
- mental_health_rating	    	: merupakan tingkat keparahan kesehatan mental yang dialami siswa
- extracurricular_participation	: merupakan informasi terkait siswa yang mengikuti kegiatan ekstrakurikuler dan yang tidak.
- exam_score 			        : merupakan informasi mengenai nilai ujian                   

Beberapa teknik EDA (Exploratory Data Analysis):
- Scatter Plot: untuk melihat hubungan antara beberapa fitur dengan fitur target (nilai ujian)
- Korelasi Fitur: Heatmap korelasi untuk mengidentifikasi hubungan antar fitur dan target.
- Pairplot: untuk melihat korelasi dan distribusi seluruh fitur dengan menggunakan scatter plot.
- Bar Chart: untuk melihat jumlah data pada data kategorikal

**Visualisasi Data**:
![Heatmap Correlation](image/heatmap_corr.png)

Heatmap korelasi di atas menunjukkan hubungan antara fitur-fitur baik hubungan positif, negatif, atau tidak terdapat hubungan sama sekali. Dari heatmap di atas terlihat dengan jelas dua fitur yang saling berkorelasi kuat di antara banyak fitur yaitu fitur yang menunjukkan lamanya durasi yang dihabiskan siswa untuk belajar dengan nilai ujian.

## Data Preparation
Langkah-langkah data preparation yang dilakukan meliputi:
1. Cek informasi data: tahapan awal ini berguna untuk mengenal struktur dataset sebelum dilakukan analisis lebih dalam. Ini juga membantu menentukan preprocessing yang sesuai. Tujuannya adalah untuk:
* Mengetahui tipe data (numerik, kategorikal, objek)
* Mengetahui jumlah entri dan kolom
* Mengidentifikasi kolom dengan nilai null/missing
2. Cek dan Mengatasi Data Hilang: model ML tidak bisa dilatih dengan data kosong (NaN). Metode Imputasi menjaga agar distribusi data tetap masuk akal tanpa kehilangan banyak informasi, terutama kalau data yang hilang cukup besar proporsinya. Tujuannya agar model tidak eror saat training.
3. Cek Data Duplikat: Duplikat bisa membuat model belajar pola yang tidak sebenarnya dan menyebabkan overfitting. Data yang bersih = hasil prediksi yang lebih valid. Tujuannya untuk menghindari bias dari entri data yang berulang.
4. Menetapkan Kolom Target/ variable dependent: tanpa target, tidak bisa dilakukan supervised learning. Di sini kamu mencoba memprediksi nilai ujian akhir siswa, sehingga Exam Score ditetapkan sebagai target. Tujuannya untuk menentukan variable yang akan diprediksi oleh model dalam hal ini adalah nilai ujian (exam_score).
5. Cek dan Menangani Outliers: outlier bisa membuat model regresi melenceng jauh, karena regresi sensitif terhadap nilai ekstrim. Dengan menangani outlier, artinya kita mampu menjaga konsistensi distribusi data tanpa menghapus data penting. Tujuannya untuk mengurangi pengaruh data ekstrem yang membelokkan model terutama model berbasis regresi seperti yang digunakan dalam proyek ini.
6. Encoding Menggunakan Label Encoder (atau OneHotEncoder): model ML hanya bisa membaca angka, bukan teks. Encoding memungkinkan model untuk menginterpretasi informasi kategorikal seperti “Kualitas Diet”, “Status Mental”, dll. Tujuannya untuk mengubah data kategorikal menjadi format numerik yang bisa dipahami oleh model ML.
7. Splitting Data: supaya kita bisa melatih model di sebagian data, lalu mengujinya pada data yang belum pernah dilihat untuk mengukur kemampuan generalisasi.Tujuannya adalah untuk memisahkan data menjadi data pelatihan dan data pengujian.
8. Feature Scaling:
* Scaling diperlukan agar semua fitur numerik berada pada skala yang sama, penting untuk model seperti Linear Regression dan KNN.
* OneHotEncoder digunakan agar fitur kategorikal tidak dianggap ordinal oleh model.
* Pipeline membuat proses lebih efisien dan terstruktur, terutama saat dipakai berulang atau saat digunakan dalam model pipeline (training, evaluasi, deployment). 
Tujuannya adalah untuk melakukan standarisasi pada fitur numerik agar memiliki mean 0 dan std 1 dan melakukan one-hot encoding pada fitur kategorikal agar bisa digunakan saat training model.

# Modeling
Dalam tahap ini, dilakukan proses pemodelan Machine Learning untuk memprediksi nilai ujian akhir siswa berdasarkan fitur-fitur kebiasaan hidup mereka. Beberapa algoritma regresi yang digunakan antara lain:

**a. Linear Regression (Baseline Model)**

Tahapan:
- Melatih model pada data latih.
- Menggunakan semua fitur numerik dan kategorikal (setelah encoding).
- Parameter: Menggunakan default dari sklearn.linear_model.LinearRegression.

Kelebihan:
- Interpretabilitas tinggi.
- Cepat dilatih dan tidak kompleks.
Kekurangan:
- Tidak dapat menangkap hubungan non-linear.
- Sensitif terhadap multikolinearitas dan outlier.

**b. XGBoost Regressor (Model Improvement)**

Tahapan:
- Gradient boosting yang efisien untuk regresi dan klasifikasi.
- Hyperparameter tuning menggunakan Grid Search.
- Parameter Tuning:
- n_estimators = [100, 200]
- learning_rate = [0.05, 0.1, 0.2]
- max_depth = [3, 5, 7]
- subsample = [0.8, 1]

Kelebihan:
- Akurasi tinggi dan cocok untuk kompetisi ML.
- Menangani data tidak seimbang dan missing value dengan baik.
Kekurangan:
- Relatif kompleks dan butuh waktu tuning lebih lama.
- Lebih sulit diinterpretasi.

**Memilih Model Terbaik Berdasarkan Metrik yang Digunakan**

1. Akurasi Lebih Tinggi (R² Score)
Linear Regression menghasilkan R² Score sebesar 0.896, lebih tinggi dibanding XGBoost (0.880). Ini berarti Linear Regression mampu menjelaskan sekitar 89,6% variansi nilai ujian akhir berdasarkan fitur gaya hidup siswa.
2. Kesalahan Lebih Kecil (MAE & RMSE)
    - MAE (Mean Absolute Error): Linear Regression = 4.19, lebih kecil dari XGBoost = 4.55. Artinya, rata-rata kesalahan prediksi Linear Regression lebih rendah.
    - RMSE (Root Mean Squared Error): Linear Regression = 5.15, juga lebih baik dari XGBoost = 5.54. Ini penting terutama jika kamu ingin meminimalkan kesalahan besar.

3. Model yang Lebih Sederhana & Interpretable
Linear Regression sangat mudah dijelaskan ke stakeholder non-teknis dan cocok untuk kasus edukasi seperti ini, di mana penting untuk mengetahui hubungan linier antar variabel seperti durasi belajar, tidur, dan nilai ujian.

4. Generalisasi Lebih Baik pada Data Ini
Dalam konteks dataset ini yang bersifat simulasi dan linier, Linear Regression terbukti memberikan performa optimal tanpa perlu kompleksitas tambahan seperti yang dimiliki XGBoost.

📌 Kesimpulan Akhir:
Linear Regression dipilih sebagai model terbaik dalam proyek ini **karena menghasilkan metrik evaluasi terbaik secara konsisten (MAE, RMSE, dan R²)**, memiliki kompleksitas rendah, serta mudah dijelaskan. Model ini ideal untuk mendukung analisis hubungan gaya hidup siswa dengan kinerja akademik secara transparan dan efisien.

## Evaluation
Dalam proyek ini, kita membandingkan dua model regresi: Linear Regression dan XGBoost untuk memprediksi nilai ujian akhir siswa berdasarkan gaya hidup mereka (seperti durasi belajar, tidur, penggunaan sosial media, dsb).

**Ringkasan Metrik Evaluasi Model**
| Model                 | MAE      | RMSE     | R² Score  |
| --------------------- | -------- | -------- | --------- |
| **Linear Regression** | **4.19** | **5.15** | **0.896** |
| XGBoost               | 4.55     | 5.54     | 0.880     |

Dari hasil di atas, dapat disimpulkan bahwa:
Linear Regression memiliki performa lebih baik dibanding XGBoost karena menghasilkan MAE dan RMSE yang lebih rendah, serta R² Score yang lebih tinggi.
Artinya, Linear Regression lebih akurat dalam memprediksi nilai ujian akhir, dan lebih efisien dalam meminimalkan kesalahan prediksi.

✅ Konteks Data & Problem Statement:
- Tujuan dari proyek ini adalah memodelkan dan memprediksi nilai akhir siswa berdasarkan faktor gaya hidup.
- Karena ini adalah masalah regresi, maka kita tidak menggunakan metrik klasifikasi seperti accuracy atau F1-score.
- Fokus utama adalah seberapa dekat prediksi model dengan nilai aktual.

✅ Solusi yang Diinginkan:
- Model dengan kemampuan prediksi yang stabil dan akurat.
- Metrik yang menggambarkan seberapa besar kesalahan prediksi dan seberapa banyak variansi yang dijelaskan oleh model.

✅ Metrik yang Digunakan Sesuai Karena:
- MAE & RMSE memberi tahu seberapa besar rata-rata kesalahan dalam satuan nilai ujian.
- R² Score menunjukkan seberapa besar bagian dari variansi nilai ujian akhir yang bisa dijelaskan oleh fitur-fitur gaya hidup.

### Penjelasan Metrik Evaluasi yang Digunakan:
#### **Mean Absolute Error (MAE)**
* Formula:
  
$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$

* Penjelasan:
MAE menghitung rata-rata dari semua selisih absolut antara nilai aktual (y) dan nilai prediksi (ŷ). Metrik ini mudah diinterpretasikan karena memiliki satuan yang sama dengan target (nilai ujian).
* Kelebihan: Tidak terlalu sensitif terhadap outlier.

#### Root Mean Squared Error (RMSE)
* Formula:

$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 }$

* Penjelasan:
RMSE menghitung akar dari rata-rata kuadrat selisih antara nilai aktual dan prediksi. Karena mengkuadratkan kesalahan, RMSE lebih sensitif terhadap kesalahan besar/outlier dibanding MAE.
* Kelebihan: Menekankan penalti lebih besar terhadap prediksi yang jauh meleset.

#### R² Score (Koefisien Determinasi)
* Formula:

$R^2 = 1 - \frac{ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }{ \sum_{i=1}^{n} (y_i - \bar{y})^2 }$

* Penjelasan:
R² Score mengukur seberapa banyak variasi target (nilai ujian akhir) yang bisa dijelaskan oleh fitur-fitur input. Nilai R² berkisar antara 0 hingga 1:
Semakin mendekati 1, semakin baik model menjelaskan data.
* R² = 0 berarti model tidak menjelaskan variansi sama sekali.
