```markdown
# Klasifikasi UTS - Praktikum Machine Learning

Notebook ini berisi tahapan klasifikasi dataset menggunakan model *Naive Bayes*, dimulai dari preprocessing data hingga evaluasi performa model.

## Tahapan Implementasi

### 1. Import Library

Library yang digunakan dalam proyek ini:
- **pandas**  
  Digunakan untuk membaca dan mengelola data dalam bentuk tabel (*DataFrame*).
- **numpy**  
  Digunakan untuk operasi numerik seperti array dan perhitungan matematis.
- **matplotlib.pyplot** & **seaborn**  
  Digunakan untuk membuat visualisasi seperti grafik, plot, dan heatmap agar analisis data lebih informatif.
- **sklearn.preprocessing** (`LabelEncoder`, `OneHotEncoder`)  
  Digunakan untuk mengubah data kategori menjadi bentuk numerik agar bisa diproses oleh model machine learning.
- **sklearn.model_selection** (`train_test_split`)  
  Digunakan untuk membagi dataset menjadi data latih (*train*) dan data uji (*test*).
- **sklearn.naive_bayes** (`GaussianNB`)  
  Algoritma klasifikasi *Naive Bayes* yang digunakan untuk memodelkan data.
- **sklearn.metrics** (`accuracy_score`, `classification_report`, `confusion_matrix`)  
  Digunakan untuk mengevaluasi performa model menggunakan berbagai metrik evaluasi.
- **imblearn.over_sampling** (`SMOTE`)  
  Digunakan untuk mengatasi data yang tidak seimbang (*imbalanced dataset*) dengan cara membuat sampel sintetis untuk kelas minoritas.


### 2. **Load Dataset**
Dataset dibaca menggunakan `pandas`:
```python
df = pd.read_csv('dataset.csv')
```

### 3. **Eksplorasi Data**
- Melihat 5 data teratas (`df.head()`)
- Menampilkan tipe data dan informasi null (`df.info()`)
- Melihat Distribusi data pada kolom target

### 4. **Preprocessing dan Spliting Dataset**
- Encoding label menggunakan `LabelEncoder`
- Melakukan pembagian dataset menjadi data latih dan uji dengan train_test_split.
- Penanganan data tidak seimbang dengan `SMOTE`
  

### 5. **Training Model**
Model yang digunakan adalah **Gaussian Naive Bayes**:
```python
model = GaussianNB()
model.fit(X_train, y_train)
```

### 6. **Evaluasi Model**
- Menghitung akurasi
- Menampilkan confusion matrix
- Menampilkan classification report

### 8. **Visualisasi**
Visualisasi data dan evaluasi dilakukan dengan menggunakan `matplotlib` dan `seaborn`.
