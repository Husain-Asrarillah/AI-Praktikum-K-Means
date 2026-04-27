# Customer Segmentation menggunakan K-Means Clustering

Proyek ini bertujuan untuk melakukan segmentasi pelanggan (*Customer Segmentation*) menggunakan algoritma **K-Means Clustering**. Data yang digunakan adalah *Mall Customer Segmentation Data* yang bersumber dari Kaggle. Analisis ini membantu bisnis memahami perilaku pelanggan berdasarkan pendapatan tahunan dan skor pengeluaran.

## 🛠️ Persiapan Lingkungan
Proyek ini dijalankan di lingkungan Google Colab dengan menggunakan library `kagglehub` untuk pengambilan dataset secara langsung.

```python
# Instalasi library kagglehub
!pip install kagglehub -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import kagglehub
```

## 1. Import Dataset
Dataset diunduh langsung dari Kaggle API menggunakan `kagglehub`.

```python
# Download dataset
path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
dataset_path = os.path.join(path, "Mall_Customers.csv")

# Memuat data ke DataFrame
dataset = pd.read_csv(dataset_path)
dataset.head()
```

## 2. Seleksi Fitur
Untuk kebutuhan visualisasi 2 dimensi, kita hanya menggunakan dua fitur utama:
* **Annual Income (k$)**: Pendapatan tahunan pelanggan.
* **Spending Score (1-100)**: Skor pengeluaran yang diberikan oleh mal.

```python
X = dataset.iloc[:, 3:5]
X.head()
```

## 3. Eksplorasi Data Singkat
Melakukan pengecekan terhadap dimensi data, nilai kosong (*missing values*), dan statistik deskriptif.

```python
# Dimensi data
print(f"Banyaknya baris: {X.shape[0]}, Banyaknya kolom: {X.shape[1]}")

# Cek missing values
print(X.isnull().sum())

# Statistik deskriptif
X.describe()
```

## 4. Menentukan Jumlah Cluster (Metode Elbow)
Metode Elbow digunakan untuk menentukan jumlah cluster ($k$) yang paling optimal dengan melihat titik "siku" pada grafik WCSS (*Within Cluster Sum of Squares*).

```python
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 14)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 15), wcss, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
```
*Berdasarkan grafik di atas, titik siku terlihat pada k=5.*

## 5. Pemodelan K-Means
Menerapkan algoritma K-Means dengan jumlah cluster sebanyak 5.

```python
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 14)
kmeans.fit(X)

# Menggabungkan hasil cluster ke dalam data frame
hasil_kmeans = X.copy()
hasil_kmeans["cluster"] = kmeans.labels_
```

## 6. Visualisasi Hasil

### A. Frekuensi Data per Cluster
Menampilkan jumlah pelanggan yang tersebar di setiap cluster.

```python
data_frekuensi = hasil_kmeans["cluster"].value_counts().sort_index()
sns.barplot(x=data_frekuensi.index, y=data_frekuensi.values, hue=data_frekuensi.index, palette='viridis', legend=False)
plt.title("Frekuensi Data pada Masing-Masing Cluster")
plt.xlabel("Cluster")
plt.ylabel("Jumlah Customer")
plt.show()
```

### B. Scatter Plot Clusters
Visualisasi persebaran pelanggan berdasarkan Annual Income dan Spending Score.

```python
# Definisi koordinat centroid
centroid_cluster = kmeans.cluster_centers_

plt.figure(figsize=(10, 7))
colors = ['blue', 'orange', 'green', 'red', 'magenta']
labels = ['Cluster 1 (Sedang)', 'Cluster 2 (Hemat)', 'Cluster 3 (Target)', 'Cluster 4 (Impulsif)', 'Cluster 5 (Loyal)']

for i in range(5):
    plt.scatter(hasil_kmeans[hasil_kmeans["cluster"] == i].iloc[:, 0], 
                hasil_kmeans[hasil_kmeans["cluster"] == i].iloc[:, 1], 
                s = 80, c = colors[i], label = labels[i])

plt.scatter(centroid_cluster[:, 0], centroid_cluster[:, 1], s = 160, c = "black", label = "Centroids", edgecolors='white')
plt.title("Segmentasi Pelanggan Mall")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
```

## 7. Interpretasi Hasil & Strategi
Berdasarkan hasil visualisasi, pelanggan terbagi menjadi 5 segmen utama:

1. **Cluster 1 (Standard)**: Pendapatan sedang, pengeluaran sedang.
2. **Cluster 2 (Economical)**: Pendapatan rendah, pengeluaran rendah.
3. **Cluster 3 (Target Potensial)**: Pendapatan tinggi, namun pengeluaran masih rendah. Strategi: Berikan promosi produk premium.
4. **Cluster 4 (Careless)**: Pendapatan rendah, namun pengeluaran sangat tinggi.
5. **Cluster 5 (Loyal/VVIP)**: Pendapatan tinggi dan pengeluaran tinggi. Strategi: Berikan program loyalitas atau reward eksklusif.

## 8. Export Hasil
Menyimpan hasil segmentasi ke dalam format CSV untuk kebutuhan analisis lebih lanjut.

```python
hasil_kmeans["CustomerID"] = dataset["CustomerID"]
hasil_kmeans.to_csv("Hasil_Clustering_Mall_Customers.csv", index = False)
```

---
**Referensi:**
* Dataset: [Mall Customer Segmentation Data - Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
* Metode: K-Means Clustering
```
