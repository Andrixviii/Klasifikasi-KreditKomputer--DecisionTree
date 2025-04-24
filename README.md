
# 📊 Klasifikasi Kelayakan Kredit Komputer dengan Decision Tree

---

## 🔍 Tahapan Pengerjaan

### 1. Import Library
import library yang dibutuhkan:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

### 2. Load Dataset
Membaca dataset menggunakan `pandas`:
```python
df = pd.read_csv('dataset_buys _comp.csv')
```

### 3. Pra-pemrosesan Data
- Melakukan encoding data kategorikal menjadi numerik.
- Memastikan semua fitur dapat dibaca oleh model:
```python
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].astype('category').cat.codes
```

### 4. Split Dataset
Membagi dataset menjadi **fitur (X)** dan **label (y)**, lalu memisahkannya menjadi data latih dan data uji:
```python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Melatih Model
Membuat model klasifikasi menggunakan `DecisionTreeClassifier` dan melatihnya dengan data latih:
```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 6. Evaluasi Model
Melakukan prediksi pada data uji dan evaluasi kinerja model:
```python
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### 7. Visualisasi Decision Tree
Menampilkan struktur Decision Tree agar lebih mudah dianalisis:
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=["Tidak Layak", "Layak"], filled=True)
plt.show()
```

