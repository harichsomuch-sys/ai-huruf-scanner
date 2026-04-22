import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("1. Membaca dataset Kaggle...")
print("   (Proses ini bisa makan waktu 1-2 menit karena filenya lumayan besar)")

# Kita ambil 50.000 baris acak saja supaya laptop nggak nge-hang dan prosesnya cepat.
# Kalau laptop lu kuat dan mau akurasi maksimal, hapus tulisan '.sample(n=50000, random_state=42)'
df = pd.read_csv('A_Z Handwritten Data.csv').sample(n=50000, random_state=42)

print("2. Memisahkan data...")
# Kolom urutan pertama (index 0) adalah label hurufnya (0 = A, 1 = B, dst)
y = df.iloc[:, 0]
# Sisa kolomnya (index 1 sampai 784) adalah data pixel gambarnya
X = df.iloc[:, 1:]

print("3. Sedang melatih AI (Random Forest)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# n_jobs=-1 biar AI pakai semua inti prosesor laptop lu biar ngebut
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42) 
model.fit(X_train, y_train)

# Cek seberapa pintar AI-nya
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred) * 100
print(f"Akurasi AI: {akurasi:.2f}%")

print("4. Menyimpan otak AI...")
joblib.dump(model, 'model_huruf_kertas.pkl')
print("Selesai! File 'model_huruf_kertas.pkl' berhasil dibuat. Silakan lanjut buka webnya!")