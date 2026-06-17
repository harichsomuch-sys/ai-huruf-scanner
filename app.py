import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2

# --- CONFIG HALAMAN ---
st.set_page_config(page_title="AI Scanner Huruf", page_icon="📝", layout="centered")

st.title("📝 AI Scanner Huruf Tulisan Tangan")
st.write("Gunakan spidol hitam di kertas putih, lalu upload foto satu huruf di sini.")

# --- PENJELASAN HYPERPARAMETER ---
with st.expander("ℹ️ Info: Bagaimana AI Mengenali Huruf?"):
    st.write("""
    Model ini menggunakan algoritma **Support Vector Machine (SVM)** untuk mengenali pola tulisan tangan.
    Berikut adalah pengaturan *hyperparameter* yang digunakan agar AI bekerja optimal:
    """)
    
    st.table({
        "Hyperparameter": ["Kernel", "C", "Gamma"],
        "Nilai": ["RBF", "1.0", "Scale"],
        "Fungsi/Alasan": [
            "Memetakan piksel ke dimensi lebih tinggi agar pola huruf yang kompleks dapat dibedakan.",
            "Menjaga keseimbangan agar model tidak terlalu 'kaku' (overfitting) pada data latihan.",
            "Mengatur jangkauan pengaruh setiap titik piksel terhadap keputusan akhir klasifikasi."
        ]
    })
    st.caption("Konfigurasi ini memastikan AI tetap presisi meski bentuk tulisan tangan tiap orang berbeda.")

# --- 1. LOAD MODEL ---
@st.cache_resource # Biar nggak loading model terus setiap klik tombol
def load_model():
    try:
        return joblib.load('model_huruf_kertas.pkl')
    except:
        return None

model = load_model()

if model is None:
    st.error("❌ File 'model_huruf_kertas.pkl' tidak ditemukan! Pastikan sudah jalankan train_model.py")
    st.stop()

# Kamus angka ke huruf (0=A, 1=B, dst)
kamus_huruf = {i: chr(65+i) for i in range(26)}

# --- 2. UPLOAD GAMBAR ---
uploaded_file = st.file_uploader("Upload Foto Huruf", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Buka gambar asli
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Asli", width=300)
    
    if st.button("🔎 Kenali Huruf Sekarang"):
        with st.spinner("AI sedang memproses..."):
            
            # --- TAHAP 1: KONVERSI KE OPENCV ---
            img_cv = np.array(image.convert('RGB'))
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            
            # --- TAHAP 2: INVERT & THRESHOLD (BERSIHKAN KERTAS) ---
            if np.mean(img_gray) > 127:
                img_gray = cv2.bitwise_not(img_gray)
            
            _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # --- TAHAP 3: AUTO-CROP & SQUARE PADDING ---
            contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                img_cropped = img_thresh[y:y+h, x:x+w]
                
                max_side = max(w, h)
                img_square = np.zeros((max_side, max_side), dtype=np.uint8)
                
                ax, ay = (max_side - w) // 2, (max_side - h) // 2
                img_square[ay:ay+h, ax:ax+w] = img_cropped
                
                pad = int(max_side * 0.2)
                img_final_prep = cv2.copyMakeBorder(img_square, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            else:
                img_final_prep = img_thresh

            # --- TAHAP 4: RESIZE & TEBALKAN ---
            img_resized = cv2.resize(img_final_prep, (28, 28), interpolation=cv2.INTER_AREA)
            kernel = np.ones((2,2), np.uint8)
            img_to_predict = cv2.dilate(img_resized, kernel, iterations=1)
            
            st.image(img_to_predict, caption="Wujud yang dilihat AI (28x28)", width=150)
            
            # --- TAHAP 5: PREDIKSI ---
            pixel_data = img_to_predict.flatten().reshape(1, -1)
            
            pred_index = model.predict(pixel_data)[0]
            pred_letter = kamus_huruf[pred_index]
            
            prob = model.predict_proba(pixel_data)[0]
            confidence = np.max(prob) * 100
            
            # --- HASIL AKHIR ---
            st.divider()
            if confidence > 10:
                st.balloons()
                st.success(f"### 🎉 Hasil Prediksi: **{pred_letter}**")
                st.info(f"Tingkat Keyakinan AI: {confidence:.2f}%")
            else:
                st.warning("AI agak bingung. Coba tulis hurufnya lebih tebal dan jelas!")
