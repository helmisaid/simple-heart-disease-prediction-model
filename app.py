import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_artifacts():
    """
    Memuat model, scaler, dan selector yang sudah dilatih dari file .joblib.
    """
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        selector = joblib.load('selector.joblib')
        
        with open('selected_features.txt', 'r') as f:
            selected_features = [line.strip() for line in f]
            
        return model, scaler, selector, selected_features
    except FileNotFoundError:
        return None, None, None, None

model, scaler, selector, selected_features = load_artifacts()

st.set_page_config(page_title="Prediksi Penyakit Jantung", page_icon="❤️", layout="wide")

st.title("❤️ Aplikasi Prediksi Penyakit Jantung")

if model is None:
    st.error("File model tidak ditemukan. Harap jalankan `train_model.py` terlebih dahulu.")
else:
    st.header("Masukkan Data Pasien untuk Prediksi")
    
    input_data = {}
    col1, col2 = st.columns(2)
    
    all_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    sex_map = {"Laki-laki": 1, "Perempuan": 0}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs_map = {"Ya (> 120 mg/dl)": 1, "Tidak (<= 120 mg/dl)": 0}
    exang_map = {"Ya": 1, "Tidak": 0}
    
    for i, feature in enumerate(all_features):
        if feature in selected_features:
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                if feature == 'age':
                    input_data[feature] = st.slider("Umur (Age)", 20, 80, 55)
                elif feature == 'sex':
                    input_data[feature] = sex_map[st.selectbox("Jenis Kelamin (Sex)", list(sex_map.keys()))]
                elif feature == 'cp':
                    input_data[feature] = cp_map[st.selectbox("Tipe Nyeri Dada (CP)", list(cp_map.keys()))]
                elif feature == 'ca':
                    input_data[feature] = st.slider("Jumlah Pembuluh Darah Utama (ca)", 0, 4, 0)
                elif feature == 'thalach':
                    input_data[feature] = st.slider("Detak Jantung Maks (thalach)", 70, 210, 150)
                elif feature == 'exang':
                     input_data[feature] = exang_map[st.selectbox("Angina Akibat Olahraga (exang)", list(exang_map.keys()))]
                elif feature not in input_data:
                     input_data[feature] = st.slider(f"Nilai untuk {feature}", 0.0, 10.0, 1.0)
        else:
            input_data[feature] = 0
            
    if st.button("**Prediksi Sekarang**", type="primary"):
        input_df = pd.DataFrame([input_data])
        input_df = input_df[all_features]
        input_scaled = scaler.transform(input_df)
        input_selected = selector.transform(input_scaled)
        prediction = model.predict(input_selected)
        prediction_proba = model.predict_proba(input_selected)
        
        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.error(f"Pasien diprediksi **MEMILIKI** penyakit jantung.", icon="💔")
            st.write(f"Probabilitas: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.success(f"Pasien diprediksi **TIDAK MEMILIKI** penyakit jantung.", icon="❤️‍🩹")
            st.write(f"Probabilitas: {prediction_proba[0][0]*100:.2f}%")