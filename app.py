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
    
    feature_labels = {
        'age': 'Umur (Tahun)',
        'sex': 'Jenis Kelamin',
        'cp': 'Tipe Nyeri Dada',
        'trestbps': 'Tekanan Darah Istirahat (mm Hg)',
        'chol': 'Kolesterol Serum (mg/dl)',
        'fbs': 'Gula Darah Puasa > 120 mg/dl',
        'restecg': 'Hasil EKG Istirahat',
        'thalach': 'Detak Jantung Maksimum Tercapai',
        'exang': 'Nyeri Dada Akibat Olahraga',
        'oldpeak': 'Depresi ST Akibat Olahraga (Oldpeak)',
        'slope': 'Kemiringan Puncak Latihan ST',
        'ca': 'Jumlah Pembuluh Darah Utama (0-4)',
        'thal': 'Status Thalassaemia'
    }

    input_data = {}
    col1, col2 = st.columns(2)
    
    all_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    sex_map = {"Laki-laki": 1, "Perempuan": 0}
    cp_map = {"0: Tipikal Angina": 0, "1: Atipikal Angina": 1, "2: Nyeri Non-Angina": 2, "3: Asimtomatik": 3}
    fbs_map = {"Ya": 1, "Tidak": 0}
    exang_map = {"Ya": 1, "Tidak": 0}
    thal_map = {"Normal": 1, "Cacat Tetap (Fixed Defect)": 2, "Cacat Reversibel (Reversible Defect)": 3}
    
    for i, feature in enumerate(all_features):
        if feature in selected_features:
            target_col = col1 if i % 2 == 0 else col2
            label = feature_labels.get(feature, feature)
            
            with target_col:
                if feature == 'age':
                    input_data[feature] = st.slider(label, 20, 80, 55)
                elif feature == 'sex':
                    input_data[feature] = sex_map[st.selectbox(label, list(sex_map.keys()))]
                elif feature == 'cp':
                    input_data[feature] = cp_map[st.selectbox(label, list(cp_map.keys()))]
                elif feature == 'ca':
                    input_data[feature] = st.slider(label, 0, 4, 0)
                elif feature == 'thalach':
                    input_data[feature] = st.slider(label, 70, 210, 150)
                elif feature == 'exang':
                     input_data[feature] = exang_map[st.selectbox(label, list(exang_map.keys()))]
                elif feature == 'thal':
                    input_data[feature] = thal_map[st.selectbox(label, list(thal_map.keys()))]
                elif feature not in input_data:
                     input_data[feature] = st.slider(label, 0.0, 10.0, 1.0)
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