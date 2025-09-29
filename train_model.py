import joblib
from processing import process_and_train 

print("Memulai proses training model...")

# Memanggil fungsi untuk mendapatkan semua objek yang sudah dilatih
model, reports, accuracy, scaler, selector, selected_features = process_and_train()

if model:
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(selector, 'selector.joblib')
    
    # List fitur yang terpilih 
    with open('selected_features.txt', 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    print("\n[✓] Model dan objek lainnya berhasil disimpan!")
    print(f"    - model.joblib")
    print(f"    - scaler.joblib")
    print(f"    - selector.joblib")
    print(f"    - selected_features.txt")
    print(f"\nAkurasi model yang disimpan: {accuracy*100:.2f}%")
else:
    print(f"\n[X] Proses Gagal: {reports}")