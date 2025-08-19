import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# --- KONFIGURASI ---
FEATURE_SAVE_PATH = 'data/datasets/X_last_24h.npy'
MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_logscaled_ws2.h5'  # Pilih salah satu model
LABELS = ['No Flare', 'B', 'C', 'M', 'X']  # Urutan HARUS sama seperti training!
WINDOW_SIZE = 2  # Ganti sesuai model yang dipilih

# --- PREDIKSI 24 JAM KE DEPAN ---
def predict_flare_per_hour():
    X_input_full = np.load(FEATURE_SAVE_PATH)
    model = load_model(MODEL_PATH)

    results = []

    # Pastikan cukup data
    n_pred = X_input_full.shape[0] - WINDOW_SIZE + 1
    for i in range(n_pred):
        X_window = X_input_full[i:i+WINDOW_SIZE, :]
        X_window = np.expand_dims(X_window, axis=0)
        y_pred = model.predict(X_window)
        predicted_class = np.argmax(y_pred, axis=1)[0]
        predicted_label = LABELS[predicted_class]
        probas = {f"prob_{label}": float(y_pred[0][idx]) for idx, label in enumerate(LABELS)}

        results.append({
            "hour_start": i,
            "hour_end": i+WINDOW_SIZE-1,
            "predicted_label": predicted_label,
            **probas
        })

    df = pd.DataFrame(results)
    print("\n=== PREDIKSI KEJADIAN FLARE PER JAM (24 JAM KE DEPAN) ===\n")
    print(df[["hour_start", "hour_end", "predicted_label"] + [f"prob_{l}" for l in LABELS]].to_string(index=False))
    print("\n==========================================================\n")
    df.to_csv('prediksi_per_jam_24jam.csv', index=False)
    return df

if __name__ == '__main__':
    predict_flare_per_hour()
