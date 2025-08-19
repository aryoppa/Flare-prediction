import os
import cv2
import requests
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from tensorflow.keras.applications import EfficientNetB0, efficientnet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model, load_model
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Konfigurasi ---
IMG_SAVE_DIR = 'data/images_latest_24h'
FEATURE_SAVE_PATH = 'data/datasets/X_last_24h.npy'
MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_logscaled_ws2.h5'
# MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_logscaled_ws4.h5'
# MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_logscaled_ws5.h5'
# MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_classweight_ws2.h5'
# MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_classweight_ws4.h5'
# MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_classweight_ws5.h5'
# MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_ws2.h5'
# MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_ws4.h5'
# MODEL_PATH = 'final/trained_model/solarflare_tcn_oversampled_ws5.h5'
LABELS = ['No Flare', 'C', 'M', 'X']
IMG_SIZE = (512, 512)

# --- Fungsi download gambar ---
def download_image(image_url, save_path):
    try:
        if not os.path.exists(save_path):
            img_data = requests.get(image_url, timeout=10).content
            with open(save_path, 'wb') as f:
                f.write(img_data)
    except Exception as e:
        print(f"Gagal unduh {image_url}: {e}")

# --- Step 1: Download 24 gambar terakhir ---
def download_latest_images():
    os.makedirs(IMG_SAVE_DIR, exist_ok=True)
    now = datetime.utcnow()
    dates_to_check = list({(now - timedelta(hours=i)).date() for i in range(24)})
    dates_to_check.sort()
    downloaded = 0

    for date in dates_to_check:
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        base_url = f"https://sdo.gsfc.nasa.gov/assets/img/browse/{year}/{month}/{day}/"

        try:
            response = requests.get(base_url, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            image_links = sorted([
                a['href'] for a in soup.find_all('a')
                if a['href'].endswith('.jpg') and '_512_0171' in a['href']
            ])

            # Ambil satu gambar per jam
            chosen = []
            seen_hours = set()
            for link in image_links:
                hour_str = link.split('_')[1][:2]
                if hour_str not in seen_hours:
                    seen_hours.add(hour_str)
                    chosen.append(link)
                if len(chosen) + downloaded >= 24:
                    break

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for link in chosen:
                    image_url = base_url + link
                    save_path = os.path.join(IMG_SAVE_DIR, link)
                    futures.append(executor.submit(download_image, image_url, save_path))

                for _ in tqdm(as_completed(futures), total=len(futures), desc=f"{year}-{month}-{day}"):
                    pass

            downloaded += len(chosen)
            if downloaded >= 24:
                break

        except Exception as err:
            print(f"Error accessing {base_url}: {err}")

# --- Step 2: Grayscaling + Ekstraksi Fitur CNN ---
def extract_features_from_images():
    model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    feature_extractor = Model(inputs=model.input, outputs=model.output)

    features = []
    image_files = sorted([f for f in os.listdir(IMG_SAVE_DIR) if f.endswith('.jpg')])[:24]

    for img_file in tqdm(image_files, desc="🔍 Ekstraksi fitur"):
        img_path = os.path.join(IMG_SAVE_DIR, img_file)

        # Grayscaling menggunakan OpenCV
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.resize(gray_img, IMG_SIZE)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)  # convert grayscale to RGB 3-channel

        img_array = img_to_array(gray_img)
        img_array = efficientnet.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        feature_map = feature_extractor.predict(img_array)
        pooled_feature = np.mean(feature_map, axis=(1, 2))
        features.append(pooled_feature[0])

    X = np.array(features)  # shape: (24, feature_dim)
    os.makedirs(os.path.dirname(FEATURE_SAVE_PATH), exist_ok=True)
    np.save(FEATURE_SAVE_PATH, X)

# --- Step 3: Prediksi flare ---
def predict_flare():
    model = load_model(MODEL_PATH)
    X_input_full = np.load(FEATURE_SAVE_PATH)

    if X_input_full.shape[0] < 2:
        raise ValueError("Minimal dibutuhkan 2 sampel fitur untuk prediksi (karena window size = 2).")

    # Ambil 2 jam terakhir → shape (2, 1280)
    X_input = X_input_full[-2:, :]
    X_input = np.expand_dims(X_input, axis=0)  # shape (1, 2, 1280)

    y_pred = model.predict(X_input)
    predicted_class = np.argmax(y_pred, axis=1)[0]
    predicted_label = LABELS[predicted_class]

    print("\n=========== HASIL PREDIKSI SOLAR FLARE ===========")
    print(f"🌞 Prediksi Solar Flare yang paling mungkin terjadi (berdasarkan 24 jam terakhir): {predicted_label}")
    print("📊 Probabilitas Kelas:")
    for label, prob in zip(LABELS, y_pred[0]):
        print(f"   - {label:<8}: {prob:.4f}")
    print("==================================================\n")


# --- Eksekusi Pipeline ---
if __name__ == '__main__':
    # print("⏬ Mulai download citra 24 jam terakhir...")

    # print("\n⚙️  Ekstraksi fitur CNN (grayscaled)...")
    # extract_features_from_images()

    print("\n🔮 Prediksi flare...")
    predict_flare()
