import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Konfigurasi direktori simpan lokal ---
save_dir = './data/images_latest_24h'
os.makedirs(save_dir, exist_ok=True)

# --- Fungsi download gambar ---
def download_image(image_url, save_path):
    try:
        if not os.path.exists(save_path):
            img_data = requests.get(image_url, timeout=10).content
            with open(save_path, 'wb') as f:
                f.write(img_data)
    except Exception as e:
        print(f"Gagal unduh {image_url}: {e}")

# --- Hitung 24 jam terakhir ---
now = datetime.utcnow()
dates_to_check = list({(now - timedelta(hours=i)).date() for i in range(24)})
dates_to_check.sort()

# --- Simpan gambar yang terpilih ---
downloaded = 0

for date in dates_to_check:
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    base_url = f"https://sdo.gsfc.nasa.gov/assets/img/browse/{year}/{month}/{day}/"

    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code != 200:
            print(f"Lewatkan {base_url} (status {response.status_code})")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Ambil semua link gambar AIA 171 dengan resolusi 512
        image_links = sorted([
            a['href'] for a in soup.find_all('a')
            if a['href'].endswith('.jpg') and '_512_0171' in a['href']
        ])

        # Pilih hanya satu gambar per jam (jam ke-00, 01, ..., 23)
        chosen = []
        seen_hours = set()
        for link in image_links:
            # Contoh nama file: 20250727_000029_512_0171.jpg
            hour_str = link.split('_')[1][:2]
            if hour_str not in seen_hours:
                seen_hours.add(hour_str)
                chosen.append(link)
            if len(chosen) + downloaded >= 24:
                break

        # Download dengan multithread
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for link in chosen:
                image_url = base_url + link
                save_path = os.path.join(save_dir, link)
                futures.append(executor.submit(download_image, image_url, save_path))

            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"{year}-{month}-{day}"):
                pass

        downloaded += len(chosen)
        if downloaded >= 24:
            break

    except Exception as err:
        print(f"Error accessing {base_url}: {err}")

print(f"\nTotal gambar berhasil diunduh: {downloaded}")
