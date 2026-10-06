import os
import zipfile
import urllib.request

URL = "https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set/archive/refs/heads/master.zip"

DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "uwb_dataset.zip")

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading dataset...")

urllib.request.urlretrieve(URL, ZIP_PATH)

print("Download complete.")

print("Extracting...")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(DATA_DIR)

print("Dataset extracted.")

os.remove(ZIP_PATH)

print("Done.")