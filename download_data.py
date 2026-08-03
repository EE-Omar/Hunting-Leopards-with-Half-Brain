"""Downloads and extracts the project dataset from Google Drive into data/."""

import os
import zipfile
import gdown
from tqdm import tqdm

FILE_ID = "1QUDgBldH_3ee5X7VhKStMr-L430aUlQ_"
ZIP_PATH = "data.zip"
DATA_DIR = "data"
DATASET_DIR = os.path.join(DATA_DIR, "v2_yolo_dataset")

if os.path.exists(DATASET_DIR):
    print(" \"data/v2_yolo_dataset\" folder already exists.")
    print("Quitting ...")
    exit()
else:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading dataset...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for file in tqdm(zf.namelist(), desc="Extracting", unit="file"):
            zf.extract(file, DATA_DIR)

    os.remove(ZIP_PATH)
    print("Done.")