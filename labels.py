import requests
import os
import pandas as pd

# The Zenodo link (records/3565489) is currently returning 404.
# Using the confirmed S3 backup link for Hart et al. (2016) labels instead.
url = "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz"
filename = "gz2_hart16.csv.gz"

if not os.path.exists(filename):
    print(f"Downloading {filename} (streaming)...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Labels downloaded successfully from S3 (via streaming).")
else:
    print(f"{filename} already exists.")

# Peek at what we have - pandas handles .gz automatically
print("Reading CSV data into memory...")
df = pd.read_csv(filename)
print(f"Data shape: {df.shape}")
print("Column headers (first 10):")
print(df.columns.tolist()[:10])
print("\nFirst 3 rows:")
print(df.head(3))
