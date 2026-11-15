"""Download NOAA CPC Global Unified daily Tmax and Tmin (0.5 degree) yearly files, 1991 to 2023.

Usage: python 02_code/02_download_temperature.py
Writes 01_data/raw/temp_cpc/{tmax,tmin}.YYYY.nc and MANIFEST.csv (url, http status, bytes, sha256, access time).
"""
import csv, datetime, hashlib, os
from concurrent.futures import ThreadPoolExecutor
import requests

BASE = "https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/"
OUT = os.path.join(os.path.dirname(__file__), "..", "01_data", "raw", "temp_cpc")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def get(name):
    url = BASE + name
    path = os.path.join(OUT, name)
    for _ in range(10):
        try:
            head = requests.head(url, timeout=60)
            size = int(head.headers.get("Content-Length", 0))
            if os.path.exists(path) and os.path.getsize(path) == size:
                status = head.status_code
                break
            r = requests.get(url, timeout=300)
            status = r.status_code
            if status == 200:
                open(path, "wb").write(r.content)
                if os.path.getsize(path) == size:
                    break
        except Exception as e:
            print(name, e, flush=True)
    print(name, status, flush=True)
    return {"file": name, "url": url, "http_status": status, "bytes": os.path.getsize(path),
            "sha256": sha(path), "access_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = [f"{v}.{y}.nc" for v in ("tmax", "tmin") for y in range(1991, 2024)]
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(get, names))
    with open(os.path.join(OUT, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
