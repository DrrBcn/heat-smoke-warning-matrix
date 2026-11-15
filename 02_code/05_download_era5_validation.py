"""Download ERA5 daily Tmax and Tmin for 20 validation cities (2019 to 2023) from the Open-Meteo archive API.

Sample: 20 primary-frame cities, stratified by WHO region (about 3 per region, then filled at random),
numpy seed 44, as fixed in ANALYSIS_PLAN_V1_0 Section 2. Point = polygon representative point.
Writes 01_data/raw/era5_openmeteo/<uc_id>.json and MANIFEST.csv (url, status, sha256, time).
Open-Meteo data licence: CC BY 4.0 (https://open-meteo.com/en/license). No key needed.
"""
import csv, datetime, hashlib, json, os, time
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "01_data", "raw", "era5_openmeteo")
API = "https://archive-api.open-meteo.com/v1/archive"


def sample():
    f = pd.read_csv(os.path.join(ROOT, "01_data", "processed", "city_frame.csv"))
    f = f[f["primary"]].sort_values("uc_id")
    rng = np.random.default_rng(44)
    picks = []
    for reg, g in f.groupby("who_region"):
        picks += list(rng.choice(g["uc_id"].values, size=3, replace=False))
    rest = f[~f["uc_id"].isin(picks)]
    picks += list(rng.choice(rest["uc_id"].values, size=20 - len(picks), replace=False))
    return f[f["uc_id"].isin(picks)]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for _, c in sample().iterrows():
        params = {"latitude": round(c["lat"], 4), "longitude": round(c["lon"], 4), "start_date": "2019-01-01",
                  "end_date": "2023-12-31", "daily": "temperature_2m_max,temperature_2m_min", "models": "era5",
                  "timezone": "GMT"}
        for attempt in range(5):
            r = requests.get(API, params=params, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(65)
        path = os.path.join(OUT, f"{int(c['uc_id'])}.json")
        open(path, "wb").write(r.content)
        rows.append({"uc_id": int(c["uc_id"]), "city": c["city"], "url": r.url, "http_status": r.status_code,
                     "sha256": hashlib.sha256(r.content).hexdigest(),
                     "access_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")})
        print(c["city"], r.status_code, flush=True)
        time.sleep(3)
    with open(os.path.join(OUT, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
