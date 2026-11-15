"""Download inputs for the review-response analyses (deviations D10 to D12).

Usage: python 02_code/10_download_isimip_gfed4.py isimip|gfed4 [n_parallel]
  isimip: ISIMIP3a 20CRv3-ERA5 obsclim and counterclim daily tasmax and tasmin, 0.5 degree, 1991 to 2021
          (https://files.isimip.org/ISIMIP3a/InputData/climate/atmosphere/<scenario>/global/daily/historical/20CRv3-ERA5/),
          with the per-file JSON metadata (sha512 checksum, licence) saved alongside.
  gfed4:  GFED4.1s monthly emissions, 2000 to 2023 (https://www.geo.vu.nl/~gwerf/GFED/GFED4/), 2017 onward beta.
Writes 01_data/raw/<source>/MANIFEST.csv (file, url, HTTP status, bytes, checksum verified, sha256, access time).
Resumes partial files.
"""
import csv, datetime, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISI = "https://files.isimip.org/ISIMIP3a/InputData/climate/atmosphere/{s}/global/daily/historical/20CRv3-ERA5/"
G4 = "https://www.geo.vu.nl/~gwerf/GFED/GFED4/"


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def fetch(url, path, sha512=None):
    status = None
    size = int(requests.head(url, timeout=60).headers.get("Content-Length", 0))
    for attempt in range(40):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if size and have == size:
            break
        try:
            with requests.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=120) as r:
                status = r.status_code
                if r.status_code not in (200, 206):
                    time.sleep(20); continue
                with open(path, "ab" if r.status_code == 206 else "wb") as f:
                    for c in r.iter_content(1 << 20):
                        f.write(c)
        except Exception as e:
            print(os.path.basename(path), "retry", attempt, e, flush=True); time.sleep(10)
    ok = os.path.exists(path) and os.path.getsize(path) == size
    if ok and sha512:
        ok = digest(path, "sha512") == sha512
    print(os.path.basename(path), "OK" if ok else "FAILED", flush=True)
    return {"file": os.path.basename(path), "url": url, "http_status": status or 200, "bytes": size,
            "checksum_verified": ok, "sha256": digest(path, "sha256") if ok else "",
            "access_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}


def main():
    src = sys.argv[1]
    npar = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out = os.path.join(ROOT, "01_data", "raw", "isimip3a" if src == "isimip" else "gfed4")
    os.makedirs(out, exist_ok=True)
    jobs = []
    if src == "isimip":
        for s in ("obsclim", "counterclim"):
            for v in ("tasmax", "tasmin"):
                for a, b in ((1991, 2000), (2001, 2010), (2011, 2020), (2021, 2021)):
                    name = f"20crv3-era5_{s}_{v}_global_daily_{a}_{b}"
                    meta = requests.get(ISI.format(s=s) + name + ".json", timeout=60).json()
                    json.dump(meta, open(os.path.join(out, name + ".json"), "w"), indent=1)
                    jobs.append((ISI.format(s=s) + name + ".nc", os.path.join(out, name + ".nc"), meta.get("checksum")))
    else:
        for y in range(2000, 2024):
            name = f"GFED4.1s_{y}{'_beta' if y >= 2017 else ''}.hdf5"
            jobs.append((G4 + name, os.path.join(out, name), None))
    with ThreadPoolExecutor(npar) as ex:
        rows = list(ex.map(lambda j: fetch(*j), jobs))
    with open(os.path.join(out, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
