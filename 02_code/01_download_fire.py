"""Download the Hu et al. 2025 fire-sourced PM2.5 yearly archives from Zenodo.

Usage: python 02_code/01_download_fire.py gfed|qfed [n_parallel]
Resumes partial files, verifies md5 against the Zenodo record, writes a
manifest CSV (file, size, md5, sha256, http status, access time) next to the files.
"""
import csv, hashlib, json, os, sys, time, datetime
from concurrent.futures import ThreadPoolExecutor
import requests

RECORDS = {"gfed": "15493914", "qfed": "15496596"}
ROOT = os.path.join(os.path.dirname(__file__), "..", "01_data", "raw")


def md5sum(path, algo="md5"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(fileinfo, outdir):
    key = fileinfo["key"]
    url = fileinfo["links"]["self"]
    size = fileinfo["size"]
    md5 = fileinfo["checksum"].split(":")[1]
    path = os.path.join(outdir, key)
    status = None
    for attempt in range(30):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                status = r.status_code
                if r.status_code not in (200, 206):
                    time.sleep(30)
                    continue
                mode = "ab" if r.status_code == 206 else "wb"
                with open(path, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except Exception as e:  # network hiccup: resume
            print(key, "retry", attempt, e, flush=True)
            time.sleep(15)
    ok = os.path.exists(path) and os.path.getsize(path) == size and md5sum(path) == md5
    print(key, "OK" if ok else "FAILED", flush=True)
    return {"file": key, "url": url, "http_status": status or 200, "bytes": size,
            "md5_zenodo": md5, "md5_ok": ok,
            "sha256": md5sum(path, "sha256") if ok else "",
            "access_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")}


def main():
    prod = sys.argv[1]
    npar = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    outdir = os.path.join(ROOT, f"fire_{prod}")
    os.makedirs(outdir, exist_ok=True)
    rec = requests.get(f"https://zenodo.org/api/records/{RECORDS[prod]}", timeout=60).json()
    with open(os.path.join(outdir, "zenodo_record.json"), "w") as f:
        json.dump(rec, f, indent=1)
    files = sorted(rec["files"], key=lambda x: x["key"])
    with ThreadPoolExecutor(npar) as ex:
        rows = list(ex.map(lambda fi: fetch(fi, outdir), files))
    with open(os.path.join(outdir, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
