"""Download the small static inputs: GHSL Urban Centre Database R2024A V1_2, World Bank income groups, WHO regions,
Natural Earth land outline. The large inputs have their own scripts (01, 02, 05, 10).

Usage: python 02_code/00_download_inputs.py
Writes into 01_data/raw/{ghsl_ucdb, groups, naturalearth}/ and unzips the GHSL and Natural Earth archives.
"""
import os, zipfile
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "01_data", "raw")
FILES = [
    ("ghsl_ucdb", "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_GLOBE_R2024A/V1-2/GHS_UCDB_GLOBE_R2024A_V1_2.zip", True),
    ("groups", "https://datacatalogfiles.worldbank.org/ddh-published/0037712/DR0090755/CLASS.xlsx", False),
    ("groups", "https://ghoapi.azureedge.net/api/DIMENSION/COUNTRY/DimensionValues", False),
    ("naturalearth", "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip", True),
]


def main():
    for sub, url, unzip in FILES:
        d = os.path.join(RAW, sub); os.makedirs(d, exist_ok=True)
        name = "who_gho_countries.json" if "ghoapi" in url else url.rsplit("/", 1)[-1]
        path = os.path.join(d, name)
        r = requests.get(url, timeout=600); r.raise_for_status()
        open(path, "wb").write(r.content)
        print(name, r.status_code, len(r.content))
        if unzip:
            zipfile.ZipFile(path).extractall(d)


if __name__ == "__main__":
    main()
