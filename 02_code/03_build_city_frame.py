"""Build the city frame: GHSL UCDB R2024A V1_2 urban centres with population >= 1 million in 2020
(primary frame, column primary=True) plus national capitals of any size (sensitivity frame).

Outputs 01_data/processed/city_frame.csv and city_polygons.gpkg (WGS84) with WHO region and
World Bank income group (FY classification file in 01_data/raw/groups).
"""
import json, os
import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "01_data", "raw")
OUT = os.path.join(ROOT, "01_data", "processed")
GPKG = os.path.join(RAW, "ghsl_ucdb", "GHS_UCDB_GLOBE_R2024A.gpkg")
POP_MIN = 1_000_000
POP_FIELD = "GH_POP_TOT_2020"

# GADM country names in the UCDB that do not match World Bank economy names
NAME_FIX = {
    "Democratic Republic of the Congo": "COD", "Republic of the Congo": "COG", "Côte d'Ivoire": "CIV",
    "Russia": "RUS", "Iran": "IRN", "Egypt": "EGY", "Syria": "SYR", "Venezuela": "VEN", "Vietnam": "VNM",
    "South Korea": "KOR", "North Korea": "PRK", "Laos": "LAO", "Turkey": "TUR", "Türkiye": "TUR", "Yemen": "YEM",
    "Tanzania": "TZA", "Kyrgyzstan": "KGZ", "Slovakia": "SVK", "Czech Republic": "CZE", "Czechia": "CZE",
    "Palestine": "PSE", "Gambia": "GMB", "Bahamas": "BHS", "Hong Kong": "HKG", "Macao": "MAC", "Taiwan": "TWN",
    "Micronesia": "FSM", "Saint Lucia": "LCA", "Somalia": "SOM", "Swaziland": "SWZ", "Eswatini": "SWZ",
    "Myanmar": "MMR", "Mexico": "MEX", "México": "MEX", "Cape Verde": "CPV", "Macedonia": "MKD", "North Macedonia": "MKD", "Moldova": "MDA",
    "Brunei": "BRN", "Bolivia": "BOL", "Kosovo": "XKX", "Puerto Rico": "PRI", "United States": "USA",
    "United Kingdom": "GBR", "Guinea-Bissau": "GNB", "East Timor": "TLS", "Timor-Leste": "TLS",
}
# WHO regions for economies absent from the WHO GHO country list (territories)
WHO_EXTRA = {"HKG": "WPR", "MAC": "WPR", "TWN": "WPR", "PRI": "AMR", "XKX": "EUR", "PSE": "EMR"}


def demojibake(s):
    """Some UCDB names are UTF-8 read as Latin-1 (for example 'MÃ©xico'); repair them."""
    try:
        return s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
        return s


def main():
    os.makedirs(OUT, exist_ok=True)
    g = gpd.read_file(GPKG, layer="GHSL_UCDB_THEME_GHSL_GLOBE_R2024A",
                      columns=["ID_UC_G0", "GC_UCN_MAI_2025", "GC_CNT_GAD_2025", "GC_UCA_KM2_2025", POP_FIELD,
                               "GH_POP_TOT_2000", "GH_POP_TOT_2005", "GH_POP_TOT_2010", "GH_POP_TOT_2015",
                               "GH_POP_TOT_2025", "GC_POP_TOT_2025"])
    cap = gpd.read_file(GPKG, layer="GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A",
                        columns=["ID_UC_G0", "GC_UCM_CAP", "GC_UCN_LIS_2025"], ignore_geometry=True)
    cap["GC_UCN_LIS_2025"] = cap["GC_UCN_LIS_2025"].map(demojibake)
    g = g.merge(cap, on="ID_UC_G0", how="left")
    # annual population-weighted total PM2.5 (ACAG V6.GL.02.02, UCDB emissions theme), for background stratification
    pmt = gpd.read_file(GPKG, layer="GHSL_UCDB_THEME_EMISSIONS_GLOBE_R2024A",
                        columns=["ID_UC_G0", "EM_PM2_CON_2015", "EM_PM2_CON_2020"], ignore_geometry=True)
    g = g.merge(pmt.rename(columns={"EM_PM2_CON_2015": "pm25_total_2015", "EM_PM2_CON_2020": "pm25_total_2020"}),
                on="ID_UC_G0", how="left")
    g["capital"] = g["GC_UCM_CAP"].fillna(0).astype(int) == 1
    g["primary"] = g[POP_FIELD] >= POP_MIN
    g = g[g["primary"] | g["capital"]].drop(columns="GC_UCM_CAP").copy()
    for c in ("GC_UCN_MAI_2025", "GC_CNT_GAD_2025"):
        g[c] = g[c].map(demojibake)
    wb = pd.read_excel(os.path.join(RAW, "groups", "CLASS.xlsx")).dropna(subset=["Code"])
    name2iso = dict(zip(wb["Economy"], wb["Code"]))
    g["iso3"] = g["GC_CNT_GAD_2025"].map(lambda n: NAME_FIX.get(n, name2iso.get(n)))
    who = json.load(open(os.path.join(RAW, "groups", "who_gho_countries.json"), encoding="utf-8"))["value"]
    iso2who = {d["Code"]: d["ParentCode"] for d in who}
    iso2who.update(WHO_EXTRA)
    g["who_region"] = g["iso3"].map(iso2who).fillna("Unassigned")  # only non-sovereign capitals (sensitivity frame)
    g["income_group"] = g["iso3"].map(dict(zip(wb["Code"], wb["Income group"]))).fillna("Unclassified")
    g = g.to_crs(4326)
    cen = g.geometry.representative_point()
    g["lon"], g["lat"] = cen.x, cen.y
    g = g.rename(columns={"ID_UC_G0": "uc_id", "GC_UCN_MAI_2025": "city", "GC_CNT_GAD_2025": "country",
                          "GC_UCA_KM2_2025": "area_km2", "GC_UCN_LIS_2025": "names_list", POP_FIELD: "pop2020", "GH_POP_TOT_2000": "pop2000", "GH_POP_TOT_2005": "pop2005",
                          "GH_POP_TOT_2010": "pop2010", "GH_POP_TOT_2015": "pop2015", "GH_POP_TOT_2025": "pop2025",
                          "GC_POP_TOT_2025": "pop2025_gc"})
    unmatched = g[g["iso3"].isna() | (g["who_region"] == "Unassigned")]["country"].unique().tolist()
    print("cities", len(g), "primary", int(g["primary"].sum()), "capitals", int(g["capital"].sum()), "unmatched countries", unmatched)
    g.to_file(os.path.join(OUT, "city_polygons.gpkg"), driver="GPKG")
    g.drop(columns="geometry").to_csv(os.path.join(OUT, "city_frame.csv"), index=False)


if __name__ == "__main__":
    main()
