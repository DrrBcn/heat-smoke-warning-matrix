"""Extract daily city series from the gridded inputs (area-weighted means over the urban centre polygon).

Usage:
  python 02_code/04_extract_city_series.py weights            -> 01_data/processed/weights_{fire,cpc}.parquet
  python 02_code/04_extract_city_series.py fire gfed|qfed     -> 01_data/processed/fire_<prod>.parquet
  python 02_code/04_extract_city_series.py temp               -> 01_data/processed/temp_cpc.parquet

Weights: for each city, every grid cell whose box intersects the urban centre polygon (WGS84) gets a
weight equal to the intersection area times cos(latitude), normalised to sum to 1. Cells with missing
values on a day are dropped and the remaining weights renormalised (CPC is land only, so coastal cities
lose sea cells). Polygons smaller than one cell still get the cell(s) they fall in.
Fire archives (.tar.gz) are streamed to a scratch file (overwritten each year, never kept in the project).
"""
import os, sys, tarfile, tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
from shapely.geometry import box

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "01_data", "raw")
PROC = os.path.join(ROOT, "01_data", "processed")
SCRATCH = os.environ.get("P44X_SCRATCH", tempfile.gettempdir())
YEARS_FIRE = range(2000, 2024)
YEARS_TEMP = range(1991, 2024)


def cell_weights(polys, lats, lons, res):
    """Return DataFrame uc_id, i_lat, i_lon, w for grid cell centres lats, lons (lon in -180..180)."""
    rows = []
    for uc, geom in zip(polys["uc_id"], polys.geometry):
        minx, miny, maxx, maxy = geom.bounds
        ii = np.where((lats >= miny - res) & (lats <= maxy + res))[0]
        jj = np.where((lons >= minx - res) & (lons <= maxx + res))[0]
        cand = []
        for i in ii:
            for j in jj:
                b = box(lons[j] - res / 2, lats[i] - res / 2, lons[j] + res / 2, lats[i] + res / 2)
                a = geom.intersection(b).area
                if a > 0:
                    cand.append((uc, i, j, a * np.cos(np.radians(lats[i]))))
        tot = sum(c[3] for c in cand)
        rows += [(c[0], c[1], c[2], c[3] / tot) for c in cand]
    return pd.DataFrame(rows, columns=["uc_id", "i_lat", "i_lon", "w"])


def weights():
    polys = gpd.read_file(os.path.join(PROC, "city_polygons.gpkg"))
    fire_lat = np.arange(-90, 90.001, 0.25)
    fire_lon = np.arange(-179.75, 180.001, 0.25)
    wf = cell_weights(polys, fire_lat, fire_lon, 0.25)
    wf.to_parquet(os.path.join(PROC, "weights_fire.parquet"))
    d = nc.Dataset(os.path.join(RAW, "temp_cpc", "tmax.2000.nc"))
    clat = np.asarray(d["lat"][:]); clon = np.asarray(d["lon"][:])
    clon180 = np.where(clon > 180, clon - 360, clon)
    order = np.argsort(clon180)
    wc = cell_weights(polys, clat, clon180[order], 0.5)
    wc["i_lon"] = order[wc["i_lon"].values]  # back to native index
    wc.to_parquet(os.path.join(PROC, "weights_cpc.parquet"))
    # fallback ring for CPC: the cells adjacent to the intersecting cells (equal weights), used only on
    # city-days when none of the intersecting cells has data (coastal cells masked as sea from 2006)
    ring = []
    for uc, g in wc.groupby("uc_id"):
        own = set(zip(g.i_lat, g.i_lon))
        nb = {((i + di) % len(clat), (j + dj) % len(clon)) for i, j in own for di in (-1, 0, 1) for dj in (-1, 0, 1)}
        nb -= own
        ring += [(uc, i, j, 1.0 / len(nb)) for i, j in sorted(nb)]
    pd.DataFrame(ring, columns=["uc_id", "i_lat", "i_lon", "w"]).to_parquet(os.path.join(PROC, "weights_cpc_ring.parquet"))
    print("fire cells", len(wf), "cpc cells", len(wc), "cities", wf.uc_id.nunique(), wc.uc_id.nunique())


def aggregate(arr, w, min_weight=0.5):
    """arr: (time, n_cells) values for the weight rows; returns (time, n_cities) weighted means.
    A city-day is missing when the available cells carry less than min_weight of the polygon weight
    (fire: 0.5; CPC: any cell with data, because the CPC land mask is coarser in early years)."""
    ucs = w["uc_id"].values
    uniq, inv = np.unique(ucs, return_inverse=True)
    vals = np.where(np.isfinite(arr), arr, 0.0)
    ok = np.isfinite(arr).astype(float)
    ww = w["w"].values[None, :]
    num = np.zeros((arr.shape[0], len(uniq))); den = np.zeros_like(num)
    for k in range(len(uniq)):
        m = inv == k
        num[:, k] = (vals[:, m] * ww[:, m]).sum(1)
        den[:, k] = (ok[:, m] * ww[:, m]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[(den < min_weight) | (den <= 0)] = np.nan
    return uniq, out


def fire(prod):
    w = pd.read_parquet(os.path.join(PROC, "weights_fire.parquet"))
    frames = []
    tmp = os.path.join(SCRATCH, f"fire_{prod}_tmp.nc")
    for y in YEARS_FIRE:
        arc = os.path.join(RAW, f"fire_{prod}", f"{y}{prod}firepm25.nc.tar.gz")
        with tarfile.open(arc, "r:gz") as t:
            m = [x for x in t.getmembers() if x.name.endswith(".nc")][0]
            with t.extractfile(m) as src, open(tmp, "wb") as dst:
                while True:
                    b = src.read(1 << 24)
                    if not b:
                        break
                    dst.write(b)
        d = nc.Dataset(tmp)
        lat = np.asarray(d["lat"][:]); lon = np.asarray(d["lon"][:])
        assert abs(lat[0] + 90) < 1e-6 and abs(lon[0] + 179.75) < 1e-6 and len(lat) == 721 and len(lon) == 1440
        dates = pd.to_datetime([s[:10] for s in d["time"][:]])
        v = d["PM25"]
        arr = np.empty((len(dates), len(w)), dtype=np.float64)
        for s in range(0, len(dates), 31):
            block = np.ma.filled(v[s:s + 31].astype(np.float64), np.nan)
            arr[s:s + 31] = block[:, w["i_lat"].values, w["i_lon"].values]
        d.close()
        nd = 366 if y % 4 == 0 else 365
        # consecutive days; full year except QFED 2000, which starts on 1 April 2000 (deviation D1)
        assert (np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1).all(), y
        assert len(dates) == nd or (prod == "qfed" and y == 2000 and str(dates[0].date()) == "2000-04-01"), y
        assert np.nanmin(arr) >= 0 and np.nanmax(arr) < 5000, (y, np.nanmin(arr), np.nanmax(arr))
        ucs, out = aggregate(arr, w)
        df = pd.DataFrame(out, index=dates, columns=ucs).stack(future_stack=True).rename("pm").reset_index()
        df.columns = ["date", "uc_id", "pm"]
        frames.append(df)
        print(prod, y, len(dates), "days", flush=True)
    pd.concat(frames).to_parquet(os.path.join(PROC, f"fire_{prod}.parquet"))


def temp():
    w = pd.read_parquet(os.path.join(PROC, "weights_cpc.parquet"))
    wr = pd.read_parquet(os.path.join(PROC, "weights_cpc_ring.parquet"))
    nfill = 0
    cov = []
    out = {}
    for var in ("tmax", "tmin"):
        frames = []
        for y in YEARS_TEMP:
            d = nc.Dataset(os.path.join(RAW, "temp_cpc", f"{var}.{y}.nc"))
            dates = pd.Timestamp("1900-01-01") + pd.to_timedelta(np.asarray(d["time"][:]), unit="h")
            a = d[var][:]
            a = np.ma.filled(a.astype(np.float64), np.nan)
            arr = a[:, w["i_lat"].values, w["i_lon"].values]
            arr[(arr < -90) | (arr > 60)] = np.nan
            ucs, agg = aggregate(arr, w, min_weight=0.0)
            arr_r = a[:, wr["i_lat"].values, wr["i_lon"].values]
            arr_r[(arr_r < -90) | (arr_r > 60)] = np.nan
            ucs_r, agg_r = aggregate(arr_r, wr, min_weight=0.0)
            assert (ucs_r == ucs).all()
            fill = np.isnan(agg) & np.isfinite(agg_r)
            nfill += int(fill.sum())
            if var == "tmax":  # own-cell weight with data (annual mean) and ring-filled days, per city-year
                okw = np.isfinite(arr) * w["w"].values[None, :]
                own = pd.DataFrame(okw.T).groupby(w["uc_id"].values).sum().T.mean(0)
                for k, uc in enumerate(ucs):
                    cov.append({"uc_id": uc, "year": y, "own_weight": float(own[uc]), "ring_days": int(fill[:, k].sum())})
            agg[fill] = agg_r[fill]
            frames.append(pd.DataFrame(agg, index=dates.normalize(), columns=ucs))
            d.close()
        out[var] = pd.concat(frames)
        print(var, "done; city-days filled from adjacent cells:", nfill, flush=True)
        if var == "tmax":
            pd.DataFrame(cov).to_csv(os.path.join(PROC, "cpc_coverage_log.csv"), index=False)
    df = out["tmax"].stack(future_stack=True).rename("tmax").to_frame().join(
        out["tmin"].stack(future_stack=True).rename("tmin")).reset_index()
    df.columns = ["date", "uc_id", "tmax", "tmin"]
    df = qc_temp(df)
    df.to_parquet(os.path.join(PROC, "temp_cpc.parquet"))


def qc_temp(df):
    """Set isolated spurious CPC values to missing (ANALYSIS_PLAN_V1_0 Section 6): a value is dropped when it
    deviates more than 15 C and more than 8 robust SDs (1.4826 x MAD) from the city's calendar-month median,
    and both values are dropped on days with tmin > tmax. Counts go to 01_data/processed/temp_qc_log.csv."""
    df["month"] = df["date"].dt.month
    log = []
    for var in ("tmax", "tmin"):
        g = df.groupby(["uc_id", "month"])[var]
        med = g.transform("median")
        mad = (df[var] - med).abs().groupby([df["uc_id"], df["month"]]).transform("median") * 1.4826
        dev = (df[var] - med).abs()
        bad = (dev > 15) & (dev > 8 * mad)
        log.append({"rule": f"{var} outlier", "n_values": int(bad.sum())})
        df.loc[bad, var] = np.nan
    inv = df["tmin"] > df["tmax"]
    log.append({"rule": "tmin > tmax", "n_values": int(inv.sum())})
    df.loc[inv, ["tmax", "tmin"]] = np.nan
    pd.DataFrame(log).to_csv(os.path.join(PROC, "temp_qc_log.csv"), index=False)
    print(log, flush=True)
    return df.drop(columns="month")


if __name__ == "__main__":
    {"weights": lambda: weights(), "fire": lambda: fire(sys.argv[2]), "temp": lambda: temp()}[sys.argv[1]]()
