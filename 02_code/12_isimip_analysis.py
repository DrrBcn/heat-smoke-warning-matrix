"""Parallel temperature analysis on ISIMIP3a 20CRv3-ERA5 (deviations D10 and D11; review comment 2).

  extract : city series of daily tasmax and tasmin (obsclim and counterclim, 1991 to 2021) from the 0.5 degree
            ISIMIP3a files, area-weighted over the urban centre polygon as for CPC (no land mask in ISIMIP)
            -> 01_data/processed/temp_isimip_<scenario>.parquet
  analyse : (a) ISIMIP obsclim in place of CPC for 2017 to 2021 (latest common five years) and 2015 to 2019,
            all 496 cities and the 453 primary cities, with CPC counted over the same windows for comparison;
            (b) the temperature step screen on ISIMIP obsclim and how many of the 43 CPC-flagged cities pass it;
            (c) counterclim (detrended, no-warming climate) against obsclim with fixed obsclim 1991-2020 city
            thresholds and observed fire PM2.5: share of activation days and fire-smoke conflict nights
            attributable to warming, primary product and each inventory.
Usage: python 02_code/12_isimip_analysis.py extract
       python 02_code/12_isimip_analysis.py analyse V0_3   (appends ISI_* tokens to P44x_NUMBERS_<v>.csv)
"""
import importlib.util, os, sys
import numpy as np
import pandas as pd
import netCDF4 as nc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "01_data", "raw", "isimip3a")
PROC = os.path.join(ROOT, "01_data", "processed")
RES = os.path.join(ROOT, "03_results")
spec = importlib.util.spec_from_file_location("ana", os.path.join(ROOT, "02_code", "06_analysis.py"))
ana = importlib.util.module_from_spec(spec); spec.loader.exec_module(ana)
spec4 = importlib.util.spec_from_file_location("ext", os.path.join(ROOT, "02_code", "04_extract_city_series.py"))
ext = importlib.util.module_from_spec(spec4); spec4.loader.exec_module(ext)
DEC = [(1991, 2000), (2001, 2010), (2011, 2020), (2021, 2021)]


def extract():
    import geopandas as gpd
    polys = gpd.read_file(os.path.join(PROC, "city_polygons.gpkg"))
    lat = np.arange(89.75, -90, -0.5); lon = np.arange(-179.75, 180, 0.5)
    order = np.argsort(lat)
    w = ext.cell_weights(polys, lat[order], lon, 0.5)
    w["i_lat"] = order[w["i_lat"].values]
    for scen in ("obsclim", "counterclim"):
        out = {}
        for var in ("tasmax", "tasmin"):
            frames = []
            for a, b in DEC:
                d = nc.Dataset(os.path.join(RAW, f"20crv3-era5_{scen}_{var}_global_daily_{a}_{b}.nc"))
                assert abs(d["lat"][0] - 89.75) < 1e-6 and abs(d["lon"][0] + 179.75) < 1e-6
                t = pd.Timestamp("1900-01-01") + pd.to_timedelta(np.asarray(d["time"][:]), unit="D")
                v = d[var]; arr = np.empty((len(t), len(w)))
                for s in range(0, len(t), 366):
                    blk = np.ma.filled(v[s:s + 366].astype(np.float64), np.nan)
                    arr[s:s + 366] = blk[:, w["i_lat"].values, w["i_lon"].values] - 273.15
                d.close()
                ucs, agg = ext.aggregate(arr, w, min_weight=0.5)
                frames.append(pd.DataFrame(agg, index=t.normalize(), columns=ucs))
                print(scen, var, a, flush=True)
            out[var] = pd.concat(frames)
        df = out["tasmax"].stack(future_stack=True).rename("tmax").to_frame().join(
            out["tasmin"].stack(future_stack=True).rename("tmin")).reset_index()
        df.columns = ["date", "uc_id", "tmax", "tmin"]
        df.to_parquet(os.path.join(PROC, f"temp_isimip_{scen}.parquet"))


def analyse(v):
    frame, days, tdays, data = ana.load()
    ids = frame["uc_id"].values
    tdi = pd.date_range("1991-01-01", "2021-12-31"); di = pd.date_range("2000-01-01", "2021-12-31")
    off = (di[0] - tdi[0]).days; nday = len(di)

    def grid(df, col):
        return df.pivot(index="date", columns="uc_id", values=col).reindex(index=tdi, columns=ids).values.T
    iso = {}
    for scen in ("obsclim", "counterclim"):
        t = pd.read_parquet(os.path.join(PROC, f"temp_isimip_{scen}.parquet")); t["date"] = pd.to_datetime(t["date"]).dt.normalize()
        iso[scen] = {"tmax_all": grid(t, "tmax"), "tmin_all": grid(t, "tmin")}
    ct = pd.read_csv(os.path.join(RES, f"city_table_{v}.csv"))
    prim = frame["uc_id"].isin(ct["uc_id"]).values                 # 453 primary cities
    allc = frame["uc_id"].isin(pd.read_csv(os.path.join(RES, f"city_table_all_{v}.csv"))["uc_id"]).values  # 496
    pop = frame["pop2020"].values
    yrs = di.year.values
    toks = []

    def tok(t, val, unit, d=2):
        toks.append({"token": t, "value": (f"{val:,.{d}f}" if d > 0 else f"{int(round(val)):,}") if isinstance(val, (float, np.floating)) and np.isfinite(val) else val,
                     "unit": unit, "script": "02_code/12_isimip_analysis.py", "inputs": "ISIMIP3a 20CRv3-ERA5, fire PM2.5, city frame"})

    def pct_from(tmax_all, tdx):
        m = (tdx >= "1991-01-01") & (tdx <= "2020-12-31")
        return {p: np.nanpercentile(tmax_all[:, m], p, axis=1) for p in (90, 95, 97.5)}

    def run(tmax_all, tmin_all, pct, product="mean"):
        d = {"tmax": tmax_all[:, off:off + nday], "tmin": tmin_all[:, off:off + nday],
             product: data[product][:, :nday]}
        f, *_ = ana.daily_flags({**ana.PRIMARY, "product": product}, d, pct)
        return f

    def pw(flag, sel, y0, y1):
        m = (yrs >= y0) & (yrs <= y1)
        per = flag[sel][:, m].sum(1) / (y1 - y0 + 1)
        return (per * pop[sel]).sum() / pop[sel].sum(), per

    # (a) ISIMIP obsclim against CPC over common windows
    pct_o = pct_from(iso["obsclim"]["tmax_all"], tdi)
    fo = run(iso["obsclim"]["tmax_all"], iso["obsclim"]["tmin_all"], pct_o)
    cpc_pct = ana.percentiles(data, tdays, "1991-2020")
    toff = (di[0] - tdays[0]).days
    fc = run(data["tmax_all"][:, :len(tdi)], data["tmin_all"][:, :len(tdi)], cpc_pct)
    for name, sel in (("PRIM", prim), ("ALL", allc)):
        for k in ("activation", "conflict"):
            for (y0, y1) in ((2017, 2021), (2015, 2019)):
                vi, per_i = pw(fo[k], sel, y0, y1); vc, per_c = pw(fc[k], sel, y0, y1)
                tok(f"ISI_{name}_{k.upper()}_{y0}", vi, f"days per resident per year {y0}-{y1}, ISIMIP3a obsclim")
                tok(f"CPC_{name}_{k.upper()}_{y0}", vc, f"days per resident per year {y0}-{y1}, CPC")
            vi, per_i = pw(fo[k], sel, 2017, 2021)
            tok(f"ISI_{name}_{k.upper()}_NGE1", int((per_i >= 1).sum()), "cities with >= 1 a year, 2017-2021, ISIMIP3a", 0)
        # trend contrast under ISIMIP (2015-2019 against 2000-2004)
        for k in ("heat", "activation", "conflict"):
            e, _ = pw(fo[k], sel, 2000, 2004); p_, _ = pw(fo[k], sel, 2015, 2019)
            tok(f"ISI_{name}_{k.upper()}_RATIO_PRE", p_ / e if e else np.nan, "ratio 2015-2019 / 2000-2004, ISIMIP3a obsclim")
    # regional and income values for all cities with the reanalysis, 2017-2021 (review of V0_3, comment 2)
    for col in ("who_region", "income_group"):
        for g_, sub in frame[allc].groupby(col):
            m = frame["uc_id"].isin(sub["uc_id"]).values & allc
            G = str(g_).upper().replace(" ", "_").replace("-", "_")
            tok(f"ISI_ALL_GRP_{G}_N", int(m.sum()), "cities", 0)
            for k in ("activation", "conflict"):
                vv, _ = pw(fo[k], m, 2017, 2021)
                tok(f"ISI_ALL_GRP_{G}_{k.upper()}", vv, "days per resident per year 2017-2021, reanalysis, all cities")
            vp, _ = pw(fc["activation"], m & prim, 2017, 2021)
            tok(f"CPC_PRIM_GRP_{G}_ACTIVATION_2017", vp, "days per resident per year 2017-2021, CPC, primary cities")
    # daily agreement with CPC
    a = iso["obsclim"]["tmax_all"][allc]; b = data["tmax_all"][allc][:, :len(tdi)]
    r = [np.corrcoef(x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)])[0, 1] for x, y in zip(a, b)]
    tok("ISI_CPC_TMAX_R_MED", float(np.nanmedian(r)), "median daily correlation of Tmax, ISIMIP3a and CPC")
    tok("ISI_CPC_TMAX_BIAS_MED", float(np.nanmedian(np.nanmean(b - a, 1))), "median bias CPC minus ISIMIP3a, C", 1)
    # (b) step screen on ISIMIP obsclim (same rule as D4)
    ty = tdi.year.values; yy = np.arange(1991, 2022)
    ann = np.stack([np.nanmean(iso["obsclim"]["tmax_all"][:, ty == y], 1) for y in yy], 1)
    blocks = [(1991, 1995), (1996, 2000), (2001, 2005), (2006, 2010), (2011, 2015), (2016, 2020), (2017, 2021)]
    bm = np.stack([np.nanmean(ann[:, (yy >= a_) & (yy <= b_)], 1) for a_, b_ in blocks], 1)
    jump = np.nanmax(np.abs(np.diff(bm, axis=1)), 1)
    er = np.nanmean(ann[:, yy >= 2017], 1) - np.nanmean(ann[:, (yy >= 2000) & (yy <= 2004)], 1)
    st = (jump > 2.0) | (np.abs(er) > 2.0)
    flagged_cpc = allc & ~prim
    tok("ISI_STEP_FLAGGED_ALL", int((allc & st).sum()), "cities flagged by the step rule on ISIMIP3a", 0)
    tok("ISI_STEP_RETURN", int((flagged_cpc & ~st).sum()), "of the CPC step-flagged cities, number passing on ISIMIP3a", 0)
    tok("ISI_TMAX_CHANGE_MEDIAN", float(np.nanmedian(er[allc])), "C, median change in annual mean Tmax, 2000-2004 to 2017-2021", 1)
    # (c) counterfactual (no-warming) climate with fixed obsclim thresholds
    for prod, tag in (("mean", ""), ("gfed", "_GFED"), ("qfed", "_QFED")):
        fo_p = run(iso["obsclim"]["tmax_all"], iso["obsclim"]["tmin_all"], pct_o, prod)
        fcf = run(iso["counterclim"]["tmax_all"], iso["counterclim"]["tmin_all"], pct_o, prod)
        for name, sel in (("PRIM", prim), ("ALL", allc)):
            for k in ("activation", "conflict"):
                for (y0, y1) in ((2015, 2019), (2017, 2021)):
                    o, _ = pw(fo_p[k], sel, y0, y1); c_, _ = pw(fcf[k], sel, y0, y1)
                    tok(f"ISI_CF{tag}_{name}_{k.upper()}_{y0}", c_, f"days per resident per year without warming, {y0}-{y1}")
                    tok(f"ISI_ADD{tag}_{name}_{k.upper()}_{y0}", 100 * (o / c_ - 1) if c_ else np.nan, f"% added by warming, {y0}-{y1}", 0)
                    tok(f"ISI_AF{tag}_{name}_{k.upper()}_{y0}", 100 * (1 - c_ / o) if o else np.nan, f"% of days attributable to warming, {y0}-{y1}", 0)
    tdiff = np.nanmean(iso["obsclim"]["tmax_all"][allc][:, (ty >= 2015) & (ty <= 2019)] - iso["counterclim"]["tmax_all"][allc][:, (ty >= 2015) & (ty <= 2019)], 1)
    tok("ISI_OBS_MINUS_CF_TMAX_MED", float(np.nanmedian(tdiff)), "C, median obsclim minus counterclim Tmax, 2015-2019", 1)
    nf = os.path.join(RES, f"P44x_NUMBERS_{v}.csv")
    n = pd.read_csv(nf)
    n = n[~n.token.str.match(r"^(ISI_|CPC_PRIM|CPC_ALL)")]
    pd.concat([n, pd.DataFrame(toks)]).to_csv(nf, index=False)
    print(pd.DataFrame(toks)[["token", "value"]].to_string())


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract()
    else:
        analyse(sys.argv[2] if len(sys.argv) > 2 else "V0_3")
