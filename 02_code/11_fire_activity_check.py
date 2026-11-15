"""Fire-activity check of the fire-sourced PM2.5 trend (deviation D12, review comment 3).

GFED4.1s monthly dry-matter (DM) emissions are an input to the fire model and do not pass through the monitor-based
correction of the PM2.5 product. For each primary city, annual DM emissions (kg) are summed over grid cells within
+/-2.5 degrees (sensitivity +/-5 degrees) of the city point, 2000 to 2023 (2017 onward are GFED beta files), and
compared with the annual mean fire-sourced PM2.5 of the city.

Usage: python 02_code/11_fire_activity_check.py V0_3
Outputs: 03_results/fire_activity_check_<v>.csv (city, ratios), tokens appended to P44x_NUMBERS_<v>.csv.
"""
import os, sys
import numpy as np
import pandas as pd
import h5py
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "01_data", "raw", "gfed4")
PROC = os.path.join(ROOT, "01_data", "processed")
RES = os.path.join(ROOT, "03_results")
YEARS = np.arange(2000, 2024)


def main():
    v = sys.argv[1] if len(sys.argv) > 1 else "V0_3"
    ct = pd.read_csv(os.path.join(RES, f"city_table_{v}.csv"))   # primary frame of the analysis
    f0 = h5py.File(os.path.join(RAW, "GFED4.1s_2000.hdf5"), "r")
    lat = f0["lat"][:, 0]; lon = f0["lon"][0, :]; area = f0["ancill/grid_cell_area"][:]
    em = {}
    for y in YEARS:
        fn = f"GFED4.1s_{y}{'_beta' if y >= 2017 else ''}.hdf5"
        with h5py.File(os.path.join(RAW, fn), "r") as f:
            em[y] = sum(f[f"emissions/{m:02d}/DM"][:] for m in range(1, 13)) * area   # kg DM per cell per year
    out = []
    for _, c in ct.iterrows():
        row = {"uc_id": c.uc_id, "city": c.city, "country": c.country, "who_region": c.who_region, "pop2020": c.pop2020}
        for r in (2.5, 5.0):
            mi = (np.abs(lat - c.lat) <= r); mj = (np.abs(((lon - c.lon + 180) % 360) - 180) <= r)
            ser = np.array([em[y][np.ix_(mi, mj)].sum() for y in YEARS])
            for y, val in zip(YEARS, ser):
                row[f"dm{int(r * 10)}_{y}"] = val
        out.append(row)
    df = pd.DataFrame(out)
    # fire PM2.5 annual means (primary mean of both inventories, with the D1 fill)
    g = pd.read_parquet(os.path.join(PROC, "fire_gfed.parquet")); q = pd.read_parquet(os.path.join(PROC, "fire_qfed.parquet"))
    for d in (g, q):
        d["year"] = pd.to_datetime(d["date"]).dt.year
    ga = g.groupby(["uc_id", "year"]).pm.mean().unstack(); qa = q.groupby(["uc_id", "year"]).pm.mean().unstack()
    qa[2000] = qa[2000].fillna(ga[2000])
    pm = ((ga + qa) / 2).reindex(df.uc_id)
    early, pre, recent = [2000, 2001, 2002, 2003, 2004], list(range(2015, 2020)), list(range(2019, 2024))
    w = df.pop2020.values
    rows = []
    for name, cols in (("dm25", [f"dm25_{y}" for y in YEARS]), ("dm50", [f"dm50_{y}" for y in YEARS])):
        a = df[cols].values
        a = pd.DataFrame(a, columns=YEARS)
        rows.append((name, a))
    toks = []
    def tok(t, val, unit, d=2):
        toks.append({"token": t, "value": ((f"{val:.{d}f}" if d > 0 else f"{int(round(val))}") if isinstance(val, float) and np.isfinite(val) else val),
                     "unit": unit, "script": "02_code/11_fire_activity_check.py", "inputs": "GFED4.1s, fire PM2.5, city table"})
    pmv = pm.values
    pm_e, pm_p, pm_r = pmv[:, :5].mean(1), pmv[:, 15:20].mean(1), pmv[:, 19:24].mean(1)
    pwr = lambda x: (x * w).sum() / w.sum()
    tok("FA_PM_RATIO_PRE", pwr(pm_p) / pwr(pm_e), "population-weighted fire PM2.5, 2015-2019 / 2000-2004")
    tok("FA_PM_RATIO_2020_2019", pwr(pmv[:, 20]) / pwr(pmv[:, 19]), "fire PM2.5, 2020 / 2019")
    ind = df.country.values == "India"
    tok("FA_PM_RATIO_PRE_INDIA", pwr_i(pm_p, w, ind) / pwr_i(pm_e, w, ind) if pwr_i(pm_e, w, ind) else np.nan, "India fire PM2.5, 2015-2019 / 2000-2004")
    for name, a in rows:
        e, p_, r_ = a.loc[:, 2000:2004].mean(1).values, a.loc[:, 2015:2019].mean(1).values, a.loc[:, 2019:2023].mean(1).values
        N = name.upper()
        tok(f"FA_{N}_RATIO_PRE", pwr(p_) / pwr(e), "population-weighted DM emissions near cities, 2015-2019 / 2000-2004")
        tok(f"FA_{N}_RATIO_2020_2019", pwr(a[2020].values) / pwr(a[2019].values), "DM emissions, 2020 / 2019")
        tok(f"FA_{N}_RATIO_PRE_INDIA", pwr_i(p_, w, ind) / pwr_i(e, w, ind) if pwr_i(e, w, ind) else np.nan, "India DM emissions near cities, 2015-2019 / 2000-2004")
        ok = (e > 0) & (pm_e > 0) & (p_ > 0)
        rho = stats.spearmanr(np.log(p_[ok] / e[ok]), np.log(pm_p[ok] / pm_e[ok]))
        tok(f"FA_{N}_SPEARMAN_PRE", float(rho.statistic), "Spearman rho across cities, log ratios of emissions and fire PM2.5")
        tok(f"FA_{N}_N", int(ok.sum()), "cities with emissions in both periods", 0)
        # cross-city agreement in 2020 drop
        ok2 = (a[2019].values > 0) & (pmv[:, 19] > 0)
        rho2 = stats.spearmanr(np.log((a[2020].values[ok2] + 1) / a[2019].values[ok2]), np.log(pmv[ok2, 20] / pmv[ok2, 19]))
        tok(f"FA_{N}_SPEARMAN_2020", float(rho2.statistic), "Spearman rho, 2020/2019 ratios")
    # season overlap (review of V0_3, comment 8): monthly fire-emission climatology within 2.5 degrees (2003-2016,
    # non-beta years) against the monthly frequency of heat days (city 95th percentile, 1991-2020), Spearman across
    # the 12 months, summarised over cities averaging at least one activation day a year in 2019-2023
    mon = np.zeros((len(df), 12))
    for y in range(2003, 2017):
        with h5py.File(os.path.join(RAW, f"GFED4.1s_{y}.hdf5"), "r") as f:
            for m in range(12):
                fld = f[f"emissions/{m + 1:02d}/DM"][:] * area
                for i, (_, c) in enumerate(ct.iterrows()):
                    mi = (np.abs(lat - c.lat) <= 2.5); mj = (np.abs(((lon - c.lon + 180) % 360) - 180) <= 2.5)
                    mon[i, m] += fld[np.ix_(mi, mj)].sum()
    hm = pd.read_csv(os.path.join(RES, f"heat_month_freq_{v}.csv")).set_index("uc_id").reindex(df.uc_id).values
    rhos = np.array([stats.spearmanr(hm[i], mon[i]).statistic if mon[i].sum() > 0 and np.nanstd(hm[i]) > 0 else np.nan
                     for i in range(len(df))])
    act = ct["activation_2019_2023"].values >= 1
    tok("FA_SEASON_RHO_MED_ACT", float(np.nanmedian(rhos[act])), "median Spearman rho, monthly heat-day frequency vs fire emissions, cities with >= 1 activation day")
    tok("FA_SEASON_RHO_POS_ACT", 100 * float(np.nanmean(rhos[act] > 0)), "% of those cities with positive rho", 0)
    tok("FA_SEASON_RHO_MED_ALL", float(np.nanmedian(rhos)), "median rho, all cities")
    df["season_rho"] = rhos
    df["pm_ratio_pre"] = pm_p / pm_e
    df["dm25_ratio_pre"] = rows[0][1].loc[:, 2015:2019].mean(1).values / rows[0][1].loc[:, 2000:2004].mean(1).values
    df.to_csv(os.path.join(RES, f"fire_activity_check_{v}.csv"), index=False)
    nf = os.path.join(RES, f"P44x_NUMBERS_{v}.csv")
    n = pd.read_csv(nf)
    n = n[~n.token.str.startswith("FA_")]
    pd.concat([n, pd.DataFrame(toks)]).to_csv(nf, index=False)
    print(pd.DataFrame(toks)[["token", "value"]].to_string())


def pwr_i(x, w, m):
    return (x[m] * w[m]).sum() / w[m].sum() if m.any() else np.nan


if __name__ == "__main__":
    main()
