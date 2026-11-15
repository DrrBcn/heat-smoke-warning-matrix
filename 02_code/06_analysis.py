"""P44x analysis (ANALYSIS_PLAN_V1_0): matrix cells, joint levels, compound days, conflict nights, trends,
bootstrap intervals, sensitivity and validation. Writes 03_results/P44x_NUMBERS_<v>.csv and result tables.

Usage: python 02_code/06_analysis.py V0_1
Inputs: 01_data/processed/{city_frame.csv, fire_gfed.parquet, fire_qfed.parquet, temp_cpc.parquet,
        cpc_coverage_log.csv}; 01_data/raw/era5_openmeteo/*.json (validation only).
All thresholds apply to fire-sourced PM2.5 (the product holds the fire component only).
"""
import json, os, sys, glob
import numpy as np
import pandas as pd
from scipy import stats
import pymannkendall as mk
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.environ.get("P44X_PROC", os.path.join(ROOT, "01_data", "processed"))
RES = os.environ.get("P44X_RES", os.path.join(ROOT, "03_results"))
SCRIPT = "02_code/06_analysis.py"
YEARS = np.arange(2000, 2024)
ND = sum(366 if y % 4 == 0 else 365 for y in YEARS)
P_BASE = ("1991-01-01", "2020-12-31")
PRIMARY = dict(product="mean", heat_pct=95, heat_abs=None, smoke=15.0, night=24.0, need=26.0,
               night_next=True, baseline="1991-2020", lag=0, frame="primary", weights="fixed2020",
               tiers="pct", rule="additive", drop_h1=False)
TIERS_PCT = (90, 95, 97.5)
TIERS_ABS = (32.0, 35.0, 38.0)
B_REST = (35.5, 55.5)  # S2 and S3 lower bounds (US AQI 2024: USG and Unhealthy)
NBOOT = 1000
SEED = 44
NUM = []


def tok(token, value, unit="", inputs="fire, temp, frame", digits=None):
    if digits == "auto":  # two decimals below 1, one decimal otherwise (repro check: avoid hiding precision)
        digits = 2 if (value is not None and np.isfinite(value) and abs(value) < 1) else 1
    if digits is not None and value is not None and not (isinstance(value, float) and np.isnan(value)):
        value = f"{value:,.{digits}f}" if digits > 0 else f"{int(round(value)):,}"
    NUM.append({"token": token, "value": value, "unit": unit, "script": SCRIPT, "inputs": inputs})


# ---------------------------------------------------------------- data
def load():
    frame = pd.read_csv(os.path.join(PROC, "city_frame.csv")).sort_values("uc_id").reset_index(drop=True)
    frame["country"] = frame["country"].replace({"México": "Mexico"})  # journal style (review comment 15)
    ids = frame["uc_id"].values
    days = pd.date_range("2000-01-01", "2023-12-31", freq="D")
    tdays = pd.date_range("1991-01-01", "2023-12-31", freq="D")

    def grid(df, col, index):
        return df.pivot(index="date", columns="uc_id", values=col).reindex(index=index, columns=ids).values.T

    data = {}
    for p in ("gfed", "qfed"):
        d = pd.read_parquet(os.path.join(PROC, f"fire_{p}.parquet"))
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        data[p] = grid(d, "pm", days)
    tp = pd.read_parquet(os.path.join(PROC, "temp_cpc.parquet"))
    tp["date"] = pd.to_datetime(tp["date"]).dt.normalize()
    data["tmax_all"] = grid(tp, "tmax", tdays)
    data["tmin_all"] = grid(tp, "tmin", tdays)
    data["mean"] = (data["gfed"] + data["qfed"]) / 2
    data["mean_nofill"] = data["mean"].copy()
    # deviation D1: QFED starts on 1 April 2000; use the GFED-based value for 1 January to 31 March 2000
    d1 = np.asarray(days < "2000-04-01")
    data["mean"][:, d1] = data["gfed"][:, d1]
    off = (days[0] - tdays[0]).days
    data["tmax"] = data["tmax_all"][:, off:]
    data["tmin"] = data["tmin_all"][:, off:]
    return frame, days, tdays, data


def percentiles(data, tdays, baseline):
    if baseline == "1991-2020":
        m = (tdays >= P_BASE[0]) & (tdays <= P_BASE[1])
    else:
        m = (tdays >= "2000-01-01") & (tdays <= "2023-12-31")
    x = data["tmax_all"][:, m]
    return {p: np.nanpercentile(x, p, axis=1) for p in (90, 95, 97.5)}


def shift_next(a):
    return np.concatenate([a[:, 1:], np.full((a.shape[0], 1), np.nan)], axis=1)


# ---------------------------------------------------------------- core counts
def daily_flags(cfg, data, pct):
    pm = data[cfg["product"]]
    tmax, tmin = data["tmax"], data["tmin"]
    tmin_n, pm_n = shift_next(tmin), shift_next(pm)
    valid = np.isfinite(pm) & np.isfinite(tmax) & np.isfinite(tmin)
    heat = tmax >= (cfg["heat_abs"] if cfg["heat_abs"] is not None else pct[cfg["heat_pct"]][:, None])
    smoke = pm >= cfg["smoke"]
    if cfg["lag"] == 1:
        smoke = smoke | (pm_n >= cfg["smoke"])
    if cfg["tiers"] == "pct":
        H = sum((tmax >= pct[p][:, None]).astype(int) for p in TIERS_PCT)
    else:
        H = sum((tmax >= t).astype(int) for t in TIERS_ABS)
    S = sum((pm >= b).astype(int) for b in (cfg["smoke"],) + B_REST)
    hmin = 2 if cfg["drop_h1"] else 1
    both = (H >= hmin) & (S >= 1)
    if cfg["rule"] == "additive":
        sc = np.where(both, H + S, 0)
        lvl = np.select([sc >= 5, sc >= 3, sc >= 2], [3, 2, 1], 0)
    else:  # "max": joint level = max(H, S)
        lvl = np.where(both, np.maximum(H, S), 0)
    # night after the heat day: Tmin of d+1; smoke on d or d+1 (the night spans both UTC days)
    if cfg["night_next"]:
        tn, sm_night, vnight = tmin_n, smoke | (pm_n >= cfg["smoke"]), np.isfinite(tmin_n)
    else:
        tn, sm_night, vnight = tmin, smoke, np.ones_like(valid)
    need = tmax >= cfg["need"]
    vent = tn < cfg["night"]
    base_c = heat & need & sm_night & vnight
    f = {
        "heat": heat, "smoke": smoke, "compound": heat & smoke,
        "activation": lvl >= 1, "level1": lvl == 1, "level2": lvl == 2, "level3": lvl == 3,
        "heat_only": (H >= 1) & (S == 0), "smoke_only": (H == 0) & (S >= 1),
        "conflict": base_c & vent, "norelief": base_c & ~vent, "heatneed_smoke": base_c,
    }
    f = {k: v & valid for k, v in f.items()}
    f["valid"] = valid
    return f, H, S, valid


def city_year(flags, days):
    yrs = days.year.values
    return {k: np.stack([v[:, yrs == y].sum(1) for y in YEARS], axis=1) for k, v in flags.items()}


def weights_matrix(frame, cfg):
    if cfg["weights"] == "fixed2020":
        return np.repeat(frame["pop2020"].values[:, None], len(YEARS), axis=1)
    ep = np.array([2000, 2005, 2010, 2015, 2020, 2025])
    P = frame[[f"pop{e}" for e in ep]].values
    return np.stack([np.array([np.interp(y, ep, row) for row in P]) for y in YEARS], axis=1)


def subset(frame, cfg):
    if cfg["frame"] == "primary":
        return frame["primary"].values
    if cfg["frame"] == "capitals":
        return frame["capital"].values
    if cfg["frame"] == "5m":
        return frame["primary"].values & (frame["pop2020"] >= 5e6).values
    raise ValueError(cfg["frame"])


def summarise(cy, frame, cfg, key, extra_excl=None, boot=False):
    """Summary numbers for one outcome under one configuration. Trend contrasts use only cities with at least
    95% valid city-days in 2000-2023 (plan 6d), optionally also excluding extra_excl (homogeneity sensitivity)."""
    sel = subset(frame, cfg)
    comp = cy["valid"][sel].sum(1) >= 0.95 * ND
    if extra_excl is not None:
        comp = comp & ~extra_excl[sel]
    x = cy[key][sel].astype(float)
    w = weights_matrix(frame, cfg)[sel]
    rec, early = YEARS >= 2019, YEARS <= 2004

    def pw(xx, ww, m):
        return (xx[:, m] * ww[:, m]).sum() / ww[:, m].sum()

    city_recent = x[:, rec].mean(1)
    out = {
        "n_cities": int(sel.sum()), "n_complete": int(comp.sum()),
        "pw_recent": pw(x, w, rec), "pw_recent_c": pw(x[comp], w[comp], rec),
        "pw_early": pw(x[comp], w[comp], early), "pw_all": pw(x[comp], w[comp], np.ones_like(rec)),
        "pw_2003": pw(x[comp], w[comp], YEARS == 2003), "pw_2023": pw(x[comp], w[comp], YEARS == 2023),
        "median_recent": float(np.median(city_recent)),
        "iqr_lo": float(np.percentile(city_recent, 25)), "iqr_hi": float(np.percentile(city_recent, 75)),
        "max_recent": float(city_recent.max()),
        "n_ge1_recent": int((city_recent >= 1).sum()),
        "n_any_recent": int((x[:, rec].sum(1) > 0).sum()),
        "persondays_recent": float((x[:, rec] * w[:, rec]).sum(0).mean()),
    }
    out["ratio"] = out["pw_recent_c"] / out["pw_early"] if out["pw_early"] > 0 else np.nan
    pre = (YEARS >= 2015) & (YEARS <= 2019)  # deviation D8: window before the 2020 break in the fire product
    out["pw_pre"] = pw(x[comp], w[comp], pre)
    out["ratio_pre"] = out["pw_pre"] / out["pw_early"] if out["pw_early"] > 0 else np.nan
    if boot:  # cluster bootstrap over cities
        rng = np.random.default_rng(SEED)
        n = x.shape[0]
        bs_rec, bs_ratio, bs_pre = [], [], []
        xc, wc = x[comp], w[comp]
        for _ in range(NBOOT):
            i = rng.integers(0, n, n)
            bs_rec.append(pw(x[i], w[i], rec))
            j = rng.integers(0, xc.shape[0], xc.shape[0])
            e = pw(xc[j], wc[j], early)
            bs_ratio.append(pw(xc[j], wc[j], rec) / e if e > 0 else np.nan)
            bs_pre.append(pw(xc[j], wc[j], pre) / e if e > 0 else np.nan)
        out["rec_lo"], out["rec_hi"] = np.nanpercentile(bs_rec, [2.5, 97.5])
        out["ratio_lo"], out["ratio_hi"] = np.nanpercentile(bs_ratio, [2.5, 97.5])
        out["pre_lo"], out["pre_hi"] = np.nanpercentile(bs_pre, [2.5, 97.5])
    return out, comp


def trends(cy, frame, key, comp_all):
    """Global population-weighted series: Theil-Sen slope, Hamed-Rao modified Mann-Kendall (original as
    sensitivity). Per city: Poisson regression of annual counts on year for cities with >= 10 events,
    Benjamini-Hochberg false discovery rate 0.05."""
    sel = frame["primary"].values & comp_all
    x = cy[key][sel].astype(float)
    w = frame["pop2020"].values[sel]
    series = (x * w[:, None]).sum(0) / w.sum()
    ts = stats.theilslopes(series, YEARS)
    hr = mk.hamed_rao_modification_test(series, alpha=0.05)
    og = mk.original_test(series, alpha=0.05)
    pv, sl = [], []
    X = sm.add_constant(YEARS - 2000.0)
    for row in x:
        if row.sum() < 10:
            continue
        try:
            r = sm.GLM(row, X, family=sm.families.Poisson()).fit(scale="X2")  # quasi-Poisson (overdispersion)
            pv.append(r.pvalues[1]); sl.append(r.params[1])
        except Exception:
            continue
    pv, sl = np.array(pv), np.array(sl)
    up = down = 0
    if len(pv):
        o = np.argsort(pv); m = len(pv)
        thr = (np.arange(1, m + 1) / m) * 0.05
        passed = pv[o] <= thr
        k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
        sig = np.zeros(m, bool); sig[o[:k]] = True
        up, down = int((sig & (sl > 0)).sum()), int((sig & (sl < 0)).sum())
    return {"n_tested": len(pv), "up": up, "down": down, "slope": ts.slope, "slope_lo": ts.low_slope,
            "slope_hi": ts.high_slope, "p_hr": hr.p, "p_orig": og.p, "series": series}


# ---------------------------------------------------------------- main
def main():
    v = sys.argv[1] if len(sys.argv) > 1 else "V0_1"
    os.makedirs(RES, exist_ok=True)
    frame, days, tdays, data = load()
    pct = {b: percentiles(data, tdays, b) for b in ("1991-2020", "2000-2023")}

    # data quality: cities without 95% valid Tmax and Tmin days in 2019-2023 leave every frame (plan 6e)
    rec_m = days >= "2019-01-01"
    okrec = (np.isfinite(data["tmax"][:, rec_m]) & np.isfinite(data["tmin"][:, rec_m])).mean(1) >= 0.95
    tok("N_CITIES_FRAME", int(frame["primary"].sum()), "urban centres of 1 million or more in GHSL (2020)")
    tok("N_EXCL_NODATA", int((frame["primary"].values & ~okrec).sum()), "cities without temperature data")
    tok("EXCL_NODATA_NAMES", "; ".join(frame.loc[frame["primary"].values & ~okrec, "city"]), "")
    for c in ("primary", "capital"):
        frame[c] = frame[c].values & okrec
    prim = frame["primary"].values
    tok("N_CITIES_ALL", int(prim.sum()), "urban centres with temperature data")

    flags, H, S, valid = daily_flags(PRIMARY, data, pct["1991-2020"])
    cy = city_year(flags, days)
    comp_all = cy["valid"].sum(1) >= 0.95 * ND
    tok("N_CITIES_TREND", int((prim & comp_all).sum()), "cities in trend contrasts (>= 95% valid days)")
    tok("N_CITIES_INCOMPLETE", int((prim & ~comp_all).sum()), "cities excluded from trend contrasts")
    tok("INCOMPLETE_NAMES", "; ".join(frame.loc[prim & ~comp_all, "city"]), "")
    vshare = valid[prim].mean(1) * 100
    tok("VALID_DAYS_MEDIAN_PCT", float(np.median(vshare)), "% valid city-days, 2000-2023", digits=1)
    tok("VALID_DAYS_MIN_PCT", float(vshare.min()), "%", digits=1)
    qc = frame.loc[prim, ["uc_id", "city", "country"]].copy()
    qc["valid_days_pct"] = vshare
    qc["tmax_p90"] = pct["1991-2020"][90][prim]; qc["tmax_p95"] = pct["1991-2020"][95][prim]
    qc["tmax_p975"] = pct["1991-2020"][97.5][prim]
    qc.to_csv(os.path.join(RES, f"QC_city_{v}.csv"), index=False)
    tok("P95_MIN", float(qc.tmax_p95.min()), "C", digits=1); tok("P95_MAX", float(qc.tmax_p95.max()), "C", digits=1)
    tok("N_P95_BELOW_NEED", int((qc.tmax_p95 < PRIMARY["need"]).sum()), "cities whose 95th percentile is below 26 C")

    # CPC homogeneity flags (plan 6f)
    cov = pd.read_csv(os.path.join(PROC, "cpc_coverage_log.csv"))
    a = cov[cov.year == 2003].set_index("uc_id")["own_weight"]; b = cov[cov.year == 2010].set_index("uc_id")["own_weight"]
    ring = cov[cov.year >= 2000].groupby("uc_id")["ring_days"].sum() / ND
    hflag = ((a - b).abs() > 0.2) | (ring > 0.01)
    inhom = frame["uc_id"].map(hflag).fillna(False).values.astype(bool)
    tok("N_INHOMOG", int((prim & inhom).sum()), "cities flagged for CPC coverage change")
    # temperature step screen (deviation D4): 5-year block means of annual mean Tmax, 1991-2023; a city is
    # flagged when consecutive blocks differ by more than the threshold or 2019-2023 differs from 2000-2004 by more
    ty = tdays.year.values
    ann = np.stack([np.nanmean(data["tmax_all"][:, ty == y], 1) for y in range(1991, 2024)], 1)
    yy = np.arange(1991, 2024)
    blocks = [(1991, 1995), (1996, 2000), (2001, 2005), (2006, 2010), (2011, 2015), (2016, 2020), (2019, 2023)]
    bm = np.stack([np.nanmean(ann[:, (yy >= a_) & (yy <= b_)], 1) for a_, b_ in blocks], 1)
    jump = np.nanmax(np.abs(np.diff(bm, axis=1)), 1)
    er = np.nanmean(ann[:, yy >= 2019], 1) - np.nanmean(ann[:, (yy >= 2000) & (yy <= 2004)], 1)
    step2 = (jump > 2.0) | (np.abs(er) > 2.0)
    step15 = (jump > 1.5) | (np.abs(er) > 1.5)
    tok("N_STEP2", int((prim & step2).sum()), "cities flagged for a temperature step above 2 C")
    tok("N_STEP15", int((prim & step15).sum()), "cities flagged at 1.5 C")
    tok("TMAX_CHANGE_MEDIAN", float(np.median(er[prim & ~step2])), "C, median change in annual mean Tmax 2000-2004 to 2019-2023, unflagged cities", digits=1)
    tok("STEP2_NAMES", "; ".join(frame.loc[prim & step2, "city"]), "")
    # deviation D7: step-flagged cities leave the primary frame for levels as well as trends
    frame["primary_all"] = prim.copy()
    frame["primary"] = prim & ~step2
    frame["capital"] = frame["capital"].values & ~step2
    prim = frame["primary"].values
    tok("N_CITIES", int(prim.sum()), "urban centres analysed (homogeneous temperature records)")
    tok("N_CAPITALS", int(frame["capital"].sum()), "capitals analysed")
    tok("N_5M", int((prim & (frame["pop2020"] >= 5e6).values).sum()), "urban centres")
    tok("POP_PRIMARY_BN", frame.loc[prim, "pop2020"].sum() / 1e9, "billion (2020)", digits=2)
    tok("N_COUNTRIES", int(frame.loc[prim, "iso3"].nunique()), "countries")
    for g_, n_ in frame[prim].groupby("income_group")["pop2020"].sum().items():
        tok("POPSHARE_" + str(g_).upper().replace(" ", "_").replace("-", "_"),
            100 * n_ / frame.loc[prim, "pop2020"].sum(), "% of population", digits=1)

    keys = ["heat", "smoke", "compound", "activation", "level1", "level2", "level3", "conflict", "norelief",
            "heatneed_smoke", "heat_only", "smoke_only"]
    summ = {}
    for k in keys:
        s, _ = summarise(cy, frame, PRIMARY, k, extra_excl=step2,
                         boot=k in ("compound", "activation", "level3", "conflict", "norelief"))
        summ[k] = s
        K = k.upper()
        tok(f"{K}_PW_RECENT", s["pw_recent"], "days per resident per year, 2019-2023", digits="auto")
        tok(f"{K}_PW_EARLY", s["pw_early"], "days per resident per year, 2000-2004 (complete cities)", digits="auto")
        tok(f"{K}_PW_ALL", s["pw_all"], "days per resident per year, 2000-2023", digits="auto")
        tok(f"{K}_PW_2003", s["pw_2003"], "days per resident, 2003 (descriptive)", digits="auto")
        tok(f"{K}_PW_2023", s["pw_2023"], "days per resident, 2023 (descriptive)", digits="auto")
        tok(f"{K}_RATIO", s["ratio"], "ratio 2019-2023 to 2000-2004", digits=2)
        tok(f"{K}_PW_PRE", s["pw_pre"], "days per resident per year, 2015-2019", digits="auto")
        tok(f"{K}_RATIO_PRE", s["ratio_pre"], "ratio 2015-2019 to 2000-2004", digits=2)
        tok(f"{K}_MEDIAN_RECENT", s["median_recent"], "days per city per year, 2019-2023", digits=1)
        tok(f"{K}_IQR_LO", s["iqr_lo"], "days", digits=1)
        tok(f"{K}_IQR_HI", s["iqr_hi"], "days", digits=1)
        tok(f"{K}_MAX_RECENT", s["max_recent"], "days per year", digits=1)
        tok(f"{K}_NCITY_GE1", s["n_ge1_recent"], "cities with a mean of at least 1 per year, 2019-2023 (primary)")
        tok(f"{K}_NCITY_ANY", s["n_any_recent"], "cities with any, 2019-2023")
        tok(f"{K}_PCTCITY_GE1", 100 * s["n_ge1_recent"] / s["n_cities"], "% of cities", digits=0)
        tok(f"{K}_PCTCITY_ANY", 100 * s["n_any_recent"] / s["n_cities"], "% of cities", digits=0)
        tok(f"{K}_PERSONDAYS_BN", s["persondays_recent"] / 1e9, "billion person-days per year, 2019-2023", digits=2)
        if "rec_lo" in s:
            tok(f"{K}_PW_RECENT_LO", s["rec_lo"], "bootstrap 2.5%", digits="auto")
            tok(f"{K}_PW_RECENT_HI", s["rec_hi"], "bootstrap 97.5%", digits="auto")
            tok(f"{K}_RATIO_LO", s["ratio_lo"], "bootstrap 2.5%", digits=2)
            tok(f"{K}_RATIO_HI", s["ratio_hi"], "bootstrap 97.5%", digits=2)
            tok(f"{K}_RATIO_PRE_LO", s["pre_lo"], "bootstrap 2.5%", digits=2)
            tok(f"{K}_RATIO_PRE_HI", s["pre_hi"], "bootstrap 97.5%", digits=2)
    assert summ["compound"]["n_complete"] == int((prim & comp_all & ~step2).sum())
    tok("N_CITIES_TREND_PRIMARY", summ["compound"]["n_complete"], "cities in trend contrasts (complete, no temperature step)")
    tok("LEVEL2_SHARE_OF_ACTIVATION", 100 * summ["level2"]["pw_recent"] / summ["activation"]["pw_recent"], "%", digits=0)
    tok("LEVEL1_SHARE_OF_ACTIVATION", 100 * summ["level1"]["pw_recent"] / summ["activation"]["pw_recent"], "%", digits=0)
    tok("CONFLICT_SHARE_OF_NEED", 100 * summ["conflict"]["pw_recent"] / summ["heatneed_smoke"]["pw_recent"]
        if summ["heatneed_smoke"]["pw_recent"] else np.nan, "% of hot smoky days whose next night allows cooling", digits=0)
    # expected compound days under independence (plan 6g)
    vd = cy["valid"][prim].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        exp = np.where(vd > 0, cy["heat"][prim] * cy["smoke"][prim] / vd, 0)
    w20 = frame.loc[prim, "pop2020"].values
    rec = YEARS >= 2019
    exp_pw = (exp[:, rec].mean(1) * w20).sum() / w20.sum()
    tok("COMPOUND_EXPECTED_INDEP_PW", exp_pw, "days per resident per year if independent, 2019-2023", digits="auto")
    tok("COMPOUND_OBS_EXP_RATIO", summ["compound"]["pw_recent"] / exp_pw if exp_pw else np.nan, "observed/expected", digits=1)

    # trends
    tr = {}
    for k in ("compound", "activation", "conflict"):
        tr[k] = trends(cy, frame, k, comp_all & ~step2)
        K = k.upper()
        tok(f"{K}_NCITY_TESTED", tr[k]["n_tested"], "cities with >= 10 events, 2000-2023")
        tok(f"{K}_NCITY_UP", tr[k]["up"], "cities with significant increase (quasi-Poisson, BH-FDR 0.05)")
        tok(f"{K}_NCITY_DOWN", tr[k]["down"], "cities with significant decrease")
        tok(f"{K}_SLOPE_DECADE", tr[k]["slope"] * 10, "days per resident per decade (Theil-Sen)", digits=2)
        tok(f"{K}_SLOPE_LO", tr[k]["slope_lo"] * 10, "95% CI low", digits=2)
        tok(f"{K}_SLOPE_HI", tr[k]["slope_hi"] * 10, "95% CI high", digits=2)
        tok(f"{K}_MK_P", tr[k]["p_hr"], "p, Hamed-Rao modified Mann-Kendall", digits=3)
        tok(f"{K}_MK_P_ORIG", tr[k]["p_orig"], "p, original Mann-Kendall", digits=3)
    pd.DataFrame({"year": YEARS, **{k: tr[k]["series"] for k in tr}}).to_csv(
        os.path.join(RES, f"global_series_{v}.csv"), index=False)
    rs = []
    for g, idx in frame[prim & comp_all & ~step2].groupby("who_region").groups.items():
        m = np.zeros(len(frame), bool); m[idx] = True
        ww = frame["pop2020"].values[m]
        for k in ("activation", "conflict"):
            ser = (cy[k][m] * ww[:, None]).sum(0) / ww.sum()
            rs += [{"who_region": g, "outcome": k, "year": y, "value": val} for y, val in zip(YEARS, ser)]
    pd.DataFrame(rs).to_csv(os.path.join(RES, f"region_series_{v}.csv"), index=False)

    # matrix cells
    w = frame["pop2020"].values[prim]
    yrs = days.year.values
    cells = []
    for h in range(4):
        for s_ in range(4):
            m = (H[prim] == h) & (S[prim] == s_) & valid[prim]
            r_ = m[:, yrs >= 2019].sum(1) / 5
            cells.append({"H": h, "S": s_, "pw_days_recent": (r_ * w).sum() / w.sum(),
                          "pw_days_all": ((m.sum(1) / 24) * w).sum() / w.sum(), "n_cities_recent_any": int((r_ > 0).sum())})
    cells = pd.DataFrame(cells)
    cells.to_csv(os.path.join(RES, f"matrix_cells_{v}.csv"), index=False)
    for _, c in cells.iterrows():
        tok(f"CELL_H{int(c.H)}S{int(c.S)}_PW", c.pw_days_recent, "days per resident per year, 2019-2023", digits=2)
        tok(f"CELL_H{int(c.H)}S{int(c.S)}_NCITY", c.n_cities_recent_any, "cities with any day, 2019-2023")

    # by region and income
    rows = []
    for col in ("who_region", "income_group"):
        for g, idx in frame[prim].groupby(col).groups.items():
            m = np.zeros(len(frame), bool); m[idx] = True
            mc = m & comp_all & ~step2
            ww, wc = frame["pop2020"].values[m], frame["pop2020"].values[mc]
            r = {"group_type": col, "group": g, "n_cities": int(m.sum()), "pop_bn": ww.sum() / 1e9}
            for k in ("compound", "activation", "level3", "conflict", "norelief"):
                x = cy[k][m][:, rec].mean(1)
                r[f"{k}_pw"] = (x * ww).sum() / ww.sum()
                r[f"{k}_ncity_ge1"] = int((x >= 1).sum())
                e = cy[k][mc][:, YEARS <= 2004].mean(1); rc = cy[k][mc][:, rec].mean(1)
                r[f"{k}_pw_early"] = (e * wc).sum() / wc.sum() if wc.sum() else np.nan
                r[f"{k}_ratio"] = ((rc * wc).sum() / (e * wc).sum()) if (e * wc).sum() > 0 else np.nan
                pr = cy[k][mc][:, (YEARS >= 2015) & (YEARS <= 2019)].mean(1)
                r[f"{k}_ratio_pre"] = ((pr * wc).sum() / (e * wc).sum()) if (e * wc).sum() > 0 else np.nan
                r[f"{k}_share_pn"] = 100 * (x * ww).sum() / (cy[k][prim][:, rec].mean(1) * frame["pop2020"].values[prim]).sum()
            rows.append(r)
    grp = pd.DataFrame(rows)
    grp.to_csv(os.path.join(RES, f"by_group_{v}.csv"), index=False)
    for _, r in grp.iterrows():
        g = str(r["group"]).upper().replace(" ", "_").replace("-", "_")
        tok(f"GRP_{g}_N", r["n_cities"], "cities")
        for k in ("compound", "activation", "level3", "conflict", "norelief"):
            tok(f"GRP_{g}_{k.upper()}_PW", r[f"{k}_pw"], "days per resident per year, 2019-2023", digits="auto")
            tok(f"GRP_{g}_{k.upper()}_NGE1", r[f"{k}_ncity_ge1"], "cities")
            # ratios are not estimable when the 2000-2004 level is below 0.1 days per resident (audit P10)
            # ratios from a 2000-2004 level below 0.1 days per resident are not estimable (review of V0_3, comment 14)
            if not r[f"{k}_pw_early"] >= 0.1:
                tok(f"GRP_{g}_{k.upper()}_RATIO", "not estimable", "2000-2004 level below 0.1 days per resident")
                tok(f"GRP_{g}_{k.upper()}_RATIO_PRE", "not estimable", "2000-2004 level below 0.1 days per resident")
            else:
                tok(f"GRP_{g}_{k.upper()}_RATIO", r[f"{k}_ratio"], "ratio 2019-2023 to 2000-2004", digits=2)
                tok(f"GRP_{g}_{k.upper()}_RATIO_PRE", r[f"{k}_ratio_pre"], "ratio 2015-2019 to 2000-2004", digits=2)
            tok(f"GRP_{g}_{k.upper()}_SHARE_PN", r[f"{k}_share_pn"], "% of all person-days or person-nights, 2019-2023", digits=0)

    # regional and income-group ratio intervals (cluster bootstrap over cities; added after the editor review, D3)
    rng = np.random.default_rng(SEED)
    ci_rows = []
    for col in ("who_region", "income_group"):
        for g, idx in frame[prim & comp_all & ~step2].groupby(col).groups.items():
            ii = np.array(list(idx))
            ww = frame["pop2020"].values[ii]
            for k in ("activation", "conflict"):
                X = cy[k][ii].astype(float)
                def ratio(sel):
                    e = (X[sel][:, YEARS <= 2004].mean(1) * ww[sel]).sum()
                    return (X[sel][:, rec].mean(1) * ww[sel]).sum() / e if e > 0 else np.nan
                bs = [ratio(rng.integers(0, len(ii), len(ii))) for _ in range(NBOOT)]
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                ci_rows.append({"group": g, "outcome": k, "ratio": ratio(np.arange(len(ii))), "lo": lo, "hi": hi})
                G = str(g).upper().replace(" ", "_").replace("-", "_")
                e0 = (X[:, YEARS <= 2004].mean(1) * ww).sum() / ww.sum()
                tok(f"GRP_{G}_{k.upper()}_RATIO_LO", lo if e0 >= 0.1 else "", "bootstrap 2.5%", digits=2 if e0 >= 0.1 else None)
                tok(f"GRP_{G}_{k.upper()}_RATIO_HI", hi if e0 >= 0.1 else "", "bootstrap 97.5%", digits=2 if e0 >= 0.1 else None)
                tok(f"GRP_{G}_{k.upper()}_RATIO_CI", f"{lo:.2f} to {hi:.2f}" if e0 >= 0.1 else "not estimable", "interval text")
    pd.DataFrame(ci_rows).to_csv(os.path.join(RES, f"group_ratio_ci_{v}.csv"), index=False)

    # exploratory decomposition (deviation D2, not in the locked plan): recombine 2019-2023 temperature with
    # 2000-2004 fire PM2.5 (same day of year, 19 years earlier) and the reverse, to separate the heat and smoke
    # contributions to the change in activation days and conflict nights
    doy = days.dayofyear.values
    yrs_ = days.year.values

    def swap(src_years, dst_years, arr):
        out = arr.copy()
        for ys, yd in zip(src_years, dst_years):
            ms, md = yrs_ == ys, yrs_ == yd
            dsrc = np.minimum(doy[md], ms.sum())  # 29 Feb and 31 Dec map to the last available day
            out[:, np.where(md)[0]] = arr[:, np.where(ms)[0][dsrc - 1]]
        return out
    early_y, rec_y = list(range(2000, 2005)), list(range(2019, 2024))
    dec = {}
    for name, comp_ in (("heat_recent_smoke_early", "smoke"), ("heat_early_smoke_recent", "temp")):
        d2 = dict(data)
        if comp_ == "smoke":
            d2["mean"] = swap(early_y, rec_y, data["mean"])
        else:
            d2["tmax"] = swap(early_y, rec_y, data["tmax"]); d2["tmin"] = swap(early_y, rec_y, data["tmin"])
        fl2, *_ = daily_flags(PRIMARY, d2, pct["1991-2020"])
        c2 = city_year(fl2, days)
        for k in ("activation", "conflict", "compound"):
            s2, _ = summarise(c2, frame, PRIMARY, k, extra_excl=step2)
            dec[(name, k)] = s2["pw_recent_c"]  # same city set as the 2000-2004 denominator
            tok(f"DEC_{name.upper()}_{k.upper()}", s2["pw_recent"], "days per resident per year (exploratory)", digits="auto")
    for k in ("activation", "conflict", "compound"):
        e = summ[k]["pw_early"]
        tok(f"DEC_{k.upper()}_HEATONLY_RATIO", dec[("heat_recent_smoke_early", k)] / e if e else np.nan,
            "ratio to 2000-2004 with recent heat and early smoke (exploratory)", digits=2)
        tok(f"DEC_{k.upper()}_SMOKEONLY_RATIO", dec[("heat_early_smoke_recent", k)] / e if e else np.nan,
            "ratio to 2000-2004 with early heat and recent smoke (exploratory)", digits=2)

    # warming experiment (deviation D9): 2019-2023 days with each city's change in annual mean Tmax and Tmin since
    # 2000-2004 removed, smoke and day-to-day pairing kept as observed
    tn_ann = np.stack([np.nanmean(data["tmin_all"][:, ty == y], 1) for y in range(1991, 2024)], 1)
    er_n = np.nanmean(tn_ann[:, yy >= 2019], 1) - np.nanmean(tn_ann[:, (yy >= 2000) & (yy <= 2004)], 1)
    d3 = dict(data)
    recd = np.asarray(days >= "2019-01-01")
    d3["tmax"] = data["tmax"].copy(); d3["tmin"] = data["tmin"].copy()
    d3["tmax"][:, recd] -= er[:, None]; d3["tmin"][:, recd] -= er_n[:, None]
    fl3, *_ = daily_flags(PRIMARY, d3, pct["1991-2020"])
    c3 = city_year(fl3, days)
    for k in ("activation", "conflict", "compound", "level3"):
        s3, _ = summarise(c3, frame, PRIMARY, k)
        obs = summ[k]["pw_recent"]
        tok(f"WARM_{k.upper()}_CF", s3["pw_recent"], "days per resident per year without warming since 2000-2004", digits="auto")
        tok(f"WARM_{k.upper()}_ADD_PCT", 100 * (obs / s3["pw_recent"] - 1) if s3["pw_recent"] else np.nan,
            "% more 2019-2023 days than without the temperature change since 2000-2004", digits=0)
    tok("WARM_TMIN_CHANGE_MEDIAN", float(np.median(er_n[prim])), "C", digits=1)
    for tag, prod, fmask in (("GFED", "gfed", None), ("QFED", "qfed", None), ("STEP15", "mean", step15)):
        cfgw = {**PRIMARY, "product": prod}
        fr_w = frame
        if fmask is not None:
            fr_w = frame.copy(); fr_w["primary"] = frame["primary_all"].values & ~fmask
        flo, *_ = daily_flags(cfgw, data, pct["1991-2020"]); clo = city_year(flo, days)
        flc, *_ = daily_flags(cfgw, d3, pct["1991-2020"]); clc = city_year(flc, days)
        for k in ("activation", "conflict"):
            so, _ = summarise(clo, fr_w, cfgw, k); sc, _ = summarise(clc, fr_w, cfgw, k)
            tok(f"WARM_{tag}_{k.upper()}_ADD_PCT", 100 * (so["pw_recent"] / sc["pw_recent"] - 1) if sc["pw_recent"] else np.nan,
                f"% added by warming, {tag}", digits=0)

    # city table
    ct = frame.loc[prim, ["uc_id", "city", "country", "names_list", "who_region", "income_group", "pop2020", "lat", "lon"]].copy()
    for k in ("heat", "smoke", "compound", "activation", "level3", "conflict", "norelief"):
        ct[f"{k}_2019_2023"] = cy[k][prim][:, rec].mean(1)
        ct[f"{k}_2000_2004"] = cy[k][prim][:, YEARS <= 2004].mean(1)
    ct["complete_2000_2023"] = comp_all[prim]
    ct["cpc_inhomogeneous"] = inhom[prim]
    ct["temp_step_flag"] = step2[prim]
    ct["tmax_change_2000_2004_to_2019_2023"] = er[prim]
    ct["conflict_max_year"] = cy["conflict"][prim].max(1)
    ct = ct.sort_values("conflict_2019_2023", ascending=False)
    ct.to_csv(os.path.join(RES, f"city_table_{v}.csv"), index=False)
    # ranking over all cities with temperature data, flagging step-change cities (Supplementary Table 3)
    pa = frame["primary_all"].values
    ca = frame.loc[pa, ["uc_id", "city", "country"]].copy()
    for k in ("heat", "activation", "conflict"):
        ca[f"{k}_2019_2023"] = cy[k][pa][:, rec].mean(1)
        ca[f"{k}_2000_2004"] = cy[k][pa][:, YEARS <= 2004].mean(1)
    ca["temp_step_flag"] = step2[pa]
    ca["tmax_change_2000_2004_to_2019_2023"] = er[pa]
    ca = ca.sort_values("conflict_2019_2023", ascending=False)
    ca.to_csv(os.path.join(RES, f"city_table_all_{v}.csv"), index=False)
    tok("TOP20_N_STEPFLAG", int(ca.head(20)["temp_step_flag"].sum()), "of the 20 cities with most conflict nights (all cities)")
    tok("TOP20_STEPFLAG_NAMES", "; ".join(ca.head(20).loc[ca.head(20)["temp_step_flag"], "city"]), "")
    for i, (_, r) in enumerate(ct[~ct["temp_step_flag"]].head(5).iterrows(), 1):
        tok(f"TOP{i}_CITY", f"{r['city']} ({r['country']})", "")
        tok(f"TOP{i}_CONFLICT", r["conflict_2019_2023"], "conflict nights per year, 2019-2023", digits=1)
    for i, (_, r) in enumerate(ct[~ct["temp_step_flag"]].sort_values("activation_2019_2023", ascending=False).head(5).iterrows(), 1):
        tok(f"TOPACT{i}_CITY", f"{r['city']} ({r['country']})", "")
        tok(f"TOPACT{i}_DAYS", r["activation_2019_2023"], "activation days per year, 2019-2023", digits=1)

    # sensitivity (one at a time)
    sens_cfgs = {
        "Primary": {},
        "GFED-based": {"product": "gfed"}, "QFED-based": {"product": "qfed"},
        "No January-March 2000 fill (D1)": {"product": "mean_nofill"},
        "Heat day 90th percentile": {"heat_pct": 90}, "Heat day 97.5th percentile": {"heat_pct": 97.5},
        "Heat day absolute 35 C": {"heat_abs": 35.0},
        "Smoke 5 ug/m3": {"smoke": 5.0}, "Smoke 25 ug/m3": {"smoke": 25.0}, "Smoke 35.5 ug/m3": {"smoke": 35.5},
        "Cooling need 24 C": {"need": 24.0}, "Cooling need 28 C": {"need": 28.0},
        "Night limit 22 C": {"night": 22.0}, "Night limit 26 C": {"night": 26.0},
        "Same-date night": {"night_next": False},
        "Matrix tiers absolute 32/35/38 C": {"tiers": "abs"}, "Joint level = max(H, S)": {"rule": "max"},
        "Activation needs heat tier 2 or 3": {"drop_h1": True},
        "Baseline 2000-2023": {"baseline": "2000-2023"}, "Smoke lag 0-1 day": {"lag": 1},
        "Frame: capitals": {"frame": "capitals"}, "Frame: 5 million or more": {"frame": "5m"},
        "Year-matched population": {"weights": "interp"},
        "Exclude CPC-inhomogeneous cities (trend)": {"_inhom": True},
        "All cities including temperature steps": {"_keepstep": True},
        "Trend: step rule 1.5 C": {"_step15": True},
    }
    srows = []
    for name, ch in sens_cfgs.items():
        cfg = {**PRIMARY, **{k: x for k, x in ch.items() if not k.startswith("_")}}
        fl, *_ = daily_flags(cfg, data, pct[cfg["baseline"]])
        c2 = city_year(fl, days)
        row = {"analysis": name}
        excl = step2.copy()
        if ch.get("_inhom"):
            excl = excl | inhom
        fr = frame
        if ch.get("_keepstep"):
            excl = np.zeros_like(step2)
            fr = frame.copy(); fr["primary"] = fr["primary_all"].values
        if ch.get("_step15"):
            excl = step15
        for k in ("compound", "activation", "level1", "level2", "level3", "conflict", "norelief"):
            s, _ = summarise(c2, fr, cfg, k, extra_excl=excl)
            row[f"{k}_pw_recent"] = s["pw_recent"]; row[f"{k}_ratio"] = s["ratio"]; row[f"{k}_ratio_pre"] = s["ratio_pre"]
            row[f"{k}_ncity_ge1"] = s["n_ge1_recent"]; row["n_cities"] = s["n_cities"]
        srows.append(row)
    sens = pd.DataFrame(srows)
    sens.to_csv(os.path.join(RES, f"sensitivity_{v}.csv"), index=False)
    import re as _re
    for _, r in sens.iterrows():
        slug = _re.sub(r"[^A-Z0-9]+", "_", r["analysis"].upper()).strip("_")
        for k in ("activation", "conflict", "compound"):
            tok(f"S_{slug}_{k.upper()}", r[f"{k}_pw_recent"], "days per resident per year, 2019-2023", digits="auto")
            tok(f"S_{slug}_{k.upper()}_RATIO", r[f"{k}_ratio"], "ratio", digits=2)
            tok(f"S_{slug}_{k.upper()}_NGE1", r[f"{k}_ncity_ge1"], "cities")
    for k in ("compound", "activation", "conflict"):
        K = k.upper()
        vals = sens[f"{k}_pw_recent"]
        tok(f"{K}_SENS_MIN", vals.min(), "days per resident per year", digits="auto")
        tok(f"{K}_SENS_MAX", vals.max(), "days per resident per year", digits="auto")
        tok(f"{K}_SENS_MIN_NAME", sens.loc[vals.idxmin(), "analysis"], "")
        tok(f"{K}_SENS_MAX_NAME", sens.loc[vals.idxmax(), "analysis"], "")
        for name, tag in (("GFED-based", "GFED"), ("QFED-based", "QFED")):
            r = sens[sens.analysis == name].iloc[0]
            tok(f"{K}_{tag}_PW", r[f"{k}_pw_recent"], "days per resident per year", digits="auto")
            tok(f"{K}_{tag}_RATIO", r[f"{k}_ratio"], "ratio", digits=2)
        rr = sens[f"{k}_ratio"].replace([np.inf], np.nan)
        tok(f"{K}_RATIO_SENS_MIN", rr.min(), "ratio", digits=2)
        tok(f"{K}_RATIO_SENS_MAX", rr.max(), "ratio", digits=2)
        tok(f"{K}_RATIO_INHOM", sens.loc[sens.analysis.str.startswith("Exclude CPC"), f"{k}_ratio"].iloc[0], "ratio", digits=2)

    # validation: products
    m = prim[:, None] & np.isfinite(data["gfed"]) & np.isfinite(data["qfed"])
    tok("VAL_GFED_QFED_R", np.corrcoef(data["gfed"][m], data["qfed"][m])[0, 1], "Pearson r, daily city values", digits=2)
    tok("VAL_GFED_MEAN", np.nanmean(data["gfed"][prim]), "ug/m3 fire-sourced PM2.5, city-day mean", digits=2)
    tok("VAL_QFED_MEAN", np.nanmean(data["qfed"][prim]), "ug/m3 fire-sourced PM2.5, city-day mean", digits=2)
    tok("VAL_GFED_GT_QFED_PCT", (data["gfed"][prim] > data["qfed"][prim]).mean() * 100, "% of city-days GFED above QFED", digits=0)
    # comparison with Wu et al. 2026 definitions: heatwave = Tmax > 90th percentile on >= 2 consecutive days;
    # smoke = fire-sourced PM2.5 >= 15 (their all-source rule is not reproducible with a fire-only product); 2000-2019
    hot = data["tmax"] > pct["1991-2020"][90][:, None]
    nxt = np.concatenate([hot[:, 1:], np.zeros((hot.shape[0], 1), bool)], 1)
    prv = np.concatenate([np.zeros((hot.shape[0], 1), bool), hot[:, :-1]], 1)
    hw = hot & (nxt | prv)
    yr_ok = np.asarray(days.year <= 2019)
    sm_ = data["mean"] >= 15
    wv = frame["pop2020"].values
    num = ((hw & sm_ & valid)[prim][:, yr_ok].sum(1) * wv[prim]).sum()
    den = ((hw & valid)[prim][:, yr_ok].sum(1) * wv[prim]).sum()
    tok("WU_SHARE_HW_SMOKE_PCT", 100 * num / den, "% of heatwave person-days with fire PM2.5 >= 15, 2000-2019", digits=1)

    events = [("Sydney", "2019-11-01", "2020-01-31"), ("Moscow", "2010-07-15", "2010-08-20"),
              ("Vancouver", "2018-08-01", "2018-08-31"), ("Seattle", "2020-09-01", "2020-09-30"),
              ("Singapore", "2015-09-01", "2015-10-31"), ("Kuala Lumpur", "2015-09-01", "2015-10-31"),
              ("New York City", "2023-06-01", "2023-06-15"), ("Los Angeles", "2020-08-15", "2020-10-15"),
              ("San Francisco", "2020-08-15", "2020-10-15"), ("Sacramento", "2018-11-08", "2018-11-25"),
              ("Sacramento", "2020-08-15", "2020-10-15")]
    ev = []
    for city, a_, b_ in events:
        idx = frame.index[(frame["city"] == city) & frame["primary"]]
        if len(idx) == 0:
            ev.append({"city": city, "period": f"{a_} to {b_}", "found": False}); continue
        i = idx[0]; mm = (days >= a_) & (days <= b_)
        row = {"city": city, "period": f"{a_} to {b_}", "found": True}
        for p in ("gfed", "qfed", "mean"):
            row[f"{p}_max"] = np.nanmax(data[p][i, mm]); row[f"{p}_days_ge15"] = int((data[p][i, mm] >= 15).sum())
        row["heat_days"] = int(flags["heat"][i, mm].sum())
        row["compound_days"] = int(flags["compound"][i, mm].sum())
        row["conflict_nights"] = int(flags["conflict"][i, mm].sum())
        ev.append(row)
    pd.DataFrame(ev).to_csv(os.path.join(RES, f"validation_events_{v}.csv"), index=False)
    ca = frame.index[prim & (frame["country"] == "United States") & frame["lon"].between(-124.5, -114) &
                     frame["lat"].between(32.5, 42)]
    y20 = days.year == 2020
    rows_ca = [{"city": frame.loc[i, "city"], "heat_2020": int(flags["heat"][i, y20].sum()),
                "smoke_2020": int(flags["smoke"][i, y20].sum()), "compound_2020": int(flags["compound"][i, y20].sum()),
                "smoke_qfed_2020": int((data["qfed"][i, y20] >= 15).sum()),
                "smoke_gfed_2020": int((data["gfed"][i, y20] >= 15).sum())} for i in ca]
    pd.DataFrame(rows_ca).to_csv(os.path.join(RES, f"validation_california_2020_{v}.csv"), index=False)

    # ERA5 validation (Open-Meteo, 2019-2023)
    vrows = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "01_data", "raw", "era5_openmeteo", "*.json"))):
        uc = int(os.path.basename(fp)[:-5]); js = json.load(open(fp))
        e = pd.DataFrame(js["daily"]); e["time"] = pd.to_datetime(e["time"])
        i = frame.index[frame["uc_id"] == uc][0]
        mm = days >= "2019-01-01"
        c = pd.DataFrame({"time": days[mm], "tmax": data["tmax"][i, mm], "tmin": data["tmin"][i, mm]})
        j = c.merge(e, on="time").dropna()

        def kappa(a1, a2):
            po = (a1 == a2).mean(); pe = a1.mean() * a2.mean() + (1 - a1.mean()) * (1 - a2.mean())
            return (po - pe) / (1 - pe) if pe < 1 else np.nan
        r = {"uc_id": uc, "city": frame.loc[i, "city"], "n_days": len(j),
             "tmax_r": j.tmax.corr(j.temperature_2m_max), "tmax_bias": (j.tmax - j.temperature_2m_max).mean(),
             "tmin_r": j.tmin.corr(j.temperature_2m_min), "tmin_bias": (j.tmin - j.temperature_2m_min).mean(),
             "heatday_kappa": kappa(j.tmax >= j.tmax.quantile(0.95), j.temperature_2m_max >= j.temperature_2m_max.quantile(0.95))}
        for L in (22, 24, 26):
            r[f"vent_agree_{L}"] = ((j.tmin < L) == (j.temperature_2m_min < L)).mean()
        vrows.append(r)
    if vrows:
        vv = pd.DataFrame(vrows); vv.to_csv(os.path.join(RES, f"validation_era5_{v}.csv"), index=False)
        tok("VAL_ERA5_N", len(vv), "cities")
        tok("VAL_ERA5_TMAX_R_MED", vv.tmax_r.median(), "median r", digits=2)
        tok("VAL_ERA5_TMAX_BIAS_MED", vv.tmax_bias.median(), "median bias, C (CPC minus ERA5)", digits=1)
        tok("VAL_ERA5_TMIN_R_MED", vv.tmin_r.median(), "median r", digits=2)
        tok("VAL_ERA5_TMIN_BIAS_MED", vv.tmin_bias.median(), "median bias, C", digits=1)
        tok("VAL_ERA5_KAPPA_MED", vv.heatday_kappa.median(), "median kappa, heat-day flags", digits=2)
        tok("VAL_ERA5_KAPPA_MIN", vv.heatday_kappa.min(), "min kappa", digits=2)
        tok("VAL_ERA5_VENT24_MED", 100 * vv.vent_agree_24.median(), "% agreement, Tmin < 24 C", digits=0)

    # ---- review-response outputs (14 Nov 2026; descriptive, from the primary results)
    pop = frame["pop2020"].values
    cn = cy["conflict"][prim][:, rec].mean(1)          # conflict nights per year per city, 2019-2023
    an = cy["activation"][prim][:, rec].mean(1)
    pn = cn * pop[prim]                                  # person-nights per year
    tok("CONFLICT_PERSONNIGHTS_M", pn.sum() / 1e6, "million person-nights per year, 2019-2023", digits=0)
    g1 = cn >= 1
    tok("CONFLICT_GE1_POP_M", pop[prim][g1].sum() / 1e6, "million people in cities with >= 1 conflict night a year", digits=0)
    tok("CONFLICT_GE1_SHARE_PN", 100 * pn[g1].sum() / pn.sum(), "% of person-nights in those cities", digits=0)
    order = np.argsort(-pn); cum = np.cumsum(pn[order]) / pn.sum()
    nhalf = int(np.searchsorted(cum, 0.5) + 1)
    tok("CONFLICT_N_CITIES_HALF_PN", nhalf, "cities holding half of conflict person-nights")
    names = frame.loc[prim, "city"].values[order][:5]
    tok("CONFLICT_TOP_PN_CITIES", ", ".join(names[:-1]) + " and " + names[-1], "five cities with most conflict person-nights")
    # triage groups
    gA = g1; gB = (an >= 1) & ~g1; gC = ~(gA | gB)
    for tag, m in (("A", gA), ("B", gB), ("C", gC)):
        tok(f"TRIAGE_{tag}_N", int(m.sum()), "cities"); tok(f"TRIAGE_{tag}_POP_M", pop[prim][m].sum() / 1e6, "million people", digits=0)
    # income shares of population and of conflict person-nights
    inc = frame.loc[prim, "income_group"].values
    for gname in ("Low income", "Lower middle income", "Upper middle income", "High income"):
        m = inc == gname; G = gname.upper().replace(" ", "_")
        tok(f"SHARE_POP_{G}", 100 * pop[prim][m].sum() / pop[prim].sum(), "% of population", digits=1)
        tok(f"SHARE_PN_{G}", 100 * pn[m].sum() / pn.sum(), "% of conflict person-nights", digits=1)
    m = np.isin(inc, ["Low income", "Lower middle income"])
    tok("SHARE_POP_LIC_LMIC", 100 * pop[prim][m].sum() / pop[prim].sum(), "% of population", digits=0)
    tok("SHARE_PN_LIC_LMIC", 100 * pn[m].sum() / pn.sum(), "% of conflict person-nights", digits=0)
    # India within the South-East Asia Region
    ind = frame.loc[prim, "iso3"].values == "IND"; sear = frame.loc[prim, "who_region"].values == "SEAR"
    a_e = cy["activation"][prim][:, YEARS <= 2004].mean(1)
    tok("INDIA_N", int(ind.sum()), "Indian cities")
    tok("INDIA_SHARE_SEAR_POP", 100 * pop[prim][ind].sum() / pop[prim][sear].sum(), "% of SEAR population", digits=0)
    tok("INDIA_ACT_EARLY", (a_e[ind] * pop[prim][ind]).sum() / pop[prim][ind].sum(), "days per resident 2000-2004", digits=2)
    tok("INDIA_ACT_RECENT", (an[ind] * pop[prim][ind]).sum() / pop[prim][ind].sum(), "days per resident 2019-2023", digits=2)
    tok("INDIA_N_NONE_EARLY", int((a_e[ind] == 0).sum()), "Indian cities with no activation day in 2000-2004")
    # months of conflict nights for the 20 cities with most conflict nights
    mo = days.month.values
    idx_prim = np.where(prim)[0]
    top = idx_prim[np.argsort(-cn)[:20]]
    rows_m = []
    for i in top:
        c = flags["conflict"][i, rec_m]
        cnt = np.bincount(mo[rec_m][c], minlength=13)[1:] / 5
        rows_m.append({"city": frame.loc[i, "city"], "country": frame.loc[i, "country"],
                       **{m_: round(float(x), 1) for m_, x in zip(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], cnt)}})
    pd.DataFrame(rows_m).to_csv(os.path.join(RES, f"conflict_months_top20_{v}.csv"), index=False)
    # like-for-like trend under the Wu et al. definitions, 2000-2019
    yv = days.year.values
    shares, exposed = [], []
    for y in range(2000, 2020):
        my = yv == y
        hwp = ((hw & valid)[prim][:, my].sum(1) * pop[prim]).sum()
        cop = ((hw & sm_ & valid)[prim][:, my].sum(1) * pop[prim]).sum()
        shares.append(cop / hwp if hwp else np.nan)
        exposed.append(pop[prim][((hw & sm_ & valid)[prim][:, my].sum(1) > 0)].sum())
    shares, exposed = np.array(shares), np.array(exposed)
    yrs20 = np.arange(2000, 2020)
    tok("WU_TREND_SHARE_RATIO", np.nanmean(shares[yrs20 >= 2015]) / np.nanmean(shares[yrs20 <= 2004]), "ratio 2015-2019 to 2000-2004", digits=2)
    tok("WU_TREND_EXPOSED_RATIO", exposed[yrs20 >= 2015].mean() / exposed[yrs20 <= 2004].mean(), "ratio of people in cities with >= 1 co-exposure day", digits=2)
    tok("WU_EXPOSED_M_2015_2019", exposed[yrs20 >= 2015].mean() / 1e6, "million people a year", digits=0)
    # level shares under the alternative joint rule
    sm = sens.set_index("analysis")
    r_ = sm.loc["Joint level = max(H, S)"]
    tok("MAXRULE_LEVEL3_PW", r_["level3_pw_recent"], "days per resident per year", digits="auto")
    tok("MAXRULE_LEVEL2_SHARE", 100 * r_["level2_pw_recent"] / r_["activation_pw_recent"], "%", digits=0)
    tok("MAXRULE_LEVEL3_SHARE", 100 * r_["level3_pw_recent"] / r_["activation_pw_recent"], "%", digits=0)

    # ---- additions after the review of V0_3 (14 Nov 2026; deviation D14)
    # (a) background total PM2.5 (annual, ACAG V6.GL.02.02 via the UCDB) and the fire-only counts
    bg = frame.loc[prim, "pm25_total_2020"].values
    for th in (15, 35):
        m = bg >= th
        tok(f"BG{th}_NCITY", int(m.sum()), f"cities with annual total PM2.5 >= {th} in 2020")
        tok(f"BG{th}_SHARE_PN", 100 * pn[m].sum() / pn.sum(), f"% of conflict person-nights in cities with background >= {th}", digits=0)
        tok(f"BG{th}_SHARE_POP", 100 * pop[prim][m].sum() / pop[prim].sum(), "% of population", digits=0)
    tok("BG_MEDIAN_TOP61", float(np.median(bg[g1])), "ug/m3, median annual total PM2.5 2020 in the cities with >= 1 conflict night", digits=0)
    firemean = data["mean"][prim][:, np.asarray(days.year == 2020)].mean(1)
    tok("FIRE_SHARE_TOTAL_MEDIAN", 100 * float(np.median(firemean / bg)), "% median fire-sourced share of annual total PM2.5, 2020", digits=0)
    # (b) threshold curve for a figure panel
    curve = []
    for thr in (2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35.5, 45, 55.5):
        cfg = {**PRIMARY, "smoke": float(thr)}
        fl_t, *_ = daily_flags(cfg, data, pct["1991-2020"]); c_t = city_year(fl_t, days)
        row = {"smoke_threshold": thr}
        for k in ("activation", "conflict"):
            st_, _ = summarise(c_t, frame, cfg, k)
            row[f"{k}_pw"] = st_["pw_recent"]; row[f"{k}_ncity_ge1"] = st_["n_ge1_recent"]
        curve.append(row)
    pd.DataFrame(curve).to_csv(os.path.join(RES, f"threshold_curve_{v}.csv"), index=False)
    # (c) multi-day runs of activation days and conflict nights, 2019-2023
    def run_share(fl, minlen):
        tot = inrun = 0
        for row in fl[prim][:, rec_m]:
            r_ = row.astype(int); tot += r_.sum()
            d_ = np.diff(np.concatenate([[0], r_, [0]])); st = np.where(d_ == 1)[0]; en = np.where(d_ == -1)[0]
            ln = en - st; inrun += ln[ln >= minlen].sum()
        return 100 * inrun / tot if tot else np.nan
    for k in ("activation", "conflict"):
        for L in (2, 3):
            tok(f"RUN{L}_{k.upper()}_PCT", run_share(flags[k], L), f"% of {k} days in runs of >= {L} consecutive days, 2019-2023", digits=0)
    # (d) where the 43 step-change cities are
    sf = frame["primary_all"].values & step2
    for col, tag in (("who_region", "REG"), ("income_group", "INC")):
        vc = frame.loc[sf, col].value_counts()
        tok(f"STEP_{tag}_LIST", "; ".join(f"{k_} {v_}" for k_, v_ in vc.items()), "step-change cities by group")
    tok("STEP_N_AFR", int((sf & (frame["who_region"].values == "AFR")).sum()), "step-change cities in the African Region")
    # (e) monthly heat-day frequency (1991-2020 baseline) for the season-overlap check with fire emissions
    mo_t = tdays.month.values; inb = np.asarray((tdays >= P_BASE[0]) & (tdays <= P_BASE[1]))
    hot = data["tmax_all"] >= pct["1991-2020"][95][:, None]
    mh = pd.DataFrame({"uc_id": frame["uc_id"].values,
                       **{f"m{m_:02d}": np.nanmean(np.where(np.isfinite(data["tmax_all"][:, inb & (mo_t == m_)]), hot[:, inb & (mo_t == m_)], np.nan), 1) for m_ in range(1, 13)}})
    mh.to_csv(os.path.join(RES, f"heat_month_freq_{v}.csv"), index=False)

    pd.DataFrame(NUM).to_csv(os.path.join(RES, f"P44x_NUMBERS_{v}.csv"), index=False)
    print("tokens", len(NUM))


if __name__ == "__main__":
    main()
