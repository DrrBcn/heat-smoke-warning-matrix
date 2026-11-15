"""Figures for P44x from files in 03_results (colour-blind safe palettes; 300 dpi PNG).

Usage: python 02_code/07_figures.py V0_1
Fig 1  the joint matrix: heat tier by fire-smoke band, joint levels (specification E1) with population-weighted
       days per resident per year in each cell, 2019-2023; panel b the conflict-night rule (E3).
Fig 2  maps of mean annual matrix activation days and conflict nights per city, 2019-2023.
Fig 3  population-weighted annual activation days and conflict nights by WHO region, 2000-2023.
Fig 4  sensitivity of activation days and conflict nights (days per resident per year, 2019-2023).
"""
import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("P44X_RES", os.path.join(ROOT, "03_results"))
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
LEVEL_COL = {0: "#f2f2f2", -1: "#d9d9d9", 1: "#FDE0A8", 2: "#F4A259", 3: "#B5432A"}  # sequential, readable in grey
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})
REG_NAME = {"AFR": "African", "AMR": "Americas", "EMR": "Eastern Mediterranean", "EUR": "European",
            "SEAR": "South-East Asia", "WPR": "Western Pacific"}


def level(h, s):
    if h >= 1 and s >= 1:
        sc = h + s
        return 3 if sc >= 5 else 2 if sc >= 3 else 1
    return -1 if (h >= 1 or s >= 1) else 0


def fig1(v):
    cells = pd.read_csv(os.path.join(RES, f"matrix_cells_{v}.csv"))
    fig = plt.figure(figsize=(7.2, 3.4))
    ax = fig.add_axes([0.13, 0.15, 0.40, 0.75])
    hl = ["H0\n< P90", "H1\nP90 to < P95", "H2\nP95 to < P97.5", "H3\n≥ P97.5"]
    sl = ["S0\n< 15", "S1\n15 to < 35.5", "S2\n35.5 to < 55.5", "S3\n≥ 55.5"]
    names = {0: "No action", -1: "Single-hazard plan", 1: "Joint level 1\nadvisory", 2: "Joint level 2\nwarning",
             3: "Joint level 3\nemergency"}
    for _, c in cells.iterrows():
        h, s = int(c.H), int(c.S)
        L = level(h, s)
        ax.add_patch(plt.Rectangle((s, h), 1, 1, facecolor=LEVEL_COL[L], edgecolor="white", lw=1.5))
        val = c.pw_days_recent
        txt = f"{val:.1f}" if val >= 0.1 else (f"{val:.2f}" if val >= 0.01 else "<0.01")
        ax.text(s + 0.5, h + 0.62, txt, ha="center", va="center", fontsize=8,
                color="white" if L == 3 else "black", fontweight="bold")
        lab = {1: "Level 1", 2: "Level 2", 3: "Level 3", -1: "single hazard", 0: ""}[L]
        ax.text(s + 0.5, h + 0.3, lab, ha="center", va="center", fontsize=5.5,
                color="white" if L == 3 else "black")
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(4) + 0.5); ax.set_xticklabels(sl, fontsize=6)
    ax.set_yticks(np.arange(4) + 0.5); ax.set_yticklabels(hl, fontsize=6)
    ax.set_xlabel("Smoke band: fire-sourced PM$_{2.5}$, 24-h mean (µg m$^{-3}$)")
    ax.set_ylabel("Heat tier: daily maximum temperature\n(city percentile, 1991–2020)")
    ax.set_title("a  Joint matrix: days per resident per year, 2019–2023", loc="left", fontsize=7.5,
                 fontweight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    # panel b: conflict rule as text flow
    bx = fig.add_axes([0.56, 0.06, 0.43, 0.86]); bx.axis("off")
    bx.set_title("b  Night rule on a hot, smoky day", loc="left", fontsize=7.5, fontweight="bold")
    steps = [
        ("Heat day (Tmax ≥ city P95) and Tmax ≥ 26 °C,\nwith fire PM$_{2.5}$ ≥ 15 on the day or the next", 0.86),
        ("Next-morning Tmin < 24 °C?", 0.63),
        ("Yes: fire-smoke conflict night\n\nOpen windows let smoke in;\nclosed windows give up cooling.\nUse a cool, filtered room,\na cleaner-air space or\nfiltered night ventilation", 0.30),
        ("No: warm night\n\nOpen windows cannot bring\nthe room below 24 °C and\nstill admit smoke. Go to a\ncooled, filtered space;\ncheck on people at risk", 0.30),
    ]
    def box(x, y, w, h, t, fc):
        bx.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=fc, ec="#555555", lw=0.6))
        bx.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=5.8)
    box(0.02, 0.78, 0.96, 0.15, steps[0][0], "#f2f2f2")
    box(0.22, 0.58, 0.56, 0.1, steps[1][0], "#ffffff")
    box(0.02, 0.08, 0.46, 0.38, steps[2][0], LEVEL_COL[2])
    box(0.52, 0.08, 0.46, 0.38, steps[3][0], LEVEL_COL[1])
    for x0, x1 in ((0.5, 0.5),):
        bx.annotate("", xy=(0.5, 0.68), xytext=(0.5, 0.78), arrowprops=dict(arrowstyle="->", lw=0.6))
    bx.annotate("", xy=(0.24, 0.46), xytext=(0.4, 0.58), arrowprops=dict(arrowstyle="->", lw=0.6))
    bx.annotate("", xy=(0.76, 0.46), xytext=(0.6, 0.58), arrowprops=dict(arrowstyle="->", lw=0.6))
    fig.savefig(os.path.join(RES, f"Fig1_matrix_{v}.png"), dpi=300)
    plt.close(fig)


def fig2(v):
    ct = pd.read_csv(os.path.join(RES, f"city_table_{v}.csv"))
    land = gpd.read_file(os.path.join(ROOT, "01_data", "raw", "naturalearth", "ne_110m_land.shp"))
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 6.2))
    for ax, col, title, vmax in ((axs[0], "activation_2019_2023", "a  Matrix activation days per year (fire-sourced PM2.5)", None),
                                 (axs[1], "conflict_2019_2023", "b  Conflict nights per year (fire-sourced PM2.5)", None)):
        land.plot(ax=ax, color="#e6e6e6", edgecolor="none")
        z = ct[ct[col] < 0.2]; nz = ct[ct[col] >= 0.2].sort_values(col)
        size = lambda p: 4 + 60 * np.sqrt(p / ct.pop2020.max())
        ax.scatter(z.lon, z.lat, s=size(z.pop2020), facecolor="none", edgecolor="#9a9a9a", lw=0.4)
        vm = np.nanpercentile(nz[col], 98) if len(nz) else 1
        sc = ax.scatter(nz.lon, nz.lat, s=size(nz.pop2020), c=nz[col], cmap="viridis", vmin=0, vmax=vm,
                        edgecolor="black", lw=0.2)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01, extend="max")
        cb.set_label("Mean per year, 2019–2023", fontsize=6); cb.ax.tick_params(labelsize=6)
        ax.set_xlim(-170, 180); ax.set_ylim(-58, 75); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title + " (open circles: < 0.2)", loc="left", fontsize=7.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"Fig2_maps_{v}.png"), dpi=300)
    plt.close(fig)


def fig3(v):
    rs = pd.read_csv(os.path.join(RES, f"region_series_{v}.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, k, t in ((axs[0], "activation", "a  Matrix activation days"), (axs[1], "conflict", "b  Conflict nights")):
        for i, (g, d) in enumerate(rs[rs.outcome == k].groupby("who_region")):
            pal = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00"]
            ax.plot(d.year, d.value, color=pal[i], lw=1.1, label=REG_NAME.get(g, g))
        ax.set_title(t + " per resident per year", loc="left", fontsize=7.5, fontweight="bold")
        ax.set_xlim(2000, 2023); ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Year")
    axs[0].set_ylabel("Population-weighted days")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, ncol=6, loc="lower center", title="WHO region", title_fontsize=6)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(RES, f"Fig3_regions_{v}.png"), dpi=300)
    plt.close(fig)


def fig_threshold(v):
    """Fig. 4: activation days and fire-smoke conflict nights against the smoke threshold of the first column."""
    c = pd.read_csv(os.path.join(RES, f"threshold_curve_{v}.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, k, t, lim in ((axs[0], "activation", "a  Activation days", 35.5), (axs[1], "conflict", "b  Fire-smoke conflict nights", 55.5)):
        d = c[c.smoke_threshold <= lim]
        ax.plot(d.smoke_threshold, d[f"{k}_pw"], color="#0072B2", marker="o", ms=3, lw=1.1, label="Days per resident per year")
        ax2 = ax.twinx()
        ax2.plot(d.smoke_threshold, d[f"{k}_ncity_ge1"], color="#D55E00", marker="s", ms=3, lw=1.1, ls="--", label="Cities with at least 1 a year")
        for x in (15, 25, 35.5):
            ax.axvline(x, color="#999999", lw=0.5, ls=":")
        ax.set_xlabel("Smoke threshold, fire-sourced PM$_{2.5}$ (µg m$^{-3}$)")
        ax.set_ylabel("Days per resident per year", color="#0072B2")
        ax2.set_ylabel("Cities with at least one a year", color="#D55E00")
        ax.set_title(t, loc="left", fontsize=7.5, fontweight="bold")
        ax.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
        ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"Fig4_threshold_{v}.png"), dpi=300)
    plt.close(fig)


def fig4(v):
    s = pd.read_csv(os.path.join(RES, f"sensitivity_{v}.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    y = np.arange(len(s))[::-1]
    for ax, k, t in ((axs[0], "activation", "a  Matrix activation days"), (axs[1], "conflict", "b  Conflict nights")):
        prim = s.loc[s.analysis == "Primary", f"{k}_pw_recent"].iloc[0]
        ax.axvline(prim, color="#999999", lw=0.6, ls="--")
        ax.scatter(s[f"{k}_pw_recent"], y, s=12, color=OKABE[4])
        ax.set_title(t, loc="left", fontsize=7.5, fontweight="bold")
        ax.set_xlabel("Days per resident per year, 2019–2023")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(left=0)
    axs[0].set_yticks(y); axs[0].set_yticklabels(s.analysis, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"ED1_sensitivity_{v}.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "V0_1"
    for f in (fig1, fig2, fig3, fig4, fig_threshold):
        f(v)
    print("figures written")
