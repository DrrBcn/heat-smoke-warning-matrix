# A joint heat and fire-smoke warning matrix for 453 cities: code and results

Code, analysis plan and results for: Rojas-Rueda D. *A joint heat and fire-smoke warning matrix for 453 cities* (manuscript in preparation).

The study gives numeric cells to the threshold matrix of heat and air quality requested in Annex 1 of the WHO Regional Office for Europe heat-health action plan guidance (2nd edition, 2026), adds a rule for when night ventilation should yield to smoke, and counts how often and where the matrix would have applied in the urban centres of one million or more people worldwide, 2000 to 2023.

## Contents

| Path | What it holds |
|---|---|
| `02_code/` | All analysis code (Python 3.12); `run_analysis.py` runs the whole pipeline and writes every result to `03_results/` |
| `ANALYSIS_PLAN.md` | Analysis plan, reproduced verbatim as fixed on 14 November 2026 before any fire-sourced PM2.5 value was extracted (it mentions internal project files that are not part of this release); later changes are listed in Supplementary Note 3 of the paper |

## Inputs (all public; not redistributed here)

- Fire-sourced PM2.5, Hu et al. 2025 (ESSD): GFED-based doi:10.5281/zenodo.15493914 and QFED-based doi:10.5281/zenodo.15496596 (CC BY 4.0)
- NOAA CPC Global Unified daily temperature (0.5 degree), https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html
- ISIMIP3a 20CRv3-ERA5 obsclim and counterclim daily temperature, doi:10.48364/ISIMIP.982724.3 (CC0)
- GFED4.1s monthly fire emissions, https://www.geo.vu.nl/~gwerf/GFED/GFED4/
- GHSL Urban Centre Database R2024A V1.2, doi:10.2905/1a338be6-7eaf-480c-9664-3a8ade88cbcd (CC BY 4.0)
- World Bank income classification (CLASS.xlsx), WHO Global Health Observatory regions, Natural Earth land outline
- ERA5 daily temperature for 20 validation cities via the Open-Meteo archive API (CC BY 4.0)

## Reproduce

```bash
pip install -r 02_code/requirements.txt
python 02_code/run_analysis.py --download --version V0_4
```

About 120 GB of downloads and 2 to 4 hours on a desktop computer; without `--download`, place the inputs in `01_data/raw/` with the file names given in the download scripts (also listed in Supplementary Table 5 of the paper). The run writes `03_results/P44x_NUMBERS_V0_4.csv`, which holds every number reported in the paper, together with the tables and figures.


## Licence 

Code: MIT licence (`LICENSE`). Input data keep their own licences.
