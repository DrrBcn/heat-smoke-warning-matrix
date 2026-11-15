"""Reproduce every result of the study from public inputs.

Usage: python 02_code/run_analysis.py [--download] [--version V0_4]
  --download  fetch all inputs first (about 120 GB: fire PM2.5 62 GB, ISIMIP3a 17 GB, CPC 6 GB, GFED4.1s 2 GB)
Steps: city frame, city series (about 1 hour), analysis, fire-emissions check, reanalysis and counterfactual,
figures. Results go to 03_results/.
"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "V0_4"


def run(*a):
    print(">>", " ".join(a), flush=True)
    if subprocess.run([sys.executable, "-W", "ignore", *a], cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode:
        sys.exit("failed: " + " ".join(a))


if "--download" in sys.argv:
    run("02_code/00_download_inputs.py")
    run("02_code/01_download_fire.py", "gfed", "8"); run("02_code/01_download_fire.py", "qfed", "6")
    run("02_code/02_download_temperature.py")
    run("02_code/10_download_isimip_gfed4.py", "isimip", "8"); run("02_code/10_download_isimip_gfed4.py", "gfed4", "4")
run("02_code/03_build_city_frame.py")
run("02_code/05_download_era5_validation.py") if "--download" in sys.argv else None
for step in (["weights"], ["temp"], ["fire", "gfed"], ["fire", "qfed"]):
    run("02_code/04_extract_city_series.py", *step)
run("02_code/06_analysis.py", V)
run("02_code/11_fire_activity_check.py", V)
run("02_code/12_isimip_analysis.py", "extract")
run("02_code/12_isimip_analysis.py", "analyse", V)
run("02_code/07_figures.py", V)
print("done; results in 03_results/")
