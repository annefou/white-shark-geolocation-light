# Snakefile — orchestrates the replication pipeline end-to-end.
#
# Replace the placeholder rules with your actual replication steps. The
# canonical pattern is one rule per pipeline stage, and each rule wraps a
# notebook executed via jupytext (so the notebook stays the source of truth
# and the Snakefile just sequences them).
#
# Usage:
#   snakemake --cores 1                  # run everything
#   snakemake --cores 1 -n               # dry run

NOTEBOOKS = "notebooks"
DATA = "data"
RESULTS = "results"
FIGURES = "figures"


rule all:
    input:
        # Replace with your actual final artefacts:
        f"{FIGURES}/main_result.png",
        f"{RESULTS}/summary.csv",


# ---------- 01: Data download ----------
# Every replication MUST be self-contained: data is downloaded by the notebook,
# never assumed to exist locally. See CLAUDE.md § Self-contained data.
# Downloads (a) the JWS deployment-metadata table + every tag's PAT/SPOT ZIP
# package from the DataONE archive (DOI 10.24431/rw1k6c3), (b) a regional ETOPO
# 2022 15-arc-second bathymetry subset (NOAA NCEI, no credentials), and (c)
# GLORYS12V1 thetao+zos over the baseline-free species-range box for each
# temperature tag (Copernicus Marine; needs a credential). The GLORYS fetch is
# idempotent — tags whose reference model is already built are skipped.
rule data_download:
    output:
        f"{DATA}/raw/JWS_metadata.csv",
        f"{DATA}/raw/sources.json",
        f"{DATA}/raw/etopo2022_15s_nepac.nc",
    shell:
        f"cd {{NOTEBOOKS}} && jupytext --to notebook --execute 01_data_download.py"


# ---------- 02: Data clean ----------
# Per-tag twilight detections + tagging events + GPE3 baseline + Argos referee,
# the native DST (depth-temperature) log, and the baseline-free GLORYS reference
# model assembled from the raw subset downloaded by notebook 01. Self-contained:
# no sibling repository.
rule data_clean:
    input:
        f"{DATA}/raw/JWS_metadata.csv",
    output:
        f"{DATA}/clean/clean_status.json",
        f"{DATA}/clean/tags/07_05/dst.csv",
        f"{DATA}/clean/reference_model_07_05_gulfext.nc",
    shell:
        f"cd {{NOTEBOOKS}} && jupytext --to notebook --execute 02_data_clean.py"


# ---------- 03: Analysis ----------
# FUSED geolocation HMM per tag (light x temperature x land x depth-floor) on the
# canonical BASELINE-FREE species-range grid, reading this repo's own GLORYS
# reference models + native DST logs; validates against the held-out Argos referee
# (pooled ~148 km; per-tag 07_05~161, 08_01~62, 08_02~206, 08_09~136).
rule analysis:
    input:
        f"{DATA}/clean/clean_status.json",
        f"{DATA}/clean/reference_model_07_05_gulfext.nc",
        f"{DATA}/clean/tags/07_05/dst.csv",
        f"{DATA}/raw/etopo2022_15s_nepac.nc",
    output:
        f"{RESULTS}/summary.csv",
        f"{RESULTS}/aggregate.json",
        f"{RESULTS}/barrier_verification.json",
        f"{RESULTS}/posterior_07_05.nc",
        f"{RESULTS}/fused_track_07_05.csv",
        f"{DATA}/clean/bathymetry_healpix_L9.nc",
    shell:
        f"cd {{NOTEBOOKS}} && jupytext --to notebook --execute 03_analysis.py"


# ---------- 04: Figures ----------
# Headline main_result.png (ours vs GPE3 + sigma-off-bound) + per-tag posterior
# cloud maps with a 10 m coastline.
rule figures:
    input:
        f"{RESULTS}/summary.csv",
        f"{RESULTS}/aggregate.json",
        f"{RESULTS}/posterior_07_05.nc",
    output:
        f"{FIGURES}/main_result.png",
        f"{FIGURES}/tracks/cloud_07_05.png",
    shell:
        f"cd {{NOTEBOOKS}} && jupytext --to notebook --execute 04_figures.py"
