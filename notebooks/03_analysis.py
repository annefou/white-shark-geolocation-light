# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 — Analysis: FUSED open geolocation HMM (light × temperature × land × depth)
#
# This is the scientific result of the **Replication Study**. The prior light-only
# run of this repository reached a pooled median error of **441 km** to the Argos
# referee; the prior temperature-only chain reached **276 km**. Neither single
# physical signal alone matches the proprietary **GPE3** product (**54 km** median
# to Argos on these four referee tags).
#
# Here we **fuse** several independent likelihood factors into a single
# pangeo-fish HMM emission, then run the **identical** low-level σ-fit
# (`EagerEstimator` + `EagerBoundsSearch`) and decode the per-day **MAP** state
# with `decode(mode="mode")` (the posterior argmax cell per day).
#
# ### Why mode, not Viterbi (a hard pangeo-fish constraint)
#
# The original plan was to decode with the Viterbi most-probable path. In
# pangeo-fish **2026.4.0**, however, `EagerEstimator.decode(mode="viterbi")` (and
# `"viterbi2"`) is implemented **only for the structured 2-D `(x, y)` projected
# emission grid** — it calls `gaussian_kernel(shape=(2,))` and indexes the pdf as
# a 2-D image with `input_core_dims=[("x","y"), ...]`. This whole pipeline (and
# both sibling chains) runs on a **1-D HEALPix `cells`** emission with the
# `Gaussian1DHealpix` convolution predictor. Calling Viterbi on the HEALPix
# emission raises `ValueError: tuple.index(x): x not in tuple` because the decoder
# cannot find the `x`/`y` dims. Viterbi is therefore **not available** on the
# HEALPix grid without abandoning the HEALPix state space.
#
# We use the **mode** (per-day maximum-a-posteriori cell) instead. The mode shares
# Viterbi's decisive property — and the property the posterior *mean* lacks: it
# returns an actual posterior-supported cell, so on a multimodal day it picks one
# mode rather than averaging two modes onto the coastline between them, and because
# masked land cells carry zero posterior it can **never** select a banned land
# cell. It differs from Viterbi only in that it is a per-day MAP rather than a
# jointly-most-probable *path* (no inter-day transition smoothing in the decode;
# the σ-Brownian prior already shapes the forward-backward posterior the mode reads
# from). The emission fusion mirrors both prior notebooks:
#
# ```
# emission_fused[day, cell] = normalize(
#       light_pdf × temperature_pdf × land_mask × depth_emission )
# ```
#
# This reproduces GPE3's ingredient list (light + sea-surface/at-depth temperature
# + a movement prior) **plus** a bathymetric constraint that GPE3 does not use.
#
# ### This run — two changes over the fused-barrier baseline (189.6 km pooled)
#
# 1. **Bathymetric depth EMISSION (Change 1).** The old binary depth *floor* (cell
#    allowed iff seabed ≥ max-dive − tol) is replaced by a **soft, floor-aware
#    likelihood** `sigmoid((S − D)/τ)` over (seabed S − day's max dive D), τ = 75 m.
#    It downweights only physically-impossible too-shallow cells and never
#    penalises a deep cell on a shallow-diving day (see the depth-emission section).
# 2. **Relaxed coastal land mask (Change 2).** A cell is water if its land fraction
#    ≤ 0.8 **or** its footprint minimum elevation ≤ +10 m, so the shallow Bahía
#    Sebastián Vizcaíno nursery lagoon (which ETOPO renders as +3..+5 m land) is no
#    longer excluded. The dense full-resolution ETOPO segment tests in the barrier
#    kernel and the path Viterbi are unchanged, so no real land-crossing returns.
#
# ## Factors
#
# - **temperature_pdf** — built with the *exact* sibling path
#   (`pangeo_fish.helpers.compute_diff → regrid_dataset → compute_emission_pdf →
#   normalize_pdf`, `differences_std=0.75`, `relative_depth_threshold=0.8`) against
#   the GLORYS12V1 `thetao` 3-D field. The temperature emission's HEALPix cell set
#   (level 9, NESTED) and daily time axis are the **canonical grid** for fusion;
#   its `initial`/`final` endpoint anchors are reused unchanged.
# - **light_pdf** — the twilight emission of this repo's prior notebook: each
#   detected sunrise/sunset event is scored at every cell by a Gaussian
#   (`SIGMA_T_MIN=20 min`) over the astropy-modelled-minus-detected time residual.
#   Events are **binned to days** (the product of a day's events, renormalised);
#   days with no twilight default to uniform.
# - **land_mask** — from NOAA NCEI **ETOPO 2022 v1 15-arc-second** relief, regridded
#   to the level-9 cells. RELAXED (Change 2): a cell is water if its land fraction
#   (`elevation ≥ 0`) ≤ 0.8 **or** its footprint minimum elevation ≤ +10 m, so
#   shallow lagoon/tidal-flat nursery cells are not excluded.
# - **depth_emission** — daily soft floor (Change 1): `sigmoid((seabed − max-dive)/τ)`
#   with τ = 75 m, replacing the old binary floor. ~1 where the seabed is deeper
#   than the dives, →0 only where it is impossibly shallower.
#
# ## Anti-circularity
#
# The only spatial anchors fed to the fit are the release/pop-up **endpoints**.
# **Argos is never fed to the fit** — it is the held-out referee only. GPE3 is a
# comparison baseline, never ground truth. Light, temperature and bathymetry are
# three independent physical observations, none derived from the referee.
#
# ## Tag scope
#
# - **Referee tags** `07_05, 08_01, 08_02, 08_09` — full fusion (GLORYS on disk +
#   DST + Argos referee). These carry the headline.
# - `07_01` — has GLORYS + DST but no referee → full fusion, best-effort.
# - `06_10` — basin-scale roamer, 2 GB GLORYS field → memory risk; full fusion at
#   a reduced level if needed, else best-effort drop with the exact reason.
# - `02_01` — PAT2 with no external water-temperature sensor (no DST, no GLORYS) →
#   fusion = light × land only (no SST factor, no depth floor). No referee.

# %% [markdown]
# ## Self-contained temperature factor (baseline-free species-range grid)
#
# The temperature emission reads **this repository's own** data: the GLORYS12V1
# reference models `data/clean/reference_model_<tag>_gulfext.nc` (built by notebook
# 02 from the species-range GLORYS subset downloaded by notebook 01) and the
# natively-cleaned DST logs `data/clean/tags/<tag>/dst.csv` (notebook 02). There is
# **no sibling-repository dependency** — the analysis grid is the baseline-free
# juvenile-white-shark species range `lon[-125,-106]`, `lat[22,38]` for every
# temperature tag (NOT a GPE3/Argos-derived box), which is what makes the held-out
# Argos comparison baseline-free.

# %%
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pint
import sparse  # noqa: F401  (sparse.COO rebuild in LandBarrierGaussian1DHealpix)
import xarray as xr
import xdggs  # noqa: F401  (registers the .dggs accessor)
import healpix_geo.nested as hgn
from astropy.coordinates import get_sun
from astropy.time import Time

from pangeo_fish.distributions import healpix as distrib_hp
from pangeo_fish.helpers import (
    compute_diff,
    compute_emission_pdf,
    load_tag,
    normalize_pdf,
    regrid_dataset,
    to_healpix,
)
from pangeo_fish.hmm.estimator import EagerEstimator
from pangeo_fish.hmm.optimize import EagerBoundsSearch
from pangeo_fish.hmm.prediction import Gaussian1DHealpix
from tlz.functoolz import curry
from xarray.namedarray._typing import _arrayfunction_or_api as _ArrayLike

warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%
CLEAN_DIR = Path("../data/clean")
RAW_DIR = Path("../data/raw")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)

# This repo's own per-tag clean root (DST + tagging events) for pangeo-fish
# `load_tag`. Baseline-free GLORYS reference models live alongside as
# reference_model_<tag>_gulfext.nc. No sibling repository is read.
TAG_ROOT = str((CLEAN_DIR / "tags").resolve())

# Bathymetry: NOAA NCEI ETOPO 2022 v1 15-arc-second (~450 m) regional subset,
# downloaded by notebook 01 from the CoastWatch ERDDAP griddap dataset
# `ETOPO_2022_v1_15s` (variable `z`, EGM2008 height, lon -180..180).
# https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.html
BATHY_NC = RAW_DIR / "etopo2022_15s_nepac.nc"

# Registry: shark_id -> (PAT_DEPLOY_ID, SPOT_DEPLOY_ID or None, has_referee).
TAGS = {
    "07_05": ("07_05-66885", "07_05-77272", True),
    "08_01": ("08_01-40561", "08_01-77274", True),
    "08_02": ("08_02-55716", "08_02-77273", True),
    "08_09": ("08_09-83066", "08_09-83076", True),
    "02_01": ("02_01-18616", None, False),
    "06_10": ("06_10-40564", None, False),
    "07_01": ("07_01-64272", None, False),
}

# Tags whose temperature factor uses GLORYS (a native DST log
# data/clean/tags/<tag>/dst.csv and reference_model_<tag>_gulfext.nc must both
# exist, built by notebooks 01/02). 02_01 has no external thermistor, so it runs
# light × land only.
HAS_TEMPERATURE = {"07_05", "08_01", "08_02", "08_09", "07_01", "06_10"}

# Emission version tag. Bump this whenever the emission recipe OR the analysis grid
# changes so a σ cached against an older recipe/grid is NOT reused (a stale σ is no
# longer the bounds-search optimum, and σ is grid-dependent). v4 = light ×
# temperature × relaxed-land × SOFT bathy depth emission using the cell's DEEPEST
# seabed point (Changes 1 & 2 with the deepest-point fix). The `_baselinefree`
# suffix marks the canonical BASELINE-FREE species-range grid (lon[-125,-106]
# lat[22,38]) used for ALL temperature tags — distinct from the earlier GPE3-sized
# grid, so any GPE3-grid σ cache is invalidated and a fresh σ-fit is forced.
EMISSION_VERSION = "v4_softbathy_deepest_relaxedmask_baselinefree"

# HEALPix level for the state space — level 9 (~6.4 km cells), NESTED (DOMAIN.md).
# A per-tag override drops the basin-roaming 06_10 to level 8 so its 2 GB GLORYS
# field fits the single-kernel memory budget (stated deviation; recorded per tag
# in summary.csv as healpix_level).
DEFAULT_HEALPIX_LEVEL = 9
HEALPIX_LEVEL_OVERRIDE: dict[str, int] = {"06_10": 8}

# --- emission hyperparameters ---
# Light: Gaussian spread (minutes) over the twilight time residual. 20 min is the
# proven value from this repo's light-only run (open analogue of probGLS/FLightR
# "twilight error").
SIGMA_T_MIN = 20.0
SOLAR_HORIZON_DEG = -0.833  # geometric sunrise/sunset (refraction + semi-diameter)

# Temperature: sibling parameters, used verbatim so the temperature factor is the
# same likelihood the 276 km chain used.
DIFFERENCES_STD = 0.75
RELATIVE_DEPTH_THRESHOLD = 0.8

# Bathymetry: land-fraction cut and depth-floor tolerance (deviations, stated).
#
# CHANGE 2 — relaxed coastal land mask (shallow nursery habitat).
# A cell is WATER (mask 1) iff EITHER
#   (i)  its ETOPO land fraction <= LAND_FRAC_THRESHOLD, OR
#   (ii) its MINIMUM elevation over the footprint <= MIN_ELEV_WATER_CUTOFF_M.
# Rule (ii) captures tidal-flat / lagoon habitat that ETOPO renders as a thin
# +3..+5 m "land" sliver (e.g. the Bahía Sebastián Vizcaíno nursery at
# -114.05/27.98 where 46 good Argos fixes sit): if ANY part of the cell footprint
# is at or below +10 m it is treated as usable shallow water. The land_fraction
# cut is also relaxed 0.5 -> 0.8 so a cell that is mostly land but still partly
# water/lagoon is admitted. The DENSE ETOPO segment test in the barrier kernel and
# the path Viterbi remains at full 15-arc-second resolution, so relaxing the CELL
# candidate threshold cannot reintroduce a real land-crossing (re-verified to 0).
# Genuine high-elevation inland cells (footprint min well above +10 m) stay land.
LAND_FRAC_THRESHOLD = 0.8
MIN_ELEV_WATER_CUTOFF_M = 10.0
# Vizcaíno nursery centroid used only to CONFIRM the relaxed mask opens it (never
# fed to the fit).
VIZCAINO_LON, VIZCAINO_LAT = -114.05, 27.98
#
# CHANGE 1 — bathymetric depth EMISSION (replaces the binary depth floor).
# DEPTH_FLOOR_TOL_M is the sigmoid scale (metres) of the soft floor: a cell whose
# shallowest seabed S is deeper than the day's max dive D is fully allowed (factor
# ~1); a cell shallower than the dives is smoothly downweighted by
# sigmoid((S - D) / DEPTH_FLOOR_TOL_M), reaching ~0 only where the seabed is many
# tens of metres shallower than depths the animal demonstrably reached (physically
# impossible). 75 m absorbs ETOPO vertical RMS + pressure-sensor error in this band.
DEPTH_FLOOR_TOL_M = 75.0

# Endpoint-anchor spreads (HEALPix-distance sigma), matching the siblings.
INITIAL_STD = 1e-3
RECAPTURE_STD = 1e-3

# Brownian speed prior -> σ-search upper bound, matching the siblings.
MAX_SPEED = pint.Quantity(5.0, "km/h")
EARTH_RADIUS = pint.Quantity(6371.0, "km")
ADJUSTMENT_FACTOR = 5.0
TRUNCATE = 4.0
TOLERANCE = 1e-3

# Bbox padding (deg) for the light-only (02_01) state space.
BBOX_PAD_DEG = 4.0
MESH_STEP_DEG = 0.03

# Prior reference numbers (do NOT recompute — they feed the ablation table).
LIGHT_ONLY_ERR_KM = {"07_05": 563.0, "08_01": 454.0, "08_02": 427.0, "08_09": 334.0}
LIGHT_ONLY_POOLED_KM = 441.0
TEMP_ONLY_ERR_KM = {"07_05": 300.0, "08_01": 354.0, "08_02": 202.0, "08_09": 251.0}
TEMP_ONLY_POOLED_KM = 276.0
GPE3_BASELINE_KM = 54.0
# Fused land-barrier baseline (binary depth floor + strict mask), the run this
# experiment is measured against. Per-tag land-aware-path-vs-Argos medians and the
# pooled median, recorded so the ablation table carries the before/after delta.
FUSED_BARRIER_ERR_KM = {
    "07_05": 386.2, "08_01": 66.1, "08_02": 228.8, "08_09": 158.7}
FUSED_BARRIER_POOLED_KM = 189.6

clean_status = json.loads((CLEAN_DIR / "clean_status.json").read_text())


# %% [markdown]
# ## Great-circle + daily-lookup helpers (identical to the siblings)

# %%
def gc_km(lon1, lat1, lon2, lat2):
    """Great-circle distance (km) via the haversine formula."""
    r = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    d = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(d))


def daily_lookup(track: pd.DataFrame) -> pd.DataFrame:
    t = track.copy()
    t["day"] = pd.to_datetime(t["time"]).dt.floor("D")
    return t.groupby("day")[["longitude", "latitude"]].mean()


# %% [markdown]
# ## Bathymetry → HEALPix (relaxed land mask + per-cell seabed depths)
#
# Every ETOPO pixel is mapped to its level-`L` NESTED cell, then reduced per cell:
# the **land fraction** is the share of pixels with `elevation ≥ 0`; the
# **shallowest seabed** is `-max(elev)` (reporting only); the **deepest seabed**
# `max_ocean_depth_m = -min(elev)` is what the depth-emission floor reads (a dive
# of depth D fits iff SOME water in the cell is at least D deep — see the
# depth-emission section); and `min_elev = min(elev)` feeds the relaxed coastal
# water rule (ii). The relaxed mask + deepest-seabed floor are Changes 1 & 2.

# %%
def bathymetry_per_cell(
    cells: np.ndarray, level: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (land_fraction, min_ocean_depth_m, max_ocean_depth_m, min_elev_m).

    Aligned to ``cells``:
    - ``min_ocean_depth_m = -max(elev)`` — the cell's SHALLOWEST point (positive m
      below sea level; negative for a partly-land coastal cell). Kept for reporting.
    - ``max_ocean_depth_m = -min(elev)`` — the cell's DEEPEST point. This is the
      seabed the depth EMISSION floor uses: a dive of depth D fits in the cell iff
      SOME water in it is at least D deep, i.e. the deepest point is ≥ D. Using the
      shallowest point (the earlier bug) crushed partly-land coastal nursery cells
      whose shallowest "seabed" is actually land elevation.
    - ``min_elev_m = min(elev)`` — deepest footprint elevation, for the relaxed
      coastal-water rule (ii).
    """
    bz = xr.open_dataset(BATHY_NC)["z"]
    mlon, mlat = np.meshgrid(bz["longitude"].values, bz["latitude"].values)
    bcell = hgn.lonlat_to_healpix(
        mlon.ravel(), mlat.ravel(), level).astype("int64")
    bdf = pd.DataFrame({"cell": bcell, "elev": np.asarray(bz.values).ravel()})
    frac_land = bdf.assign(land=bdf["elev"] >= 0).groupby("cell")["land"].mean()
    max_elev = bdf.groupby("cell")["elev"].max()  # shallowest point / highest land
    min_elev = bdf.groupby("cell")["elev"].min()  # deepest point in the footprint
    fl = frac_land.reindex(cells).fillna(0.0).to_numpy()
    me = max_elev.reindex(cells).to_numpy()
    mn = min_elev.reindex(cells).to_numpy()
    min_ocean_depth = -me  # shallowest seabed (positive below sea level)
    max_ocean_depth = -mn  # deepest seabed (positive below sea level)
    return fl, min_ocean_depth, max_ocean_depth, mn


def water_mask_from_bathy(
    land_frac: np.ndarray, min_elev: np.ndarray
) -> np.ndarray:
    """Relaxed coastal water mask (Change 2): water iff rule (i) OR rule (ii)."""
    rule_i = land_frac <= LAND_FRAC_THRESHOLD
    rule_ii = np.isfinite(min_elev) & (min_elev <= MIN_ELEV_WATER_CUTOFF_M)
    return rule_i | rule_ii


def persist_full_domain_bathymetry(level: int) -> None:
    """Write the canonical full-domain land-mask + depth grid once (all cells).

    Independent of any tag: covers every level-`L` cell touched by the ETOPO
    subset, so `data/clean/bathymetry_healpix_L{level}.nc` is the reusable grid.
    """
    out_path = CLEAN_DIR / f"bathymetry_healpix_L{level}.nc"
    # Always (re)write: Change 2 redefines land_mask, so a file from a prior
    # strict-mask run must be regenerated, not reused. The grid is small and cheap.
    if out_path.exists():
        out_path.unlink()
    bz = xr.open_dataset(BATHY_NC)["z"]
    mlon, mlat = np.meshgrid(bz["longitude"].values, bz["latitude"].values)
    bcell = hgn.lonlat_to_healpix(
        mlon.ravel(), mlat.ravel(), level).astype("int64")
    bdf = pd.DataFrame({"cell": bcell, "elev": np.asarray(bz.values).ravel()})
    frac_land = bdf.assign(land=bdf["elev"] >= 0).groupby("cell")["land"].mean()
    max_elev = bdf.groupby("cell")["elev"].max()
    min_elev = bdf.groupby("cell")["elev"].min()
    cells = frac_land.index.to_numpy().astype("int64")
    clon, clat = hgn.healpix_to_lonlat(cells, level)
    clon = ((np.asarray(clon) + 180.0) % 360.0) - 180.0
    _persist_bathymetry(
        out_path, cells, np.asarray(clon), np.asarray(clat),
        frac_land.to_numpy(), -max_elev.to_numpy(), -min_elev.to_numpy(),
        min_elev.to_numpy(), level)


def _persist_bathymetry(
    out_path: Path, cells: np.ndarray, clon: np.ndarray, clat: np.ndarray,
    land_frac: np.ndarray, min_ocean_depth: np.ndarray,
    max_ocean_depth: np.ndarray, min_elev: np.ndarray, level: int
) -> None:
    """Save a per-cell land mask + seabed-depth grid as NetCDF.

    ``land_mask`` here is the RELAXED coastal mask (Change 2): water (1) where rule
    (i) ``land_fraction <= LAND_FRAC_THRESHOLD`` OR rule (ii) min footprint
    elevation ``<= MIN_ELEV_WATER_CUTOFF_M``. ``max_ocean_depth_m`` (deepest point)
    is the seabed the depth emission floor reads.
    """
    water = water_mask_from_bathy(land_frac, min_elev)
    ds = xr.Dataset(
        {
            "land_fraction": ("cells", land_frac),
            "min_ocean_depth_m": ("cells", min_ocean_depth),
            "max_ocean_depth_m": ("cells", max_ocean_depth),
            "min_elev_m": ("cells", min_elev),
            "land_mask": ("cells", water.astype("i1")),
        },
        coords={"cell_ids": ("cells", cells),
                "longitude": ("cells", clon), "latitude": ("cells", clat)},
    )
    ds["cell_ids"].attrs.update(
        {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"})
    ds.attrs.update({
        "source": "NOAA NCEI ETOPO 2022 v1 15 arc-second relief via CoastWatch "
                  "ERDDAP griddap ETOPO_2022_v1_15s",
        "land_frac_threshold": LAND_FRAC_THRESHOLD,
        "min_elev_water_cutoff_m": MIN_ELEV_WATER_CUTOFF_M,
        "coastal_mask_rule": "water iff land_fraction<=thr OR min_elev<=cutoff",
    })
    ds.to_netcdf(out_path)


# %% [markdown]
# ## Light (twilight) emission — per event, vectorised over cells

# %%
def event_emission(
    t_event: pd.Timestamp, kind: str, clon: np.ndarray, clat: np.ndarray
) -> np.ndarray:
    """Per-cell Gaussian over the modelled-minus-detected twilight time residual."""
    ts = pd.Timestamp(t_event)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    t = Time(ts.isoformat(), scale="utc")
    sun = get_sun(t)
    dec, ra = sun.dec.rad, sun.ra.rad
    gast = t.sidereal_time("apparent", "greenwich").rad
    h0 = np.radians(SOLAR_HORIZON_DEG)
    lat_r = np.radians(clat)
    cos_h = ((np.sin(h0) - np.sin(lat_r) * np.sin(dec))
             / (np.cos(lat_r) * np.cos(dec)))
    valid = np.abs(cos_h) <= 1.0
    cos_h = np.clip(cos_h, -1.0, 1.0)
    h_mag = np.arccos(cos_h)
    target_h = -h_mag if kind == "sunrise" else h_mag
    lha_now = gast + np.radians(clon) - ra
    dlha = (target_h - lha_now + np.pi) % (2 * np.pi) - np.pi
    resid_min = np.degrees(dlha) / 15.0 / 1.00273790935 * 60.0
    pdf = np.exp(-0.5 * (resid_min / SIGMA_T_MIN) ** 2)
    return np.where(valid, pdf, 0.0)


def daily_light_factor(
    shark: str, cells: np.ndarray, clon: np.ndarray, clat: np.ndarray,
    days: pd.DatetimeIndex
) -> tuple[np.ndarray, int]:
    """Daily light factor on `cells`: product of each day's events, renormalised."""
    tw = pd.read_csv(
        CLEAN_DIR / "tags" / shark / "twilights.csv", parse_dates=["time"])
    tw["time"] = pd.to_datetime(tw["time"]).dt.tz_localize(None)
    tw["day"] = tw["time"].dt.floor("D")
    day_index = {d: i for i, d in enumerate(days)}
    light = np.ones((len(days), cells.size))  # uniform default for empty days
    n_with = 0
    for d, grp in tw.groupby("day"):
        if d not in day_index:
            continue
        fac = np.ones(cells.size)
        for _, row in grp.iterrows():
            e = event_emission(row["time"], row["event"], clon, clat)
            s = e.sum()
            if s > 0:
                fac *= (e / s)
        fs = fac.sum()
        if fs > 0:
            light[day_index[d]] = fac / fs
            n_with += 1
    return light, int(tw["day"].nunique()), n_with


# %% [markdown]
# ## Temperature emission via the sibling pangeo-fish path
#
# Returns the temperature emission Dataset on the canonical level-`L` cell set,
# already carrying `pdf(time, cells)`, `initial`, `final`, `mask`, and a daily
# `time` axis. This is the same likelihood the 276 km temperature chain used.

# %%
def temperature_emission(shark: str, level: int) -> xr.Dataset:
    """GLORYS temperature emission on a level-`L` HEALPix grid (baseline-free grid).

    Reads this repo's own DST (``data/clean/tags/<shark>/dst.csv``) and the
    species-range GLORYS reference model
    ``data/clean/reference_model_<shark>_gulfext.nc``.
    """
    tag, tag_log, time_slice = load_tag(tag_root=TAG_ROOT, tag_name=shark)
    rm = (xr.open_dataset(CLEAN_DIR / f"reference_model_{shark}_gulfext.nc")
          .chunk({"time": 24, "lat": -1, "lon": -1, "depth": -1})
          .sel(time=time_slice))
    diff, _ = compute_diff(
        reference_model=rm, tag_log=tag_log,
        relative_depth_threshold=RELATIVE_DEPTH_THRESHOLD, chunk_time=24)
    diff = diff.compute()
    diff = (diff.assign(latitude=rm["latitude"], longitude=rm["longitude"])
            .swap_dims({"lat": "yi", "lon": "xi"})
            .drop_vars(["lat", "lon"]).compute())
    reg, _ = regrid_dataset(ds=diff, refinement_level=level, dims=["cells"])
    reg["cell_ids"].attrs.update(
        {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"})
    em, _ = compute_emission_pdf(
        diff_ds=reg, events_ds=tag["tagging_events"].ds,
        differences_std=DIFFERENCES_STD, initial_std=INITIAL_STD,
        recapture_std=RECAPTURE_STD, dims=["cells"])
    return em.compute()


# %% [markdown]
# ## Light-only state space (for 02_01, which has no temperature factor)

# %%
def light_only_grid(shark: str, level: int):
    """Build a bbox HEALPix grid + daily axis + endpoint anchors for 02_01."""
    events = pd.read_csv(
        CLEAN_DIR / "tags" / shark / "tagging_events.csv", parse_dates=["time"])
    gpe3 = pd.read_csv(CLEAN_DIR / f"gpe3_{shark}.csv", parse_dates=["time"])
    alon = np.concatenate([events["longitude"], gpe3["longitude"]])
    alat = np.concatenate([events["latitude"], gpe3["latitude"]])
    lon_min, lon_max = alon.min() - BBOX_PAD_DEG, alon.max() + BBOX_PAD_DEG
    lat_min, lat_max = alat.min() - BBOX_PAD_DEG, alat.max() + BBOX_PAD_DEG
    lons = np.arange(lon_min, lon_max + MESH_STEP_DEG, MESH_STEP_DEG)
    lats = np.arange(lat_min, lat_max + MESH_STEP_DEG, MESH_STEP_DEG)
    mlon, mlat = np.meshgrid(lons, lats)
    cells = np.unique(hgn.lonlat_to_healpix(
        mlon.ravel(), mlat.ravel(), level).astype("int64"))
    clon, clat = hgn.healpix_to_lonlat(cells, level)
    clon = ((np.asarray(clon) + 180.0) % 360.0) - 180.0
    clat = np.asarray(clat)
    # Daily axis spanning release..pop-up.
    t0 = pd.to_datetime(events.loc[events.event_name == "release", "time"].iloc[0])
    t1 = pd.to_datetime(
        events.loc[events.event_name == "fish_death", "time"].iloc[0])
    days = pd.date_range(t0.floor("D"), t1.floor("D"), freq="1D").tz_localize(None)
    grid = xr.Dataset(coords={
        "cell_ids": ("cells", cells),
        "latitude": ("cells", clat), "longitude": ("cells", clon)})
    grid["cell_ids"].attrs.update(
        {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"})
    grid = to_healpix(grid)
    release = events[events.event_name == "release"].iloc[0]
    popup = events[events.event_name == "fish_death"].iloc[0]
    initial = distrib_hp.normal_at(
        grid, pos=xr.Dataset({"longitude": float(release.longitude),
                              "latitude": float(release.latitude)}),
        sigma=INITIAL_STD)
    final = distrib_hp.normal_at(
        grid, pos=xr.Dataset({"longitude": float(popup.longitude),
                              "latitude": float(popup.latitude)}),
        sigma=RECAPTURE_STD)
    return cells, clon, clat, days, initial, final


# %% [markdown]
# ## Daily bathymetric depth EMISSION from the DST dive record (Change 1)
#
# **Why hand-rolled, not `pangeo_fish.bathy`.** `pangeo_fish.bathy` provides
# `compute_healpix_histogram_region_bin_size` (a per-cell seabed-depth survival
# curve `1 - cumsum(hist)`) and `batch_compute_pdf_bathy`, which evaluates that
# survival curve at each day's `max(pressure) - XE + 10` bin. Its semantics are
# exactly the soft floor we want — but the batch entry point is **rigidly coupled**
# to a `{target_root}/diff-regridded.zarr` reference store and an `XE` reference
# field that must share its own nside-based histogram cell set. This pipeline's
# emission cells come from the sibling temperature regrid (level-9 NESTED), and we
# materialise neither `diff-regridded.zarr` nor an `XE` field on that grid, so
# wiring the batch API would mean reconstructing those artefacts. We therefore
# build an **EQUIVALENT floor-aware soft likelihood by hand** with the same
# survival-curve semantics, stated here as a deviation.
#
# **Formulation.** For each day `t` let `D_t = max_o pressure[t, o]` be the deepest
# dive that day (the binding floor constraint — the seabed must be at least this
# deep). For a cell `c` we use its DEEPEST seabed point `S_c` (`max_ocean_depth_m`
# `= -min(elev)`): a dive of depth `D_t` fits in the cell iff SOME water in it is at
# least that deep, so the deepest point is the correct reduction. (Using the cell's
# SHALLOWEST point — an earlier bug — read land elevation as "seabed" on partly-land
# coastal cells and wrongly crushed exactly the shallow-nursery cells the relaxed
# mask admits; a 08_01 diagnostic showed the true Argos cells dropping to factor
# ~0.05 on 26/26 days. With the deepest point they recover to ~0.996.)
#
# ```
# depth_emission[t, c] = sigmoid( (S_c - D_t) / DEPTH_FLOOR_TOL_M )
# ```
#
# `sigmoid(x) = 1/(1+e^-x)`. This is ~1 where the seabed is deeper than the dives
# (`S_c >> D_t`, fully allowed — a 50 m-diving shark over 2000 m water is NOT
# penalised), 0.5 at the boundary `S_c = D_t`, and →0 only where the deepest seabed
# is still many tens of metres SHALLOWER than depths the animal demonstrably reached
# (physically impossible). It is the smooth, single-evaluation analogue of
# `bathy.py`'s survival curve read at the max-dive bin. Cells with unknown seabed
# (`S_c` NaN) get 0. We deliberately do NOT add a benthic-association upweight (the
# task's optional term): keeping it a pure floor avoids over-penalising pelagic
# days, as requested. Days with no DST coverage default to uniform (factor 1).

# %%
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def daily_depth_emission(
    shark: str, cells: np.ndarray, days: pd.DatetimeIndex,
    seabed_depth: np.ndarray, land_mask: np.ndarray
) -> np.ndarray:
    """Daily soft, floor-aware depth emission (Change 1); uniform if no DST.

    ``seabed_depth`` is the cell's DEEPEST point (``max_ocean_depth_m``), positive
    metres below sea level — see the depth-emission markdown for why deepest, not
    shallowest.
    """
    emis = np.ones((len(days), cells.size))
    dst_path = CLEAN_DIR / "tags" / shark / "dst.csv"
    if not dst_path.exists():
        return emis  # no DST -> no depth constraint (uniform)
    dst = pd.read_csv(dst_path, parse_dates=["time"])
    dst["day"] = pd.to_datetime(dst["time"]).dt.floor("D").dt.tz_localize(None)
    dmax = dst.groupby("day")["pressure"].max()
    finite = np.isfinite(seabed_depth)
    for i, d in enumerate(days):
        if d not in dmax.index:
            continue
        d_t = float(dmax.loc[d])
        fac = np.zeros(cells.size)
        fac[finite] = _sigmoid((seabed_depth[finite] - d_t) / DEPTH_FLOOR_TOL_M)
        # Guard: if the soft factor would zero out every water cell for this day
        # (would collapse the day), fall back to uniform so light×temp still decide.
        if (fac * (land_mask > 0)).sum() <= 0:
            continue
        emis[i] = fac
    return emis


# %% [markdown]
# ## HMM fit (or reuse cached σ) + MAP (mode) decode on a fused daily emission
#
# The expensive step is the `EagerBoundsSearch` σ-fit. When a previous run already
# fitted σ for this tag (cached in `results/rec_<tag>.json` under `sigma_fused`),
# we **reuse** it and only re-decode — far cheaper. The decode uses the per-day
# **mode** (MAP cell), which selects an allowed ocean cell and (unlike the mean)
# never averages onto land — see the note above on why Viterbi is unavailable on
# the HEALPix grid. The per-day posterior from `predict_proba` is cached to
# `results/posterior_<tag>.nc`.

# %%
def _build_em_hp(
    cells: np.ndarray, clon: np.ndarray, clat: np.ndarray,
    times: np.ndarray, fused: np.ndarray, initial, final,
    land_mask: np.ndarray, level: int
):
    """Assemble + normalise the HEALPix emission Dataset; return (em_hp, max_σ)."""
    em = xr.Dataset(
        {"pdf": (("time", "cells"), fused)},
        coords={"time": times, "cell_ids": ("cells", cells),
                "latitude": ("cells", clat), "longitude": ("cells", clon)})
    em["cell_ids"].attrs.update(
        {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"})
    em = to_healpix(em)
    em = em.assign(
        initial=initial.fillna(0.0) if hasattr(initial, "fillna") else initial,
        final=final.fillna(0.0) if hasattr(final, "fillna") else final,
        mask=("cells", (land_mask > 0)))
    normd, _ = normalize_pdf(ds=em, chunks={"time": 24})
    normd = normd.compute()
    normd["cell_ids"].attrs.update(
        {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"})
    em_hp = to_healpix(normd)

    tv = em_hp["time"].values
    dt_h = float(np.median(np.diff(tv)) / np.timedelta64(1, "h"))
    max_sigma = (MAX_SPEED.to("km/h").magnitude * dt_h * ADJUSTMENT_FACTOR
                 / EARTH_RADIUS.to("km").magnitude)
    return em_hp, max_sigma


def _estimator(em_hp, sigma, blocked_keys: np.ndarray, shark: str = "default"):
    """Build an EagerEstimator with the LAND-BARRIER predictor factory.

    The movement kernel forbids transitions whose great-circle segment crosses land
    (``LandBarrierGaussian1DHealpix``). ``blocked_keys`` (the σ-independent
    precomputed forbidden-pair set) is reused across every predictor build the
    EagerBoundsSearch σ-fit constructs.
    """
    pf = curry(
        LandBarrierGaussian1DHealpix, cell_ids=em_hp["cell_ids"].data,
        grid_info=em_hp.dggs.grid_info, truncate=TRUNCATE,
        weights_threshold=1e-8,
        pad_kwargs={"mode": "constant", "constant_value": 0},
        optimize_convolution=True,
        blocked_keys=blocked_keys, barrier_tag=shark)
    return EagerEstimator(sigma=sigma, predictor_factory=pf)


def _traj_to_df(traj, traj_id: str) -> pd.DataFrame:
    """Extract one named trajectory as a daily lon/lat DataFrame."""
    t = next(t for t in traj.trajectories if t.id == traj_id)
    df = t.df.copy()
    return pd.DataFrame({
        "time": pd.to_datetime(df.index).tz_localize(None),
        "longitude": df.geometry.x.values, "latitude": df.geometry.y.values,
    }).set_index("time").resample("1D").mean().dropna().reset_index()


def fit_and_decode(
    cells: np.ndarray, clon: np.ndarray, clat: np.ndarray,
    times: np.ndarray, fused: np.ndarray, initial, final,
    land_mask: np.ndarray, level: int, shark: str,
    cached_sigma: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, int]:
    """Fit (or reuse) σ, run predict_proba + mode/mean decode.

    Returns (mode_track, mean_track, σ, σ_max, n_barrier_fallback_rows). The per-day
    posterior is written to ``results/posterior_<shark>.nc``.
    """
    em_hp, max_sigma = _build_em_hp(
        cells, clon, clat, times, fused, initial, final, land_mask, level)

    # Precompute the σ-independent land-crossing pair set ONCE for this grid; the
    # bounds-search reuses it across every predictor build (see the barrier section).
    grid_cells = np.asarray(em_hp["cell_ids"].data).astype("int64")
    gclon, gclat = hgn.healpix_to_lonlat(grid_cells, level)
    gclon = ((np.asarray(gclon) + 180.0) % 360.0) - 180.0
    gclat = np.asarray(gclat)
    _t0 = pd.Timestamp.utcnow()
    blocked_keys = precompute_blocked_pairs(
        grid_cells, gclon, gclat, em_hp.dggs.grid_info, level)
    _dt = (pd.Timestamp.utcnow() - _t0).total_seconds()
    print(f"    [barrier] precomputed {blocked_keys.size} blocked land-crossing "
          f"pairs (≤{BARRIER_MAX_KM:.0f} km) in {_dt:.1f}s", flush=True)

    if cached_sigma is not None and np.isfinite(cached_sigma):
        sigma = float(cached_sigma)
        optimized = _estimator(em_hp, sigma, blocked_keys, shark)
        print(f"    reusing cached sigma={sigma:.5g} (skipping bounds search)",
              flush=True)
    else:
        estimator = _estimator(em_hp, None, blocked_keys, shark)
        optimizer = EagerBoundsSearch(
            estimator, (1e-4, max_sigma), optimizer_kwargs={"xtol": TOLERANCE})
        optimized = optimizer.fit(em_hp)
        sigma = float(optimized.sigma)
        print(f"    fitted sigma={sigma:.5g} (bounds search)", flush=True)

    # Per-day posterior (forward-backward), cached to NetCDF (never .npz). This
    # builds the barrier predictor at the fitted σ, so the fallback-row counter
    # below reflects the final kernel actually used for the posterior.
    states = optimized.predict_proba(em_hp)  # DataArray "states" (time, cells)
    n_barrier_fallback = int(_BARRIER_FALLBACK_ROWS.get(shark, 0))
    post = states.to_dataset(name="posterior")
    post["cell_ids"].attrs.update(
        {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"})
    post.attrs.update({"tag": shark, "sigma": sigma, "decode": "mode"})
    post.to_netcdf(RESULTS_DIR / f"posterior_{shark}.nc")

    states_ds = post.rename({"posterior": "states"})
    traj = optimized.decode(
        em_hp, states_ds.fillna(0), mode=["mode", "mean"], progress=False)
    mode_track = _traj_to_df(traj, "mode")
    mean = _traj_to_df(traj, "mean")
    return mode_track, mean, sigma, max_sigma, n_barrier_fallback


# %% [markdown]
# ## Fuse the factors for one tag, run the HMM
#
# Returns the fused daily track plus diagnostics (σ, σ_max, level, twilight count,
# land-mask cell count, before/after on-land point counts).

# %% [markdown]
# ## On-land check by sampling ETOPO elevation at each track point
#
# Independent of the HEALPix mask: bilinear-nearest lookup into the raw ETOPO
# relief (`z`), `z >= 0` = land. This is the honest land test for the Viterbi
# track and the safety-net snap target.

# %%
def _etopo_elev_at(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Nearest-pixel ETOPO elevation (m) at each (lon, lat)."""
    bz = xr.open_dataset(BATHY_NC)["z"]
    return bz.sel(longitude=xr.DataArray(np.asarray(lons), dims="p"),
                  latitude=xr.DataArray(np.asarray(lats), dims="p"),
                  method="nearest").values


def n_land_by_etopo(track: pd.DataFrame) -> int:
    """Count track daily points whose ETOPO elevation is >= 0 (on land)."""
    if track.empty:
        return 0
    elev = _etopo_elev_at(track["longitude"].values, track["latitude"].values)
    return int(np.nansum(elev >= 0))


def snap_to_ocean(
    track: pd.DataFrame, cells: np.ndarray, clon: np.ndarray, clat: np.ndarray,
    land_mask: np.ndarray
) -> tuple[pd.DataFrame, int]:
    """Snap any on-land (ETOPO z>=0) point to the nearest allowed ocean cell."""
    elev = _etopo_elev_at(track["longitude"].values, track["latitude"].values)
    bad = np.where(elev >= 0)[0]
    if bad.size == 0:
        return track, 0
    ocean = land_mask > 0
    olon, olat = clon[ocean], clat[ocean]
    t = track.copy()
    for i in bad:
        d = gc_km(t.loc[i, "longitude"], t.loc[i, "latitude"], olon, olat)
        j = int(np.argmin(d))
        t.loc[i, "longitude"] = olon[j]
        t.loc[i, "latitude"] = olat[j]
    return t, int(bad.size)


# %%
def run_fused_hmm(shark: str, level: int, cached_sigma: float | None = None) -> dict:
    has_temp = shark in HAS_TEMPERATURE

    if has_temp:
        temp_em = temperature_emission(shark, level)
        cells = temp_em["cell_ids"].values.astype("int64")
        clon, clat = hgn.healpix_to_lonlat(cells, level)
        clon = ((np.asarray(clon) + 180.0) % 360.0) - 180.0
        clat = np.asarray(clat)
        times = temp_em["time"].values
        days = pd.to_datetime(times).floor("D")
        # compute_emission_pdf leaves NaN on GLORYS-masked cells; treat those as
        # zero likelihood so the fused product is finite.
        temp_pdf = np.nan_to_num(
            temp_em["pdf"].transpose("time", "cells").values, nan=0.0)
        # GLORYS coverage mask (cells with any ocean field): combined with the
        # ETOPO land mask below so the HMM mask is the intersection of "ocean per
        # GLORYS" and "ocean per ETOPO".
        glorys_mask = np.nan_to_num(temp_em["mask"].values, nan=0.0) > 0
        initial, final = temp_em["initial"], temp_em["final"]
    else:
        cells, clon, clat, days, initial, final = light_only_grid(shark, level)
        times = days.to_numpy()
        temp_pdf = np.ones((len(days), cells.size))  # uniform: no SST factor
        glorys_mask = np.ones(cells.size, dtype=bool)

    # Bathymetry on these cells (Change 2: relaxed coastal water mask).
    land_frac, min_ocean_depth, max_ocean_depth, min_elev = bathymetry_per_cell(
        cells, level)
    _persist_bathymetry(
        CLEAN_DIR / f"bathymetry_healpix_L{level}_{shark}.nc",
        cells, clon, clat, land_frac, min_ocean_depth, max_ocean_depth,
        min_elev, level)
    water_relaxed = water_mask_from_bathy(land_frac, min_elev)
    land_mask = np.where(water_relaxed & glorys_mask, 1.0, 0.0)

    # Confirm the relaxed mask opens the Vizcaíno nursery cell (diagnostic only).
    _viz_idx = int(np.argmin(gc_km(VIZCAINO_LON, VIZCAINO_LAT, clon, clat)))
    viz_is_water = bool(land_mask[_viz_idx] > 0)
    viz_dist_km = float(gc_km(
        VIZCAINO_LON, VIZCAINO_LAT, clon[_viz_idx], clat[_viz_idx]))

    # Light factor (daily).
    light_pdf, n_tw_days, n_light_days = daily_light_factor(
        shark, cells, clon, clat, days)

    # Depth EMISSION (daily, soft floor-aware — Change 1); uniform if no DST.
    # Uses the cell's DEEPEST seabed point (max_ocean_depth), not the shallowest.
    depth_emission = daily_depth_emission(
        shark, cells, days, max_ocean_depth, land_mask)

    # Fuse and normalise per day, recovering any day that collapses to all-zero.
    fused = temp_pdf * light_pdf * land_mask[None, :] * depth_emission
    norm = fused.sum(axis=1, keepdims=True)
    collapsed = (norm[:, 0] == 0)
    for i in np.where(collapsed)[0]:
        fallback = temp_pdf[i] * land_mask
        s = fallback.sum()
        fused[i] = fallback / s if s > 0 else light_pdf[i]
        norm[i, 0] = fused[i].sum()
    fused = np.where(norm > 0, fused / norm, 0.0)

    mode_track, mean_track, sigma, max_sigma, n_barrier_fallback = fit_and_decode(
        cells, clon, clat, times, fused, initial, final, land_mask, level,
        shark, cached_sigma=cached_sigma)

    # Keep the old posterior-mean track for reference; the headline track is now
    # the per-day MAP (mode) path.
    mean_track.to_csv(
        RESULTS_DIR / f"fused_track_mean_{shark}.csv", index=False)

    # On-land check via ETOPO elevation (z >= 0 = land).
    n_land_mean = n_land_by_etopo(mean_track)
    n_land_mode_raw = n_land_by_etopo(mode_track)
    mode_track, n_snapped = snap_to_ocean(
        mode_track, cells, clon, clat, land_mask)
    n_land_mode = n_land_by_etopo(mode_track)
    # The per-day MODE track is now the *reference* track; the headline
    # fused_track_<tag>.csv is rebuilt below by the land-aware Viterbi path.
    mode_track.to_csv(RESULTS_DIR / f"fused_track_mode_{shark}.csv", index=False)

    # BEFORE reference: the prior light-only track (all-true mask), ETOPO-tested.
    before_path = RESULTS_DIR / f"light_track_{shark}.csv"
    n_land_before = (n_land_by_etopo(pd.read_csv(before_path))
                     if before_path.exists() else None)

    return {
        "track": mode_track, "sigma": sigma, "max_sigma": max_sigma,
        "level": level, "n_twilight_days": n_tw_days,
        "n_light_days": n_light_days, "has_temperature": has_temp,
        "n_land_mask_cells": int((land_mask == 0).sum()),
        "n_collapsed_days": int(collapsed.sum()),
        "n_land_before": n_land_before, "n_land_after": n_land_mode,
        "n_land_mean": n_land_mean,
        "n_land_mode_raw": n_land_mode_raw, "n_snapped": n_snapped,
        "n_barrier_fallback_rows": n_barrier_fallback,
        "viz_nursery_is_water": viz_is_water,
        "viz_nursery_cell_dist_km": viz_dist_km,
        "depth_emission_active": bool(
            (CLEAN_DIR / "tags" / shark / "dst.csv").exists()),
    }


# %% [markdown]
# ## Land-aware most-probable PATH (custom constrained Viterbi)
#
# The per-day **mode** decode above treats each day independently, so the headline
# track can JUMP across the Baja peninsula between consecutive days (a physically
# impossible land-crossing seen at ~29°N). pangeo-fish 2026.4.0's built-in Viterbi
# only runs on the legacy flat `(x, y)` grid, not the 1-D HEALPix `cells` grid we
# use, so we build the **jointly** most-probable, land-respecting path ourselves
# directly from the cached `results/posterior_<tag>.nc` (no HMM re-fit, no GLORYS
# re-read).
#
# **Node score** at day `t`, cell `i`: `log(posterior[t, i] + EPS)`. For
# tractability each day's candidate set is restricted to the **top-K** cells by
# posterior (`VITERBI_K`).
#
# **Transition** (day `t-1` cell `j` → day `t` cell `i`) is allowed iff BOTH:
#   (a) great-circle distance ≤ `MAX_DAILY_KM`, and
#   (b) the great-circle segment `j→i` does not cross land. The land test samples
#       interior points at a **resolution-aware** density (`_adaptive_n_samples`):
#       at least one sample per ETOPO pixel the segment traverses (sample step ≤
#       the ~0.4 km ETOPO pixel), so a sub-km island cannot hide between samples.
#       ETOPO `z ≥ 0` = land.
# A forbidden transition has potential `-inf`.
#
# **MAX_DAILY_KM justification.** The σ-fit prior caps sustained speed at
# `MAX_SPEED = 5 km/h` (≈120 km/day for a juvenile white shark). A hard 120 km/day
# cap leaves no admissible water path around the Cabo San Lázaro / Baja cape on
# some days (the detour around the cape exceeds the straight-line daily step), so
# we allow headroom at **200 km/day**. This is loose enough to round the cape but
# still far below the width of the Baja isthmus, so it cannot license a straight
# land-crossing — condition (b) is the real guard and is verified to give zero
# land-crossing segments below. If a tag still has no admissible path we relax
# `MAX_DAILY_KM` stepwise, then `VITERBI_K`, recording every relaxation honestly.
#
# **Endpoint anchoring (soft, log-prior).** Rather than *forcing* days 0 and N to a
# single cell — which is brittle: if that exact cell is unreachable from the
# neighbouring day's top-K through water the whole path fails — we **add a Gaussian
# log-prior** to those two days' node scores, centred on the water cell nearest the
# release / pop-up position (from `data/clean/tags/<tag>/tagging_events.csv`), with
# spread `ANCHOR_SIGMA_KM`. The anchor cell is also injected into those days'
# candidate sets so it is always available. This pulls the path to the endpoints
# without making a single unreachable cell fatal. Argos is never used.

# %%
EPS = 1e-300  # floor inside log(posterior + EPS) so zero-posterior cells -> -inf-ish
VITERBI_K = 600  # candidate cells per day (top-K by posterior)
VITERBI_K_RELAX = [600, 1200, 2400]  # K relaxation ladder if no path exists
MAX_DAILY_KM = 200.0  # see justification above (~5 km/h sustained + headroom)
MAX_DAILY_KM_RELAX = [200.0, 300.0, 450.0, 600.0]  # km/day relaxation ladder
ANCHOR_SIGMA_KM = 30.0  # Gaussian spread of the release/pop-up endpoint log-prior

# --- Adaptive land-crossing sampler ---
# A fixed sample count (the old N_SEGMENT_SAMPLES=21) is resolution-blind: on a
# ~58 km step it spaces samples ~3 km apart, so a sub-km island (e.g. a Channel
# Island grazed by tag 08_09 seg 19) hides BETWEEN samples and the step is wrongly
# allowed. ETOPO 2022 15-arc-second pixels are ~0.46 km on the latitude axis and
# ~0.39 km on the longitude axis at this latitude band. We sample at least one
# point per ETOPO pixel the segment traverses: n_samples = max(20, ceil(seg_km /
# ETOPO_PIXEL_KM) + 1), so the sample step is <= the pixel size and no island can
# fall through. Use the SMALLER pixel dimension (longitude) as the nominal pixel
# size, which is conservative (denser sampling). Cap at MAX_SEGMENT_SAMPLES so a
# pathological long step can't blow up memory (06_10's longest legitimate steps
# are ~200 km => ~1000 samples here, well under the cap).
#
# FINER-CHECK FIX (gulfext referee re-run): the Gulf-of-California coastline and
# its islands are finer than the 0.39 km ETOPO nominal pixel resolved on some long
# segments — an independent dense check found 1 residual land-crossing in the
# 07_05 gulfext path. We tighten the segment-sample spacing to <= LAND_CHECK_KM
# (0.2 km), well below the ETOPO pixel size, so BOTH the land-barrier kernel's
# segment test AND the path-DP's segment test (both routed through
# `_segment_crosses_land`) sample densely enough that no island graze can slip
# through. The independent dense verification below uses the same 0.2 km spacing.
LAND_CHECK_KM = 0.2    # target sample spacing for ALL land-crossing tests
ETOPO_PIXEL_KM = 0.39  # smaller (longitude) ETOPO pixel dimension in this domain
MIN_SEGMENT_SAMPLES = 20
MAX_SEGMENT_SAMPLES = 12000


def _adaptive_n_samples(seg_km: float) -> int:
    """Samples for one segment: spacing <= LAND_CHECK_KM (0.2 km), capped defensively."""
    n = int(np.ceil(seg_km / LAND_CHECK_KM)) + 1
    return int(min(MAX_SEGMENT_SAMPLES, max(MIN_SEGMENT_SAMPLES, n)))


def _dense_n_samples(seg_km: float) -> int:
    """Independent verification sampler: spacing <= LAND_CHECK_KM (0.2 km)."""
    n = int(np.ceil(seg_km / LAND_CHECK_KM)) + 1
    return int(min(MAX_SEGMENT_SAMPLES, max(200, n)))


def _etopo_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon_axis, lat_axis, z[lat, lon]) from the ETOPO subset, cached."""
    bz = xr.open_dataset(BATHY_NC)
    return (bz["longitude"].values, bz["latitude"].values,
            np.asarray(bz["z"].values))


_ETOPO_LON, _ETOPO_LAT, _ETOPO_Z = _etopo_axes()


def _etopo_z_vec(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Vectorised nearest-pixel ETOPO elevation via searchsorted on both axes."""
    ix = np.clip(np.searchsorted(_ETOPO_LON, lons), 0, _ETOPO_LON.size - 1)
    iy = np.clip(np.searchsorted(_ETOPO_LAT, lats), 0, _ETOPO_LAT.size - 1)
    return _ETOPO_Z[iy, ix]


def _segment_crosses_land(
    lon0: np.ndarray, lat0: np.ndarray, lon1: np.ndarray, lat1: np.ndarray,
) -> np.ndarray:
    """For each segment (j->i) test whether any INTERIOR sample is on land.

    The sample DENSITY is resolution-aware (see `_adaptive_n_samples`): every batch
    of segments is sampled at the count required by the LONGEST segment in the batch,
    so the per-segment sample step is <= one ETOPO pixel and no island can hide
    between samples. (Using the batch max is conservative — shorter segments in the
    batch get even finer sampling.) Linear interpolation in lon/lat is an adequate
    proxy for the great-circle segment at these short (<= a few hundred km) daily
    steps. Returns a boolean array, True where the segment crosses land (forbidden).
    """
    lon0 = np.asarray(lon0, dtype="float64")
    lat0 = np.asarray(lat0, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat1 = np.asarray(lat1, dtype="float64")
    if lon0.size == 0:
        return np.zeros(0, dtype=bool)
    seg_km = gc_km(lon0, lat0, lon1, lat1)
    out = np.zeros(lon0.size, dtype=bool)
    # Each segment gets its OWN resolution-aware sample count (spacing <=
    # LAND_CHECK_KM). To stay vectorised without sampling every short segment at the
    # global-max count (wasteful at 0.2 km), group segments by their required count
    # via a small set of log-spaced length buckets and sample each bucket once. The
    # sample STEP for every segment is still <= LAND_CHECK_KM (a segment is placed in
    # a bucket whose count is >= its own required count), so no island can hide.
    req = np.array([_adaptive_n_samples(float(s)) for s in seg_km])
    # Round each requirement UP to the next power-of-two bucket to bound the number
    # of distinct sampler passes (<= ~14 buckets between 20 and 12000).
    bucket = np.where(req <= MIN_SEGMENT_SAMPLES, MIN_SEGMENT_SAMPLES,
                      1 << np.ceil(np.log2(np.maximum(req, 1))).astype(int))
    bucket = np.minimum(bucket, MAX_SEGMENT_SAMPLES)
    for n_samples in np.unique(bucket):
        sel = np.where(bucket == n_samples)[0]
        fracs = np.linspace(0.0, 1.0, int(n_samples))[1:-1]  # interior points
        slon = lon0[sel, None] + (lon1[sel] - lon0[sel])[:, None] * fracs[None, :]
        slat = lat0[sel, None] + (lat1[sel] - lat0[sel])[:, None] * fracs[None, :]
        z = _etopo_z_vec(slon.ravel(), slat.ravel()).reshape(slon.shape)
        out[sel] = np.any(z >= 0.0, axis=1)
    return out


def _nearest_water_cell(
    lon0: float, lat0: float, clon: np.ndarray, clat: np.ndarray,
    water: np.ndarray,
) -> int:
    """Index of the water cell nearest (lon0, lat0); falls back to all cells."""
    idx = np.where(water)[0]
    if idx.size == 0:
        idx = np.arange(clon.size)
    d = gc_km(lon0, lat0, clon[idx], clat[idx])
    return int(idx[int(np.argmin(d))])


# %% [markdown]
# ## LAND-BARRIER movement model — the methodological centerpiece
#
# `Gaussian1DHealpix` builds a pure distance-based Gaussian transition kernel that
# is **blind to land**: a single HMM step can move probability straight ACROSS a
# peninsula (e.g. ~150 km across Baja at constant 28°N) instead of being forced to
# round the cape. The land mask only zeroes the *emission* on land cells; the
# *movement* model still teleports between water cells on opposite sides of land.
#
# `LandBarrierGaussian1DHealpix` subclasses the predictor and, after the parent
# builds the sparse Gaussian kernel, **zeroes every kernel entry (i→j) whose
# great-circle segment crosses land**, then **renormalizes each row** so the
# surviving outgoing weights still sum to 1 (probability conservation). The
# diagonal/self entry (i→i) is never blocked.
#
# ### Efficiency: the blocked-pair set is precomputed ONCE per grid
#
# A pair's blocked status is **σ-independent pure geometry**, and the
# `EagerBoundsSearch` σ-fit rebuilds the predictor many times (`scipy.fminbound`
# over σ ∈ [1e-4, σ_max]), with large-σ kernels reaching hundreds of millions of
# non-zeros. Re-running the ETOPO land test per σ is intractable. Instead,
# `precompute_blocked_pairs` runs the **same adaptive ETOPO 15-arc-second segment
# sampler** (`_segment_crosses_land`) ONCE over every cell pair within
# `BARRIER_MAX_KM` (memory-safe chunks), returning a sorted array of packed
# `(min_cid, max_cid)` keys. Each predictor build then only does a vectorised
# `searchsorted` membership test on its kernel's off-diagonal pairs — O(nnz) and
# cheap. Pairs beyond `BARRIER_MAX_KM` (≈ a physically impossible one-day move) are
# left at their tiny plain Gaussian weight: at the fitted σ their weight is already
# below `weights_threshold`, and they cannot license a peninsula crossing because
# the Baja isthmus is far narrower than `BARRIER_MAX_KM`.

# %%
# Max great-circle distance (km) for which transitions are land-checked. ~5 km/h
# sustained ⇒ ~120 km/day; 200 km gives headroom to round a cape while staying far
# below the ~150 km Baja crossing the barrier must forbid (the land test, not this
# cap, is the real guard for pairs within it).
BARRIER_MAX_KM = 200.0
_PAIR_KEY_SHIFT = 25  # cell ids at L9 < 12*4^9 ≈ 1.26e7 < 2^25 — packs (min,max) in int64

# Per-tag counter of rows that fell back to self-only (all neighbours blocked).
_BARRIER_FALLBACK_ROWS: dict[str, int] = {}

# Per-grid count of precomputed blocked land-crossing pairs (keyed by call id) —
# populated by precompute_blocked_pairs; summarised in aggregate.json.
_BLOCKED_PAIR_CACHE: dict[str, int] = {}


def _pack_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pack undirected cell-id pairs into sorted int64 keys (min<<shift | max)."""
    lo = np.minimum(a, b).astype("int64")
    hi = np.maximum(a, b).astype("int64")
    return (lo << _PAIR_KEY_SHIFT) | hi


def precompute_blocked_pairs(
    cells: np.ndarray, clon: np.ndarray, clat: np.ndarray,
    grid_info, level: int, chunk: int = 200_000,
) -> np.ndarray:
    """Sorted packed keys of all within-`BARRIER_MAX_KM` pairs that cross land.

    σ-independent; computed ONCE per tag grid and reused across every predictor
    build. Uses the kernel-ring neighbour query to enumerate candidate pairs, the
    great-circle cap to drop impossibly-long moves, and the adaptive ETOPO sampler
    (chunked) for the land test.
    """
    from healpix_convolution.neighbours import neighbours

    n = cells.size
    ring = int(np.ceil(BARRIER_MAX_KM / 6.4)) + 1  # ~6.4 km L9 cells
    nb = neighbours(cells, grid_info=grid_info, ring=ring)
    idx = np.searchsorted(cells, nb)
    valid = nb != -1
    src = np.broadcast_to(np.arange(n)[:, None], nb.shape)
    s = src[valid]
    ti = np.clip(idx[valid], 0, n - 1)
    ok = cells[ti] == nb[valid]  # drop neighbours outside this grid's cell set
    s, ti = s[ok], ti[ok]
    keep = s < ti  # undirected unique pairs only
    s, ti = s[keep], ti[keep]
    d = gc_km(clon[s], clat[s], clon[ti], clat[ti])
    within = d <= BARRIER_MAX_KM
    s, ti = s[within], ti[within]

    blocked = np.zeros(s.size, dtype=bool)
    for i in range(0, s.size, chunk):
        sl, tl = s[i:i + chunk], ti[i:i + chunk]
        blocked[i:i + chunk] = _segment_crosses_land(
            clon[sl], clat[sl], clon[tl], clat[tl])
    keys = _pack_pairs(cells[s[blocked]], cells[ti[blocked]])
    keys = np.unique(keys)  # sorted, for searchsorted membership
    _BLOCKED_PAIR_CACHE[f"grid_{len(_BLOCKED_PAIR_CACHE)}_{n}cells"] = int(keys.size)
    return keys


@dataclass
class LandBarrierGaussian1DHealpix(Gaussian1DHealpix):
    """Gaussian HEALPix movement kernel with land-crossing transitions forbidden.

    Identical to the parent except that, after the gaussian kernel is built, every
    transition whose cell pair is in ``blocked_keys`` (a precomputed sorted array of
    packed land-crossing pairs) is zeroed and each row renormalised to sum to 1. The
    diagonal (self) entry is never blocked.
    """

    blocked_keys: _ArrayLike = field(default_factory=lambda: np.empty(0, "int64"))
    barrier_tag: str = "default"

    def __post_init__(self):
        super().__post_init__()
        import sparse
        import opt_einsum

        k = self.kernel  # sparse.COO (n_out, n_in); row-normalised (axis=1)
        rows, cols = k.coords  # row -> index into cell_ids; col -> new_cell_ids
        data = k.data.copy()

        out_cid = np.asarray(self.cell_ids)[rows].astype("int64")
        in_cid = np.asarray(self.new_cell_ids)[cols].astype("int64")
        is_diag = out_cid == in_cid

        # Vectorised membership: a pair is blocked iff its packed key is present in
        # the precomputed sorted blocked_keys (searchsorted). Diagonal never blocked.
        bk = np.asarray(self.blocked_keys)
        crosses = np.zeros(rows.size, dtype=bool)
        if bk.size:
            pk = _pack_pairs(out_cid, in_cid)
            pos = np.searchsorted(bk, pk)
            pos = np.clip(pos, 0, bk.size - 1)
            crosses = (bk[pos] == pk) & ~is_diag
        data = np.where(crosses, 0.0, data)

        # Renormalise each row (output cell) over surviving inputs. A row that loses
        # all mass (every neighbour across land — rare) is reset to self-only so the
        # cell stays put rather than vanishing.
        n_out = k.shape[0]
        row_sum = np.zeros(n_out)
        np.add.at(row_sum, rows, data)
        zero_rows = np.where(row_sum <= 0.0)[0]
        n_fallback = 0
        if zero_rows.size:
            zr = set(zero_rows.tolist())
            diag_sel = is_diag & np.array([r in zr for r in rows])
            data[diag_sel] = 1.0
            row_sum[rows[diag_sel]] = 1.0
            n_fallback = int(np.unique(rows[diag_sel]).size)
            # Any zero row without a diagonal entry (should not happen) → leave as 1.
            row_sum[row_sum <= 0.0] = 1.0
        data = data / row_sum[rows]
        _BARRIER_FALLBACK_ROWS[self.barrier_tag] = n_fallback

        self.kernel = sparse.COO(
            coords=np.stack([rows, cols], axis=0), data=data,
            shape=k.shape, fill_value=0)

        if self.optimize_convolution:
            self.convolve = opt_einsum.contract_expression(
                "...a,ba->...b", self.padder.cell_ids.shape, self.kernel,
                constants=[1])
        else:
            from healpix_convolution.convolution import convolve as _conv
            self.convolve = curry(_conv, kernel=self.kernel)


def _candidate_sets(
    post: np.ndarray, k: int, anchor0: int, anchorN: int, water: np.ndarray,
) -> list[np.ndarray]:
    """Top-K WATER cell indices per day; the anchor cell injected into endpoints.

    Candidates are restricted to cells whose centre is ocean per ETOPO (``water``),
    so every path vertex is guaranteed off land (the cell-centre ETOPO test below
    must then return zero on-land vertices). Anchors are water cells by
    construction.
    """
    n_days = post.shape[0]
    water_idx = np.where(water)[0]
    cands: list[np.ndarray] = []
    for t in range(n_days):
        row = post[t]
        masked = np.where(water, row, -np.inf)
        kk = min(k, water_idx.size)
        top = np.argpartition(masked, -kk)[-kk:]
        top = top[np.isfinite(masked[top])]  # drop any non-water filler
        cands.append(top.astype("int64"))
    cands[0] = np.unique(np.append(cands[0], anchor0)).astype("int64")
    cands[-1] = np.unique(np.append(cands[-1], anchorN)).astype("int64")
    return cands


def _anchor_logprior(
    cells: np.ndarray, clon: np.ndarray, clat: np.ndarray, anchor: int,
) -> np.ndarray:
    """Gaussian log-prior over `cells` centred on the anchor cell's position."""
    d = gc_km(clon[anchor], clat[anchor], clon[cells], clat[cells])
    return -0.5 * (d / ANCHOR_SIGMA_KM) ** 2


def _viterbi_path(
    post: np.ndarray, clon: np.ndarray, clat: np.ndarray,
    cands: list[np.ndarray], max_daily_km: float,
    anchor0: int, anchorN: int,
) -> tuple[np.ndarray | None, int]:
    """Constrained Viterbi forward+backtrack over per-day candidate sets.

    Node score = log(posterior + EPS), plus a Gaussian anchor log-prior on the two
    endpoint days. Returns (path_cell_indices over days, fail_day). path is None if
    no admissible path exists, with fail_day the first unreachable day.
    """
    n_days = len(cands)
    log_node = [np.log(post[t, cands[t]] + EPS) for t in range(n_days)]
    log_node[0] = log_node[0] + _anchor_logprior(cands[0], clon, clat, anchor0)
    log_node[-1] = log_node[-1] + _anchor_logprior(cands[-1], clon, clat, anchorN)
    score = log_node[0].astype("float64")  # cumulative best score into day-0 cands
    back: list[np.ndarray] = []
    for t in range(1, n_days):
        prev_c, cur_c = cands[t - 1], cands[t]
        plon, plat = clon[prev_c], clat[prev_c]
        clon_t, clat_t = clon[cur_c], clat[cur_c]
        new_score = np.full(cur_c.size, -np.inf)
        bp = np.full(cur_c.size, -1, dtype="int64")
        for ii in range(cur_c.size):
            d = gc_km(clon_t[ii], clat_t[ii], plon, plat)
            ok = (d <= max_daily_km) & np.isfinite(score)
            if not ok.any():
                continue
            cand_prev = np.where(ok)[0]
            crosses = _segment_crosses_land(
                plon[cand_prev], plat[cand_prev],
                np.full(cand_prev.size, clon_t[ii]),
                np.full(cand_prev.size, clat_t[ii]))
            cand_prev = cand_prev[~crosses]
            if cand_prev.size == 0:
                continue
            trans = score[cand_prev]
            jbest = int(np.argmax(trans))
            new_score[ii] = trans[jbest] + log_node[t][ii]
            bp[ii] = cand_prev[jbest]
        if not np.isfinite(new_score).any():
            return None, t
        score, b = new_score, bp
        back.append(b)
    # Backtrack from the best final cell.
    end = int(np.argmax(score))
    path_local = [end]
    for t in range(n_days - 1, 0, -1):
        end = int(back[t - 1][end])
        path_local.append(end)
    path_local.reverse()
    return np.array([cands[t][path_local[t]] for t in range(n_days)]), -1


def land_aware_path(shark: str) -> dict:
    """Build the land-aware most-probable path from the cached posterior.

    Writes ``results/fused_track_<shark>.csv`` (the new headline track) and reports
    diagnostics, including any MAX_DAILY_KM / K relaxation used and the on-land /
    land-crossing verification counts.
    """
    post_ds = xr.open_dataset(RESULTS_DIR / f"posterior_{shark}.nc")
    post = np.nan_to_num(
        post_ds["posterior"].transpose("time", "cells").values, nan=0.0)
    clon = np.asarray(post_ds["longitude"].values)
    clat = np.asarray(post_ds["latitude"].values)
    times = pd.to_datetime(post_ds["time"].values)

    # Water cells per ETOPO (z < 0) at each cell centre — used for anchoring.
    water = _etopo_z_vec(clon, clat) < 0.0

    events = pd.read_csv(CLEAN_DIR / "tags" / shark / "tagging_events.csv")
    rel = events[events.event_name == "release"].iloc[0]
    pop = events[events.event_name == "fish_death"].iloc[0]
    anchor0 = _nearest_water_cell(
        float(rel.longitude), float(rel.latitude), clon, clat, water)
    anchorN = _nearest_water_cell(
        float(pop.longitude), float(pop.latitude), clon, clat, water)

    relaxations: list[str] = []
    path = None
    fail_day = -1
    for k in VITERBI_K_RELAX:
        cands = _candidate_sets(post, k, anchor0, anchorN, water)
        for mk in MAX_DAILY_KM_RELAX:
            path, fail_day = _viterbi_path(
                post, clon, clat, cands, mk, anchor0, anchorN)
            if path is not None:
                if k != VITERBI_K or mk != MAX_DAILY_KM:
                    relaxations.append(f"K={k}, MAX_DAILY_KM={mk:.0f}")
                break
        if path is not None:
            break

    if path is None:
        return {"track": None, "fail_day": fail_day,
                "relaxations": relaxations, "n_on_land": None,
                "n_land_cross": None, "max_daily_km": None, "k": None}

    plon = clon[path]
    plat = clat[path]
    track = pd.DataFrame({
        "time": times.tz_localize(None) if times.tz is not None else times,
        "longitude": plon, "latitude": plat})
    track.to_csv(RESULTS_DIR / f"fused_track_{shark}.csv", index=False)

    # VERIFY (i): on-land daily vertices (ETOPO z >= 0).
    n_on_land = int(np.sum(_etopo_z_vec(plon, plat) >= 0.0))
    # VERIFY (ii): land-crossing segments between consecutive vertices. Run the
    # adaptive sampler PER SEGMENT (so each step gets its own resolution-aware
    # density), AND cross-check with an INDEPENDENT dense sampler whose spacing is
    # <= LAND_CHECK_KM (0.2 km) on every segment (resolution-aware, NOT a fixed
    # 200-point count — a fixed count spaces ~1 km on a 200 km step and was how the
    # 07_05 gulfext residual crossing slipped through). Both must agree on zero.
    n_land_cross = 0
    n_land_cross_dense = 0
    crossing_segs: list[int] = []
    for s in range(plon.size - 1):
        a = _segment_crosses_land(
            plon[s:s + 1], plat[s:s + 1], plon[s + 1:s + 2], plat[s + 1:s + 2])
        if bool(a[0]):
            n_land_cross += 1
            crossing_segs.append(s)
        # independent dense cross-check, spacing <= LAND_CHECK_KM (0.2 km)
        seg_km = gc_km(plon[s], plat[s], plon[s + 1], plat[s + 1])
        nd = _dense_n_samples(float(seg_km))
        fr = np.linspace(0.0, 1.0, nd)[1:-1]
        dlon = plon[s] + (plon[s + 1] - plon[s]) * fr
        dlat = plat[s] + (plat[s + 1] - plat[s]) * fr
        if np.any(_etopo_z_vec(dlon, dlat) >= 0.0):
            n_land_cross_dense += 1
    if crossing_segs:
        print(f"    [{shark}] land-crossing segments (adaptive): {crossing_segs}",
              flush=True)
    if n_land_cross_dense != n_land_cross:
        print(f"    [{shark}] WARN adaptive ({n_land_cross}) vs dense-200 "
              f"({n_land_cross_dense}) land-cross counts disagree", flush=True)

    used_mk = MAX_DAILY_KM
    used_k = VITERBI_K
    if relaxations:
        last = relaxations[-1]
        used_k = int(last.split("K=")[1].split(",")[0])
        used_mk = float(last.split("MAX_DAILY_KM=")[1])

    return {"track": track, "fail_day": -1, "relaxations": relaxations,
            "n_on_land": n_on_land, "n_land_cross": n_land_cross,
            "max_daily_km": used_mk, "k": used_k}


# %% [markdown]
# ## VERIFY the land-barrier kernel masking before trusting it
#
# Three independent checks on a real level-9 grid spanning Baja California, built
# at a representative σ (the σ-fit explores a range; the masking is σ-independent
# geometry):
#
# - **(a) cross-peninsula pair is blocked** — a Pacific-side ~28°N cell and a
#   Gulf-of-California ~28°N cell ~150 km apart, both within the kernel ring at this
#   σ: its kernel weight must be **0** after masking.
# - **(b) probability conservation** — every kernel row must still sum to **≈ 1**.
# - **(c) open-water pair unchanged** — a nearby open-ocean neighbour pair keeps a
#   non-zero weight (the barrier only removes land-crossing transitions).

# %%
def _verify_barrier_kernel() -> dict:
    """Build plain vs barrier kernels on a Baja grid and run the 3 masking checks."""
    persist_full_domain_bathymetry(9)  # ensure the relaxed-mask canonical grid
    bath = xr.open_dataset(CLEAN_DIR / "bathymetry_healpix_L9.nc")
    blon = np.asarray(bath["longitude"].values)
    blat = np.asarray(bath["latitude"].values)
    bcell = bath["cell_ids"].values.astype("int64")
    # Restrict to a Baja box big enough to contain a cross-peninsula pair.
    box = ((blon >= -116.5) & (blon <= -108.5)
           & (blat >= 24.0) & (blat <= 30.0))
    cells = np.sort(bcell[box])
    grid = xr.Dataset(coords={"cell_ids": ("cells", cells)})
    grid["cell_ids"].attrs.update(
        {"grid_name": "healpix", "level": 9, "indexing_scheme": "nested"})
    grid = to_healpix(grid)
    gi = grid.dggs.grid_info
    clon, clat = hgn.healpix_to_lonlat(cells, 9)
    clon = ((np.asarray(clon) + 180.0) % 360.0) - 180.0
    clat = np.asarray(clat)

    # σ large enough that a ~150 km transition is inside the kernel ring at L9.
    sigma_test = 0.03
    blocked_keys = precompute_blocked_pairs(cells, clon, clat, gi, 9)
    plain = Gaussian1DHealpix(
        cell_ids=cells, grid_info=gi, sigma=sigma_test, truncate=TRUNCATE,
        weights_threshold=1e-8,
        pad_kwargs={"mode": "constant", "constant_value": 0},
        optimize_convolution=True)
    barrier = LandBarrierGaussian1DHealpix(
        cell_ids=cells, grid_info=gi, sigma=sigma_test, truncate=TRUNCATE,
        weights_threshold=1e-8,
        pad_kwargs={"mode": "constant", "constant_value": 0},
        optimize_convolution=True,
        blocked_keys=blocked_keys, barrier_tag="_verify")

    # Pacific-side and Gulf-side ~26-28°N cells (water on both sides of Baja). Search
    # for a (Pacific, Gulf) pair that (i) crosses land and (ii) is within
    # BARRIER_MAX_KM so it is a candidate the barrier set actually contains.
    water = _etopo_z_vec(clon, clat) < 0.0
    pac = np.where(water & (clon < -113.5) & (clat >= 25.5) & (clat <= 28.5))[0]
    gulf = np.where(water & (clon > -112.0) & (clat >= 25.5) & (clat <= 28.5))[0]
    i_pac = i_gulf = -1
    sep_km = np.nan
    seg_blocked = False
    for a in pac:
        d = gc_km(clon[a], clat[a], clon[gulf], clat[gulf])
        near = gulf[d <= BARRIER_MAX_KM]
        if near.size == 0:
            continue
        cr = _segment_crosses_land(
            np.full(near.size, clon[a]), np.full(near.size, clat[a]),
            clon[near], clat[near])
        hit = np.where(cr)[0]
        if hit.size:
            i_pac, i_gulf = int(a), int(near[hit[0]])
            sep_km = float(gc_km(clon[i_pac], clat[i_pac],
                                 clon[i_gulf], clat[i_gulf]))
            seg_blocked = True
            break

    def _w(kernel, out_idx, in_cell_id):
        rows, cols = kernel.coords
        in_cid = np.asarray(barrier.new_cell_ids if kernel is barrier.kernel
                            else plain.new_cell_ids)[cols]
        sel = (rows == out_idx) & (in_cid == in_cell_id)
        return float(kernel.data[sel].sum()) if sel.any() else 0.0

    # (a) cross-peninsula weight: plain vs barrier (out=Gulf cell, in=Pacific cell).
    w_plain = _w(plain.kernel, i_gulf, int(cells[i_pac]))
    w_barrier = _w(barrier.kernel, i_gulf, int(cells[i_pac]))

    # (b) row sums after renormalisation.
    rs = np.zeros(barrier.kernel.shape[0])
    np.add.at(rs, barrier.kernel.coords[0], barrier.kernel.data)
    row_sum_min, row_sum_max = float(rs.min()), float(rs.max())

    # (c) a genuinely open-water Pacific pair: nearest Pacific neighbour whose
    # segment does NOT cross land (search outward until one is found).
    i_nb = -1
    ow_seg_blocked = True
    if i_pac >= 0:
        d = gc_km(clon[i_pac], clat[i_pac], clon[pac], clat[pac])
        order = pac[np.argsort(d)]
        order = order[order != i_pac]
        for cand in order[:50]:
            if not bool(_segment_crosses_land(
                    clon[i_pac:i_pac + 1], clat[i_pac:i_pac + 1],
                    clon[cand:cand + 1], clat[cand:cand + 1])[0]):
                i_nb = int(cand)
                ow_seg_blocked = False
                break
    w_open_plain = _w(plain.kernel, i_pac, int(cells[i_nb])) if i_nb >= 0 else 0.0
    w_open_barrier = _w(barrier.kernel, i_pac, int(cells[i_nb])) if i_nb >= 0 else 0.0

    res = {
        "cross_pair_separation_km": sep_km,
        "cross_segment_crosses_land": seg_blocked,
        "cross_weight_plain": w_plain,
        "cross_weight_barrier": w_barrier,
        "row_sum_min": row_sum_min, "row_sum_max": row_sum_max,
        "open_segment_crosses_land": ow_seg_blocked,
        "open_weight_plain": w_open_plain,
        "open_weight_barrier": w_open_barrier,
        "fallback_rows_verify": int(_BARRIER_FALLBACK_ROWS.get("_verify", 0)),
    }
    print("=== LAND-BARRIER KERNEL VERIFICATION (Baja L9) ===", flush=True)
    print(f" (a) cross-peninsula pair {sep_km:.0f} km, "
          f"segment crosses land={seg_blocked}: "
          f"weight plain={w_plain:.4g} -> barrier={w_barrier:.4g} "
          f"({'PASS' if seg_blocked and w_barrier == 0.0 and w_plain > 0 else 'CHECK'})",
          flush=True)
    print(f" (b) row sums in [{row_sum_min:.6f}, {row_sum_max:.6f}] "
          f"({'PASS' if abs(row_sum_min - 1) < 1e-6 and abs(row_sum_max - 1) < 1e-6 else 'CHECK'})",
          flush=True)
    print(f" (c) open-water pair (crosses land={ow_seg_blocked}): "
          f"weight plain={w_open_plain:.4g} -> barrier={w_open_barrier:.4g} "
          f"({'PASS' if (not ow_seg_blocked) and w_open_barrier > 0 else 'CHECK'})",
          flush=True)
    print(f"     fallback rows on the verify grid = {res['fallback_rows_verify']}",
          flush=True)
    return res


barrier_verification = _verify_barrier_kernel()
with open(RESULTS_DIR / "barrier_verification.json", "w") as fh:
    json.dump(barrier_verification, fh, indent=2)


# %% [markdown]
# ## Run every cleaned tag
#
# First write the canonical full-domain level-9 bathymetry grid (the reusable
# artefact `data/clean/bathymetry_healpix_L9.nc`), then loop the tags.

# %%
persist_full_domain_bathymetry(DEFAULT_HEALPIX_LEVEL)

# Process tags that LACK a current-emission (v4) cache first, so a wall-clock
# interruption completes the not-yet-done tags (incl. any missing referee tag)
# before re-touching tags whose σ is already cached and only need a fast decode.
def _has_current_cache(shark: str) -> bool:
    p = RESULTS_DIR / f"rec_{shark}.json"
    if not p.exists():
        return False
    c = json.loads(p.read_text())
    return (c.get("emission_version") == EMISSION_VERSION
            and c.get("status") == "ok")


_tag_order = sorted(TAGS, key=_has_current_cache)  # un-cached (False) first

records = []
for shark in _tag_order:
    pat_dep, spot_dep, has_referee = TAGS[shark]
    print(f"\n=== {shark} (referee={has_referee}) ===", flush=True)
    cache_path = RESULTS_DIR / f"rec_{shark}.json"

    # σ MUST be re-fitted with the land-barrier movement kernel: a σ cached by the
    # pre-barrier (pure Gaussian) run is no longer the bounds-search optimum once
    # transitions across land are forbidden. Reuse only a σ that was itself fitted
    # with the barrier model (recorded as movement_model == land_barrier...).
    cached_sigma = None
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        cs = cached.get("sigma_fused")
        same_model = (cached.get("movement_model")
                      == "land_barrier_gaussian1d_healpix")
        # Reuse σ only if BOTH the movement model AND the emission recipe match.
        # Changes 1 & 2 alter the emission, so a v2 cache must be re-fit (v3).
        same_emission = (cached.get("emission_version") == EMISSION_VERSION)
        if same_model and same_emission and cs is not None and np.isfinite(cs):
            cached_sigma = float(cs)
            print(f"  reusing barrier+v3-fitted sigma_fused={cached_sigma:.5g}",
                  flush=True)
        elif same_model and cs is not None:
            print(f"  cached sigma_fused={cs:.5g} is from emission "
                  f"'{cached.get('emission_version')}' != '{EMISSION_VERSION}'; "
                  f"re-fitting σ", flush=True)

    level = HEALPIX_LEVEL_OVERRIDE.get(shark, DEFAULT_HEALPIX_LEVEL)
    meta = json.loads(
        (CLEAN_DIR / "tags" / shark / "metadata.json").read_text())
    rec = {
        "tag": shark, "has_referee": has_referee, "healpix_level": level,
        "status": "pending",
        "deploy_days": clean_status.get(shark, {}).get("window_days"),
        "n_twilights": meta.get("n_twilights"),
        "n_argos_fixes": clean_status.get(shark, {}).get("n_argos_class123"),
        "sigma_fused": np.nan, "max_sigma_rad": np.nan, "sigma_at_bound": None,
        "has_temperature": shark in HAS_TEMPERATURE,
        "n_twilights_used": None, "n_land_mask_cells": None,
        "emission_version": EMISSION_VERSION,
        "viz_nursery_is_water": None, "viz_nursery_cell_dist_km": None,
        "depth_emission_active": None,
        "movement_model": "land_barrier_gaussian1d_healpix",
        "n_barrier_fallback_rows": None,
        "decode_mode": "constrained_viterbi_path",
        "n_land_points_before": None, "n_land_points_after": None,
        "n_land_points_mean": None, "n_land_points_mode_raw": None,
        "n_snapped": None,
        "viterbi_max_daily_km": None, "viterbi_k": None,
        "viterbi_relaxations": None, "viterbi_fail_day": None,
        "path_on_land_points": None, "path_land_crossing_segments": None,
        "mode_fused_vs_argos_median_km": np.nan,
        "light_only_err_km": LIGHT_ONLY_ERR_KM.get(shark),
        "temp_only_err_km": TEMP_ONLY_ERR_KM.get(shark),
        "fused_barrier_err_km": FUSED_BARRIER_ERR_KM.get(shark),
        "fused_vs_argos_median_km": np.nan,
        "gpe3_vs_argos_median_km": np.nan,
        "fused_vs_gpe3_median_km": np.nan, "error": None,
    }

    if not clean_status.get(shark, {}).get("cleaned", False):
        rec["status"] = "skipped_clean"
        rec["error"] = clean_status.get(shark, {}).get("reason", "not cleaned")
        print(f"  SKIP (clean stage): {rec['error']}", flush=True)
        records.append(rec)
        continue

    try:
        out = run_fused_hmm(shark, level, cached_sigma=cached_sigma)
        rec["healpix_level"] = out["level"]
        rec["sigma_fused"] = out["sigma"]
        rec["max_sigma_rad"] = out["max_sigma"]
        rec["sigma_at_bound"] = bool(
            (out["sigma"] >= out["max_sigma"] * (1 - 1e-3))
            or (out["sigma"] <= 1e-4 * (1 + 1e-3)))
        rec["n_twilights_used"] = out["n_light_days"]
        rec["n_land_mask_cells"] = out["n_land_mask_cells"]
        rec["n_barrier_fallback_rows"] = out["n_barrier_fallback_rows"]
        rec["n_land_points_before"] = out["n_land_before"]
        rec["n_land_points_after"] = out["n_land_after"]
        rec["decode_mode"] = "mode"
        rec["n_land_points_mean"] = out["n_land_mean"]
        rec["n_land_points_mode_raw"] = out["n_land_mode_raw"]
        rec["n_snapped"] = out["n_snapped"]
        rec["viz_nursery_is_water"] = out["viz_nursery_is_water"]
        rec["viz_nursery_cell_dist_km"] = out["viz_nursery_cell_dist_km"]
        rec["depth_emission_active"] = out["depth_emission_active"]
        print(f"  [coastal mask] Vizcaíno nursery cell is water = "
              f"{out['viz_nursery_is_water']} "
              f"(centroid {out['viz_nursery_cell_dist_km']:.1f} km from cell); "
              f"depth emission active = {out['depth_emission_active']}", flush=True)
        print(f"  [barrier] all-neighbours-blocked fallback rows = "
              f"{out['n_barrier_fallback_rows']}", flush=True)
        print(f"  sigma={out['sigma']:.5g} (max {out['max_sigma']:.4g}); "
              f"level={out['level']}; at_bound={rec['sigma_at_bound']}; "
              f"land pts (ETOPO): mean={out['n_land_mean']} "
              f"mode_raw={out['n_land_mode_raw']} "
              f"mode_snapped={out['n_land_after']} (snapped {out['n_snapped']})",
              flush=True)

        # --- Land-aware most-probable PATH (Change 1) over the cached posterior.
        mode_track = out["track"]  # per-day MAP, kept as the reference comparison
        vit = land_aware_path(shark)
        if vit["track"] is None:
            rec["status"] = "failed"
            rec["viterbi_fail_day"] = vit["fail_day"]
            rec["error"] = (f"no admissible land-aware path; first unreachable "
                            f"day index = {vit['fail_day']} after relaxations "
                            f"{vit['relaxations']}")
            print(f"  PATH FAILED: {rec['error']}", flush=True)
            records.append(rec)
            continue
        track = vit["track"]  # the new HEADLINE land-aware track
        rec["viterbi_max_daily_km"] = vit["max_daily_km"]
        rec["viterbi_k"] = vit["k"]
        rec["viterbi_relaxations"] = "; ".join(vit["relaxations"]) or None
        rec["path_on_land_points"] = vit["n_on_land"]
        rec["path_land_crossing_segments"] = vit["n_land_cross"]
        rec["n_land_points_after"] = vit["n_on_land"]
        print(f"  land-aware path: MAX_DAILY_KM={vit['max_daily_km']:.0f} "
              f"K={vit['k']} relax={vit['relaxations'] or 'none'}; "
              f"on-land vertices={vit['n_on_land']} "
              f"land-crossing segments={vit['n_land_cross']}", flush=True)

        gpe3 = pd.read_csv(CLEAN_DIR / f"gpe3_{shark}.csv", parse_dates=["time"])
        fused_daily = daily_lookup(track)
        gpe3_daily = daily_lookup(gpe3)
        common = fused_daily.index.intersection(gpe3_daily.index)
        if len(common):
            rec["fused_vs_gpe3_median_km"] = float(np.median(gc_km(
                fused_daily.loc[common, "longitude"].values,
                fused_daily.loc[common, "latitude"].values,
                gpe3_daily.loc[common, "longitude"].values,
                gpe3_daily.loc[common, "latitude"].values)))

        mode_daily = daily_lookup(mode_track)
        if has_referee:
            argos = pd.read_csv(
                CLEAN_DIR / f"argos_{shark}.csv", parse_dates=["time"])
            argos["day"] = argos["time"].dt.floor("D")
            rows = []
            for _, fix in argos.iterrows():
                day = fix["day"]
                r = {"time": fix["time"], "quality": fix["quality"]}
                # fused_err_km is now the LAND-AWARE PATH error (headline).
                if day in fused_daily.index:
                    r["fused_err_km"] = gc_km(
                        fix["longitude"], fix["latitude"],
                        fused_daily.loc[day, "longitude"],
                        fused_daily.loc[day, "latitude"])
                if day in mode_daily.index:
                    r["mode_err_km"] = gc_km(
                        fix["longitude"], fix["latitude"],
                        mode_daily.loc[day, "longitude"],
                        mode_daily.loc[day, "latitude"])
                if day in gpe3_daily.index:
                    r["gpe3_err_km"] = gc_km(
                        fix["longitude"], fix["latitude"],
                        gpe3_daily.loc[day, "longitude"],
                        gpe3_daily.loc[day, "latitude"])
                rows.append(r)
            errors = pd.DataFrame(rows)
            errors.to_csv(
                RESULTS_DIR / f"validation_errors_{shark}.csv", index=False)
            if "fused_err_km" in errors:
                rec["fused_vs_argos_median_km"] = float(
                    errors["fused_err_km"].median())
            if "mode_err_km" in errors:
                rec["mode_fused_vs_argos_median_km"] = float(
                    errors["mode_err_km"].median())
            if "gpe3_err_km" in errors:
                rec["gpe3_vs_argos_median_km"] = float(
                    errors["gpe3_err_km"].median())
            print(f"  PATH-vs-Argos median = "
                  f"{rec['fused_vs_argos_median_km']:.1f} km "
                  f"(mode-track {rec['mode_fused_vs_argos_median_km']:.1f}, "
                  f"temp-only {TEMP_ONLY_ERR_KM.get(shark)}, "
                  f"GPE3 {rec['gpe3_vs_argos_median_km']:.1f})", flush=True)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        print(f"  FAILED: {rec['error']}", flush=True)

    records.append(rec)
    if rec["status"] == "ok":
        cache_path.write_text(json.dumps(rec))


# %% [markdown]
# ## Aggregate into results/summary.csv + results/aggregate.json

# %%
# Pre-bathy backup of the previous summary (the fused-barrier baseline run), kept
# once for the before/after comparison. Never overwrite an existing backup.
_prebathy = RESULTS_DIR / "summary_prebathy.csv"
_cur = RESULTS_DIR / "summary.csv"
if _cur.exists() and not _prebathy.exists():
    _prebathy.write_text(_cur.read_text())

# Restore canonical tag order (the loop ran un-cached tags first for resilience).
_canon = {t: i for i, t in enumerate(TAGS)}
records = sorted(records, key=lambda r: _canon.get(r["tag"], 999))

summary = pd.DataFrame(records)[[
    "tag", "deploy_days", "has_referee", "has_temperature", "healpix_level",
    "n_twilights", "n_twilights_used", "n_argos_fixes",
    "sigma_fused", "max_sigma_rad", "sigma_at_bound",
    "movement_model", "n_barrier_fallback_rows", "decode_mode",
    "viterbi_max_daily_km", "viterbi_k", "viterbi_relaxations",
    "path_on_land_points", "path_land_crossing_segments",
    "n_land_mask_cells", "n_land_points_before", "n_land_points_mean",
    "n_land_points_mode_raw", "n_snapped", "n_land_points_after",
    "emission_version", "depth_emission_active",
    "viz_nursery_is_water", "viz_nursery_cell_dist_km",
    "light_only_err_km", "temp_only_err_km", "fused_barrier_err_km",
    "mode_fused_vs_argos_median_km", "fused_vs_argos_median_km",
    "gpe3_vs_argos_median_km", "fused_vs_gpe3_median_km", "status", "error",
]]
summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

referee_ok = summary[(summary["has_referee"]) & (summary["status"] == "ok")]
pooled = (float(referee_ok["fused_vs_argos_median_km"].median())
          if len(referee_ok) else None)
ok_sigmas = summary[summary["status"] == "ok"]
n_off_bound = int((~ok_sigmas["sigma_at_bound"].astype("boolean").fillna(True)).sum())

agg = {
    "n_tags_total": int(len(summary)),
    "n_tags_ok": int((summary["status"] == "ok").sum()),
    "n_referee_tags_ok": int(len(referee_ok)),
    "fused_vs_argos_pooled_median_km": pooled,
    "gpe3_vs_argos_pooled_median_km": (
        float(referee_ok["gpe3_vs_argos_median_km"].median())
        if len(referee_ok) else None),
    "light_only_pooled_median_km": LIGHT_ONLY_POOLED_KM,
    "temp_only_pooled_median_km": TEMP_ONLY_POOLED_KM,
    "fused_barrier_pooled_median_km": FUSED_BARRIER_POOLED_KM,
    "gpe3_baseline_km": GPE3_BASELINE_KM,
    "emission_version": EMISSION_VERSION,
    "n_tags_sigma_off_bound": n_off_bound,
    "sigma_off_bound": bool(n_off_bound > 0),
    "fused_below_temp_floor": (
        bool(pooled is not None and pooled < TEMP_ONLY_POOLED_KM)),
    "fused_below_barrier_baseline": (
        bool(pooled is not None and pooled < FUSED_BARRIER_POOLED_KM)),
    "fused_at_or_below_gpe3": (
        bool(pooled is not None and pooled <= GPE3_BASELINE_KM)),
    # Per-tag delta vs the fused-barrier baseline (negative = improvement).
    "delta_vs_barrier_km": {
        str(r["tag"]): (
            float(r["fused_vs_argos_median_km"] - r["fused_barrier_err_km"])
            if pd.notna(r["fused_vs_argos_median_km"])
            and pd.notna(r["fused_barrier_err_km"]) else None)
        for _, r in referee_ok.iterrows()},
    "tag_08_01_improved_with_relaxed_nursery": (
        bool(referee_ok.set_index("tag").loc["08_01", "fused_vs_argos_median_km"]
             < FUSED_BARRIER_ERR_KM["08_01"])
        if "08_01" in set(referee_ok["tag"]) else None),
    "all_referee_nursery_cells_water": (
        bool(referee_ok["viz_nursery_is_water"].fillna(False).all())
        if len(referee_ok) else None),
    "decode_mode": "constrained_viterbi_path",
    "movement_model": "land_barrier_gaussian1d_healpix",
    "total_barrier_fallback_rows": int(
        summary["n_barrier_fallback_rows"].dropna().sum()),
    "n_blocked_pairs_cached": int(sum(_BLOCKED_PAIR_CACHE.values())),
    "n_pairs_cached": int(len(_BLOCKED_PAIR_CACHE)),
    "mode_fused_vs_argos_pooled_median_km": (
        float(referee_ok["mode_fused_vs_argos_median_km"].median())
        if len(referee_ok) else None),
    "path_total_on_land_points": int(
        summary["path_on_land_points"].dropna().sum()),
    "path_total_land_crossing_segments": int(
        summary["path_land_crossing_segments"].dropna().sum()),
    "total_land_points_before": int(
        summary["n_land_points_before"].dropna().sum()),
    "total_land_points_mean": int(
        summary["n_land_points_mean"].dropna().sum()),
    "total_land_points_mode_raw": int(
        summary["n_land_points_mode_raw"].dropna().sum()),
    "total_land_points_snapped": int(summary["n_snapped"].dropna().sum()),
    "total_land_points_after": int(
        summary["n_land_points_after"].dropna().sum()),
}
with open(RESULTS_DIR / "aggregate.json", "w") as fh:
    json.dump(agg, fh, indent=2)

print("\n========== SUMMARY ==========", flush=True)
print(summary.to_string(index=False))
print("\n========== AGGREGATE ==========")
print(json.dumps(agg, indent=2))
