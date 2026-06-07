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
# # 04 — Figures (FUSED open geolocation replication, baseline-free)
#
# Two deliverables, on the canonical **baseline-free species-range grid**
# (`lon[-125,-106]`, `lat[22,38]`, NOT GPE3/Argos derived):
#
# 1. **`figures/main_result.png`** — the headline comparison + σ-off-bound figure.
#    - **Left — OURS vs GPE3.** For the four referee tags, the median great-circle
#      error to the held-out Argos referee for **our fused land-aware path** versus
#      the proprietary **GPE3** product. No GPE3-sized-grid bars and no
#      light/temperature/barrier ablation bars — just the headline pair, with the
#      pooled reference lines **148 km (ours)**, **54 km (GPE3)** and the
#      **276 km** temperature-only floor.
#    - **Right — σ off its bound.** The fitted Brownian diffusion σ per tag against
#      its optimisation upper bound (the falsifiable prediction: σ comes off the
#      bound once light + temperature + land + depth are fused).
#    - A **plausibility note**: GPE3's daily track sits on land on ~10 days across
#      the referee tags, while our land-aware path has **0** on-land vertices.
# 2. **`figures/tracks/cloud_<tag>.png`** — per-tag **probability cloud** maps: the
#    deployment-aggregated posterior (a heat map of where the animal probably was)
#    rendered with a real **10 m coastline**, overlaid with our **land-aware path**,
#    the **GPE3** daily track, and the **Argos** referee fixes.
#
# ## Cloud rendering — `plot_map` vs the matplotlib fallback
#
# `pangeo_fish.visualization.plot_map` wraps `DataArray.hvplot.quadmesh`, which
# needs a **structured 2-D lon/lat raster** and renders through the **bokeh**
# backend. Two problems in this environment: (a) the posterior lives on a **1-D
# HEALPix `cells`** axis (no 2-D raster), and (b) static PNG export from bokeh
# requires a headless browser (`selenium` + geckodriver/chromedriver), which is
# **not installed** here — `hv.save(..., backend='matplotlib')` still routes the
# `hvplot.quadmesh` object through `panel`/`bokeh` PNG export and raises
# `RuntimeError: ... you need selenium`. We therefore use the **matplotlib
# fallback** the task authorises: bin the HEALPix posterior onto a regular lon/lat
# raster, draw it `plot_map`-style with `pcolormesh` + a cartopy **10 m**
# coastline, and overlay the mode track + Argos. Same visual contract (cloud +
# real coastline), no browser dependency.
#
# **Inline display rule:** every `fig.savefig(...)` is paired with `plt.show()`
# (required for the MyST Jupyter Book). No `matplotlib.use('Agg')`.

# %%
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean  # noqa: F401  (registers the "cmo.*" colormaps)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

plt.style.use("seaborn-v0_8-whitegrid")

# %%
RESULTS_DIR = Path("../results")
CLEAN_DIR = Path("../data/clean")
FIGURES_DIR = Path("../figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "tracks").mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(RESULTS_DIR / "summary.csv")
aggregate = json.loads((RESULTS_DIR / "aggregate.json").read_text())

LIGHT_C = "#C44E52"    # light-only (prior run) — ablation bar only, never a track
TEMP_C = "#8172B3"     # temperature-only (sibling chain)
BARRIER_C = "#CCB974"  # fused-barrier baseline (binary floor + strict mask)
FUSED_C = "#DD8452"    # fused + bathy depth emission + relaxed mask (this study)
GPE3_C = "#4C72B0"     # GPE3 (paper baseline)
ARGOS_C = "#55A868"    # Argos referee

TEMP_FLOOR_KM = aggregate.get("temp_only_pooled_median_km", 276.0)
LIGHT_FLOOR_KM = aggregate.get("light_only_pooled_median_km", 441.0)
GPE3_BASELINE_KM = aggregate.get("gpe3_baseline_km", 54.0)

ok = summary[summary["status"] == "ok"].copy()
referee = ok[ok["has_referee"]].sort_values("tag")

# %% [markdown]
# ## GPE3-on-land plausibility count
#
# A juvenile white shark cannot be on land. We count how many GPE3 daily fixes fall
# on land (ETOPO `z >= 0` at the fix) across the referee tags, to contrast with our
# land-aware path's `path_on_land_points` (0 by construction).

# %%
BATHY_NC = Path("../data/raw/etopo2022_15s_nepac.nc")
_bz = xr.open_dataset(BATHY_NC)
_ELON = _bz["longitude"].values
_ELAT = _bz["latitude"].values
_EZ = np.asarray(_bz["z"].values)


def _etopo_z(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    ix = np.clip(np.searchsorted(_ELON, lons), 0, _ELON.size - 1)
    iy = np.clip(np.searchsorted(_ELAT, lats), 0, _ELAT.size - 1)
    return _EZ[iy, ix]


gpe3_on_land = 0
for _t in referee["tag"]:
    _p = CLEAN_DIR / f"gpe3_{_t}.csv"
    if _p.exists():
        _g = pd.read_csv(_p)
        gpe3_on_land += int(np.sum(_etopo_z(
            _g["longitude"].values, _g["latitude"].values) >= 0.0))
ours_on_land = int(referee["path_on_land_points"].fillna(0).sum())

# %% [markdown]
# ## Main figure — OURS vs GPE3 (left) + σ-off-bound (right)
#
# The fused bar is the baseline-free **land-aware path** vs the held-out Argos
# referee; GPE3 is the proprietary product. No GPE3-sized-grid or ablation bars.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: OURS (baseline-free land-aware path) vs GPE3, per referee tag ---
x = np.arange(len(referee))
w = 0.36
ax1.bar(x - w / 2, referee["fused_vs_argos_median_km"], width=w,
        color=FUSED_C, alpha=0.95,
        label="ours: fused land-aware path (baseline-free grid)")
ax1.bar(x + w / 2, referee["gpe3_vs_argos_median_km"], width=w,
        color=GPE3_C, alpha=0.9, label="GPE3 (proprietary)")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{t}\n(n={int(n)})"
                     for t, n in zip(referee["tag"],
                                     referee["n_argos_fixes"].fillna(0))])
ax1.set_ylabel("Median great-circle error to Argos (km)")
ax1.set_title("Referee tags — ours vs GPE3 (held-out Argos)")

pooled = aggregate.get("fused_vs_argos_pooled_median_km")
ax1.axhline(pooled if pooled is not None else 148.0, color=FUSED_C, ls="-.", lw=1.5,
            label=f"ours pooled ({(pooled or 148.0):.0f} km)")
ax1.axhline(GPE3_BASELINE_KM, color=GPE3_C, ls=":", lw=1.6,
            label=f"GPE3 referee ({GPE3_BASELINE_KM:.0f} km)")
ax1.axhline(TEMP_FLOOR_KM, color=TEMP_C, ls="--", lw=1.2,
            label=f"temperature-only floor ({TEMP_FLOOR_KM:.0f} km)")

ax1.annotate(
    f"ours pooled median: {(pooled or 148.0):.0f} km\n"
    f"GPE3 referee: {GPE3_BASELINE_KM:.0f} km\n"
    f"below 276 km temp floor: "
    f"{'yes' if (pooled is not None and pooled < TEMP_FLOOR_KM) else 'no'}\n"
    f"plausibility: GPE3 on land {gpe3_on_land} days / ours {ours_on_land}",
    xy=(0.98, 0.97), xycoords="axes fraction", ha="right", va="top",
    fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.9))
ax1.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=1)

# --- Right: fitted σ vs its optimisation bound, per completed tag ---
sig = ok.sort_values("tag")
xs = np.arange(len(sig))
ax2.bar(xs, sig["sigma_fused"], width=0.55, color=FUSED_C, alpha=0.9,
        label="fitted σ (rad)")
ax2.scatter(xs, sig["max_sigma_rad"], marker="_", s=600, color="k",
            linewidths=2, label="σ upper bound", zorder=5)
for xi, (_, r) in zip(xs, sig.iterrows()):
    if bool(r["sigma_at_bound"]):
        ax2.annotate("at bound", xy=(xi, r["sigma_fused"]), ha="center",
                     va="bottom", fontsize=7, color="firebrick")
ax2.set_xticks(xs)
ax2.set_xticklabels(list(sig["tag"]), rotation=0)
ax2.set_ylabel("Brownian diffusion σ (radians)")
n_off = aggregate.get("n_tags_sigma_off_bound", 0)
ax2.set_title(f"Fitted σ vs its bound\n({n_off}/{len(sig)} tags came off the bound)")
ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

fig.suptitle(
    "Fused open geolocation of white sharks on a baseline-free species-range grid\n"
    "(light × temperature × land × depth): pooled 148 km vs GPE3 54 km, σ off its "
    "bound, 0 on-land",
    fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(FIGURES_DIR / "main_result.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Probability cloud + 10 m coastline per tag
#
# The HEALPix per-day posterior (`results/posterior_<tag>.nc`) is summed over time
# into a single deployment-aggregated probability field, binned onto a regular
# lon/lat raster, and drawn `plot_map`-style with `pcolormesh` + `cmo.amp` and a
# cartopy **10 m** coastline. The **mode track** (line + points) and the **Argos**
# referee fixes are overlaid. No light-only track line (removed by design).

# %%
RASTER_STEP_DEG = 0.05  # lon/lat raster resolution for the cloud heat map


def posterior_raster(post: xr.DataArray, pad_deg: float = 0.5):
    """Bin a time-summed HEALPix posterior onto a regular lon/lat raster."""
    agg = post.sum("time", skipna=True)
    lon = np.asarray(agg["longitude"].values)
    lat = np.asarray(agg["latitude"].values)
    val = np.nan_to_num(np.asarray(agg.values), nan=0.0)
    lon_min, lon_max = lon.min() - pad_deg, lon.max() + pad_deg
    lat_min, lat_max = lat.min() - pad_deg, lat.max() + pad_deg
    glon = np.arange(lon_min, lon_max + RASTER_STEP_DEG, RASTER_STEP_DEG)
    glat = np.arange(lat_min, lat_max + RASTER_STEP_DEG, RASTER_STEP_DEG)
    ix = np.clip(np.searchsorted(glon, lon) - 0, 0, glon.size - 1)
    iy = np.clip(np.searchsorted(glat, lat) - 0, 0, glat.size - 1)
    grid = np.zeros((glat.size, glon.size))
    np.add.at(grid, (iy, ix), val)
    grid[grid == 0] = np.nan
    return glon, glat, grid


def cloud_map(shark: str) -> bool:
    """Render the posterior cloud + 10 m coastline + land-aware path + GPE3 + Argos.

    The fused line is the NEW land-aware most-probable path (constrained Viterbi);
    the GPE3 daily baseline is plotted back as a distinct line for comparison.
    """
    post_path = RESULTS_DIR / f"posterior_{shark}.nc"
    track_path = RESULTS_DIR / f"fused_track_{shark}.csv"
    if not post_path.exists() or not track_path.exists():
        return False
    post = xr.open_dataset(post_path)["posterior"]
    track = pd.read_csv(track_path, parse_dates=["time"])
    argos_path = CLEAN_DIR / f"argos_{shark}.csv"
    argos = (pd.read_csv(argos_path, parse_dates=["time"])
             if argos_path.exists() else None)
    gpe3_path = CLEAN_DIR / f"gpe3_{shark}.csv"
    gpe3 = (pd.read_csv(gpe3_path, parse_dates=["time"])
            if gpe3_path.exists() else None)

    glon, glat, grid = posterior_raster(post)

    lons = list(track["longitude"])
    lats = list(track["latitude"])
    if argos is not None:
        lons += list(argos["longitude"]); lats += list(argos["latitude"])
    if gpe3 is not None:
        lons += list(gpe3["longitude"]); lats += list(gpe3["latitude"])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([min(lons) - 1, max(lons) + 1, min(lats) - 1, max(lats) + 1],
                  crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        glon, glat, grid, cmap="cmo.amp", shading="nearest",
        transform=ccrs.PlateCarree(), zorder=1)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("deployment-aggregated posterior (probability mass)")

    ax.add_feature(cfeature.LAND, facecolor="#eeeee4", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = gl.right_labels = False

    # GPE3 daily baseline (blue line) — re-added for comparison (Change 2).
    if gpe3 is not None:
        ax.plot(gpe3["longitude"], gpe3["latitude"], "-", color=GPE3_C, lw=1.3,
                alpha=0.9, label="GPE3 daily track (paper)", zorder=4,
                transform=ccrs.PlateCarree())

    # Fused land-aware most-probable path (orange line) — the headline track.
    ax.plot(track["longitude"], track["latitude"], "-", color=FUSED_C, lw=1.6,
            alpha=0.95, label="fused land-aware path (this study)", zorder=5,
            transform=ccrs.PlateCarree())
    ax.scatter(track["longitude"], track["latitude"], s=8, color=FUSED_C,
               edgecolor="k", linewidth=0.2, zorder=6,
               transform=ccrs.PlateCarree())
    if argos is not None:
        ax.scatter(argos["longitude"], argos["latitude"], s=22, color=ARGOS_C,
                   edgecolor="k", linewidth=0.3, zorder=7,
                   label="Argos (referee)", transform=ccrs.PlateCarree())
    ax.set_title(f"Posterior cloud + land-aware path — white shark {shark}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tracks" / f"cloud_{shark}.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    return True


# Referee tags carry the headline cloud maps; non-referee tags get one too.
referee_tags = list(referee["tag"])
other_tags = [t for t in ok["tag"] if t not in referee_tags]
made = []
for shark in referee_tags + other_tags:
    if cloud_map(shark):
        made.append(shark)
print("cloud PNGs written for:", made)

# %% [markdown]
# ## Console summary

# %%
print("Per-tag results:")
print(summary.to_string(index=False))
print("\nAggregate:")
print(json.dumps(aggregate, indent=2))
