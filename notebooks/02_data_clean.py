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
# # 02 — Data clean (light-level geolocation)
#
# Turns the raw archive into analysis-ready inputs for the pangeo-fish HMM with a
# **light / twilight emission** (notebook 03), for every recovered-PAT tag. For
# each tag we build, under per-tag paths:
#
# ```
# data/clean/tags/<tag>/twilights.csv       time (UTC), event ("sunrise"/"sunset")
# data/clean/tags/<tag>/tagging_events.csv  event_name, time, lon, lat
# data/clean/tags/<tag>/dst.csv             time, pressure (m), temperature (degC)
# data/clean/tags/<tag>/metadata.json
# data/clean/gpe3_<tag>.csv                 GPE3 daily baseline track
# data/clean/argos_<tag>.csv                SPOT Argos referee (referee tags only)
# data/clean/reference_model_<tag>_gulfext.nc  GLORYS species-range ocean fields
# ```
#
# The **DST** (`dst.csv`) and **reference model** are built only for the six
# temperature tags (`02_01` is light-only). The reference model is assembled from
# the raw GLORYS subset downloaded by notebook 01 over the baseline-free
# species-range box `lon[-125,-106]`, `lat[22,38]`; both DST cleaning and the
# reference-model build replace the previous cross-repo dependency on the sibling
# temperature-replication repository, so this repo is now self-contained.
#
# The scientific input for this study is each tag's **detected twilight events**
# (`out-LightLoc.csv`): the on-tag dawn/dusk detections, whose `Day`+`Time`
# columns are stored in **UTC**. We map `Dawn -> "sunrise"` and `Dusk ->
# "sunset"`. These are exactly the events an open light-level geolocation method
# (probGLS / FLightR / TwGeos style) scores against an astronomical
# sunrise/sunset model.
#
# **In-scope tag list — re-derived, not copied.** The light emission needs only
# light, so a tag enters the analysis if its archive carries usable twilight
# events. This **re-admits `02_01`**, the 2001 PAT2 that the temperature chain
# dropped for lacking an external water-temperature sensor: it still records
# light and twilights. A tag with too few twilights is recorded skipped with the
# exact count and reason — nothing is fabricated.
#
# **Anti-circularity note.** `out-LightLoc.csv` also carries the tag's own
# `InitLat`/`InitLon` and `SSTTime`/`SSTTemp` seed columns. We deliberately do
# **not** use them: the only thing taken from light is the twilight *times*; the
# spatial likelihood is built astronomically in notebook 03, and the only spatial
# anchors are the release/pop-up endpoints (which are never used as the referee).

# %%
import json
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# %%
RAW_DIR = Path("../data/raw")
CLEAN_DIR = Path("../data/clean")
TAGS_DIR = CLEAN_DIR / "tags"
TAGS_DIR.mkdir(parents=True, exist_ok=True)
TAG_PKG_DIR = RAW_DIR / "tag_packages"
GLORYS_DIR = RAW_DIR / "glorys"

# Tags carrying an external water-temperature sensor (the temperature factor); a
# DST log + GLORYS reference model are built for these. 02_01 is light-only.
TEMPERATURE_TAGS = ["07_05", "08_01", "08_02", "08_09", "06_10", "07_01"]

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

# A tag needs at least this many detected twilights to support a light fit.
MIN_TWILIGHTS = 10

# %%
sharks = pd.read_csv(RAW_DIR / "JWS_metadata.csv", dtype=str, encoding="latin-1")
sharks = sharks.apply(lambda c: c.str.strip() if c.dtype == "object" else c)

# %% [markdown]
# ## Per-tag cleaning helpers

# %%
def clean_twilights(pat_dep: str) -> pd.DataFrame:
    """Detected dawn/dusk twilights from a PAT ZIP `out-LightLoc.csv`.

    The `Day`+`Time` columns are stored in UTC. We keep only `Dawn`/`Dusk`
    rows, map them to `sunrise`/`sunset`, and return a tidy `(time, event)`
    table sorted in time.
    """
    zf = zipfile.ZipFile(TAG_PKG_DIR / f"{pat_dep}.zip")
    with zf.open("out-LightLoc.csv") as fh:
        ll = pd.read_csv(fh, encoding="latin-1", low_memory=False)
    ll.columns = [c.strip() for c in ll.columns]
    ll["Type"] = ll["Type"].astype(str).str.strip()
    tw = ll[ll["Type"].isin(["Dawn", "Dusk"])].copy()
    tw["time"] = pd.to_datetime(
        tw["Day"].astype(str).str.strip() + " "
        + tw["Time"].astype(str).str.strip(),
        format="%d-%b-%y %H:%M:%S", errors="coerce")
    tw = tw.dropna(subset=["time"])
    tw["event"] = tw["Type"].map({"Dawn": "sunrise", "Dusk": "sunset"})
    return (tw[["time", "event"]].sort_values("time")
            .drop_duplicates(subset=["time"]).reset_index(drop=True))


def tagging_events(shark: str) -> pd.DataFrame:
    row = sharks[sharks["SHARK_ID"] == shark].iloc[0]
    return pd.DataFrame({
        "event_name": ["release", "fish_death"],
        "time": [pd.to_datetime(row["DATE_START"]),
                 pd.to_datetime(row["PAT_END"])],
        "longitude": [float(row["LON_REL"]), float(row["LON_END_PAT"])],
        "latitude": [float(row["LAT_REL"]), float(row["LAT_END_PAT"])],
    })


def clean_gpe3(pat_dep: str) -> pd.DataFrame:
    """Daily most-likely GPE3 positions (glob the `*-GPE3.csv`)."""
    zf = zipfile.ZipFile(TAG_PKG_DIR / f"{pat_dep}.zip")
    name = next(n for n in zf.namelist() if n.endswith("-GPE3.csv"))
    raw = zf.open(name).read().decode("latin-1")
    lines = raw.splitlines()
    hdr = next(i for i, ln in enumerate(lines)
               if (not ln.startswith(";")) and ("Most Likely Latitude" in ln))
    gpe3 = pd.read_csv(StringIO("\n".join(lines[hdr:])))
    gpe3["time"] = pd.to_datetime(gpe3["Date"], errors="coerce")
    gpe3 = (
        gpe3.dropna(subset=["time", "Most Likely Latitude",
                            "Most Likely Longitude"])
        .rename(columns={"Most Likely Latitude": "latitude",
                         "Most Likely Longitude": "longitude"})
        [["time", "latitude", "longitude"]]
        .sort_values("time")
    )
    return (gpe3.set_index("time")[["latitude", "longitude"]]
            .resample("1D").mean().dropna().reset_index())


def clean_argos(spot_dep: str) -> pd.DataFrame:
    """SPOT Argos fixes, classes 1/2/3 only (error < ~1.5 km)."""
    zf = zipfile.ZipFile(TAG_PKG_DIR / f"{spot_dep}.zip")
    loc_name = next(n for n in zf.namelist() if n.endswith("-Locations.csv"))
    with zf.open(loc_name) as fh:
        spot = pd.read_csv(fh)
    spot["time"] = pd.to_datetime(spot["Date"], errors="coerce")
    return (
        spot[spot["Type"] == "Argos"]
        .assign(quality=lambda d: d["Quality"].astype(str))
        .loc[lambda d: d["quality"].isin(["1", "2", "3"])]
        .dropna(subset=["time", "Latitude", "Longitude"])
        .rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
        [["time", "latitude", "longitude", "quality"]]
        .sort_values("time").reset_index(drop=True)
    )


# %% [markdown]
# ## Native DST (depth–temperature) cleaning
#
# The fused HMM's **temperature** factor needs each tag's external-temperature dive
# record. We clean it **natively** from the PAT archive's `out-Archive.csv` (which
# carries the dense `Depth` + `External Temperature` time series, `Time` in the tag
# clock `%H:%M:%S %d-%b-%Y`), so the repo no longer reads the sibling's DST. We use
# the **exact** proven recipe of the prior temperature replication: keep `Depth` as
# `pressure` and the **External** (ambient water) `Temperature`, clip negative
# pressures to 0, and **resample to 10-minute means** (`DST_RESAMPLE`). The 10-min
# resample bounds memory (the raw record is ~2.5 M samples) while preserving the
# vertical structure the HMM matches against GLORYS depth levels — pangeo-fish then
# bins this to the GLORYS daily axis (`reshape_by_bins`). A tag whose archive has
# **no External Temperature** column (the 2001 PAT2 `02_01`, internal sensor only)
# returns `None` and gets no DST. Output `data/clean/tags/<tag>/dst.csv` is
# time-indexed (tz-aware UTC) with columns `pressure` (m), `temperature` (degC), as
# `pangeo_fish.io.open_tag` expects.

# %%
# Resample cadence for the DST (matches the prior temperature replication exactly).
DST_RESAMPLE = "10min"


def clean_dst(pat_dep: str) -> pd.DataFrame | None:
    """Depth + External-Temperature series from a PAT ZIP `out-Archive.csv`.

    Returns a 10-minute-mean, time-indexed ``(pressure, temperature)`` frame, or
    ``None`` if the archive has no External Temperature sensor (PAT2 records only
    an internal recorder temperature, not ambient water).
    """
    zf = zipfile.ZipFile(TAG_PKG_DIR / f"{pat_dep}.zip")
    with zf.open("out-Archive.csv") as fh:
        head = pd.read_csv(fh, nrows=1)
    if "External Temperature" not in head.columns:
        return None
    with zf.open("out-Archive.csv") as fh:
        arch = pd.read_csv(
            fh, usecols=["Time", "Depth", "External Temperature"],
            low_memory=False)
    arch["time"] = pd.to_datetime(
        arch["Time"], format="%H:%M:%S %d-%b-%Y", errors="coerce")
    dst = (
        arch.dropna(subset=["time", "Depth", "External Temperature"])
        .rename(columns={"Depth": "pressure",
                         "External Temperature": "temperature"})
        .set_index("time")[["pressure", "temperature"]]
        .sort_index())
    dst["pressure"] = dst["pressure"].clip(lower=0.0)
    dst = dst.resample(DST_RESAMPLE).mean().dropna()
    return dst


# %% [markdown]
# ## GLORYS reference-model builder (baseline-free species-range grid)
#
# Build `data/clean/reference_model_<tag>_gulfext.nc` from the raw GLORYS subset
# downloaded by notebook 01 (`data/raw/glorys/`). The recipe matches the gulfext
# reference model the temperature emission reads in notebook 03: rename
# `thetao→TEMP`, `zos→XE`, attach static `deptho→H0`, derive `dynamic_depth` and
# `dynamic_bathymetry`, and add 2-D `latitude`/`longitude` coordinates. Idempotent:
# a tag whose reference model already exists is left untouched.

# %%
def build_reference_model(shark: str) -> Path | None:
    """Build reference_model_<shark>_gulfext.nc from raw GLORYS; skip if present."""
    out = CLEAN_DIR / f"reference_model_{shark}_gulfext.nc"
    if out.exists():
        return out
    thetao_nc = GLORYS_DIR / f"glorys_thetao_{shark}_gulfext.nc"
    static_nc = GLORYS_DIR / "glorys_static_gulfext.nc"
    if not (thetao_nc.exists() and static_nc.exists()):
        return None  # raw GLORYS not on disk (notebook 01 skipped or no creds)

    thetao = xr.open_dataset(thetao_nc)
    static = xr.open_dataset(static_nc)
    static_on = static.interp(latitude=thetao["latitude"],
                              longitude=thetao["longitude"], method="nearest")
    deptho = static_on["deptho"]
    if "latitude" in deptho.dims:
        deptho = deptho.rename({"latitude": "lat", "longitude": "lon"})

    model = (
        thetao.rename({"thetao": "TEMP", "zos": "XE",
                       "latitude": "lat", "longitude": "lon"})
        .assign(H0=deptho))
    if "depth" in model["XE"].dims:
        model["XE"] = model["XE"].isel(depth=0, drop=True)
    model = model[["TEMP", "XE", "H0"]]
    model = model.assign(
        dynamic_depth=(model["depth"] + model["XE"]).assign_attrs(
            units="m", positive="down"),
        dynamic_bathymetry=(model["H0"] + model["XE"]).assign_attrs(
            units="m", positive="down"))
    if "units" not in model["TEMP"].attrs:
        model["TEMP"].attrs["units"] = "degC"
    lon2d, lat2d = np.meshgrid(model["lon"].values, model["lat"].values)
    model = model.assign_coords(
        latitude=(("lat", "lon"), lat2d), longitude=(("lat", "lon"), lon2d))
    model.attrs.update(thetao.attrs)
    model.attrs["sensitivity_note"] = (
        f"{shark} BASELINE-FREE species-range grid lon[-125,-106] lat[22,38] "
        "(NOT GPE3/Argos derived)")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.to_netcdf(out)
    return out


# %% [markdown]
# ## Clean every tag
#
# Each tag is processed independently; failures are recorded with their exact
# reason and the loop continues — a partial clean is an honest outcome.

# %%
clean_status = {}
for shark, (pat_dep, spot_dep, has_referee) in TAGS.items():
    print(f"=== {shark} (PAT {pat_dep}, referee={has_referee}) ===")
    status = {"pat_deploy_id": pat_dep, "spot_deploy_id": spot_dep,
              "has_referee": has_referee, "cleaned": False, "reason": None}
    try:
        tw = clean_twilights(pat_dep)
        if len(tw) < MIN_TWILIGHTS:
            status["reason"] = (
                f"only {len(tw)} detected twilights (< {MIN_TWILIGHTS}); "
                "too few to support a light geolocation fit")
            print(f"  SKIP: {status['reason']}")
            clean_status[shark] = status
            continue

        row = sharks[sharks["SHARK_ID"] == shark].iloc[0]
        window_days = (pd.to_datetime(row["PAT_END"])
                       - pd.to_datetime(row["DATE_START"])).days

        tag_root = TAGS_DIR / shark
        tag_root.mkdir(parents=True, exist_ok=True)
        tw_out = tw.copy()
        tw_out["time"] = pd.to_datetime(tw_out["time"]).dt.tz_localize("UTC")
        tw_out.to_csv(tag_root / "twilights.csv", index=False)

        events = tagging_events(shark)
        events_out = events.copy()
        events_out["time"] = pd.to_datetime(
            events_out["time"]).dt.tz_localize("UTC")
        events_out.to_csv(tag_root / "tagging_events.csv", index=False)

        (tag_root / "metadata.json").write_text(json.dumps({
            "tag_name": shark, "shark_id": shark,
            "pat_model": row["PAT_MODEL"], "pat_id": row["PAT_ID"],
            "sex": row["SEX"], "tbl_cm": row["TBL_cm"],
            "release_location": row["LOCATION_REL"],
            "window_days": int(window_days),
            "has_referee": has_referee,
            "n_twilights": int(len(tw)),
            "n_sunrise": int((tw["event"] == "sunrise").sum()),
            "n_sunset": int((tw["event"] == "sunset").sum()),
        }, indent=2))

        gpe3 = clean_gpe3(pat_dep)
        gpe3.to_csv(CLEAN_DIR / f"gpe3_{shark}.csv", index=False)

        n_argos = 0
        if has_referee and spot_dep is not None:
            argos = clean_argos(spot_dep)
            argos.to_csv(CLEAN_DIR / f"argos_{shark}.csv", index=False)
            n_argos = len(argos)

        # Native DST (depth-temperature) for the temperature factor, written to the
        # pangeo-fish tag layout (data/clean/tags/<tag>/dst.csv), tz-aware UTC index.
        # A tag with no External Temperature sensor (02_01 PAT2) returns None → no DST.
        n_dst = 0
        ref_built = None
        if shark in TEMPERATURE_TAGS:
            dst = clean_dst(pat_dep)
            if dst is not None:
                dst_out = dst.copy()
                dst_out.index = dst_out.index.tz_localize("UTC")
                dst_out.to_csv(tag_root / "dst.csv")
                n_dst = len(dst_out)
                # Build the baseline-free GLORYS reference model (idempotent; needs
                # the raw GLORYS subset from notebook 01, else None).
                ref = build_reference_model(shark)
                ref_built = (ref.name if ref is not None else None)

        status.update({
            "cleaned": True, "window_days": int(window_days),
            "n_twilights": int(len(tw)), "n_gpe3_days": int(len(gpe3)),
            "n_argos_class123": int(n_argos),
            "n_dst_rows": int(n_dst),
            "reference_model": ref_built,
        })
        print(f"  OK: {len(tw)} twilights "
              f"({int((tw['event'] == 'sunrise').sum())} sunrise / "
              f"{int((tw['event'] == 'sunset').sum())} sunset), "
              f"{len(gpe3)} GPE3 days, {n_argos} Argos fixes, "
              f"{n_dst} DST rows, "
              f"reference_model={ref_built or 'n/a'}")
    except Exception as e:  # noqa: BLE001
        status["reason"] = f"{type(e).__name__}: {e}"
        print(f"  FAILED: {status['reason']}")
    clean_status[shark] = status

# %% [markdown]
# ## Clean-stage status log

# %%
with open(CLEAN_DIR / "clean_status.json", "w") as fh:
    json.dump(clean_status, fh, indent=2, default=str)

ok = [s for s, v in clean_status.items() if v["cleaned"]]
skipped = [s for s, v in clean_status.items() if not v["cleaned"]]
print(f"\nCleaned OK ({len(ok)}): {ok}")
print(f"Skipped/failed ({len(skipped)}): "
      + ", ".join(f"{s} ({clean_status[s]['reason']})" for s in skipped))
