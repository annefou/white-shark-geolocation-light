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
# # 01 — Data download (light-level geolocation)
#
# Fetches all inputs for the **open light-level geolocation** extension of the
# prior temperature-at-depth replication of O'Sullivan et al. 2022 (*Scientific
# Data* 9:142, [10.1038/s41597-022-01235-3](https://doi.org/10.1038/s41597-022-01235-3)):
# **"A biologging database of juvenile white sharks from the northeast Pacific."**
#
# This study adds an **open light / solar-elevation emission** to the same
# pangeo-fish HMM on the same HEALPix NESTED level-9 grid, judged against the
# same held-out Argos referee, to test the falsifiable prediction that the
# fitted Brownian σ comes off its bound and median error drops.
#
# Three sources are pulled here:
#
# 1. The CC-BY **biologging archive** (dataset DOI
#    [10.24431/rw1k6c3](https://doi.org/10.24431/rw1k6c3)), hosted on the ATN DAC /
#    Research Workspace (a DataONE member node) — light, depth-temperature (DST),
#    GPE3 baseline and co-deployed Argos referee fixes (no credentials).
# 2. NOAA NCEI **ETOPO 2022** 15-arc-second bathymetry (no credentials).
# 3. **GLORYS12V1** ocean reanalysis (`thetao`+`zos`) via Copernicus Marine, over
#    the juvenile-white-shark species-range box, for the temperature emission —
#    this **does** need a Copernicus Marine credential (Step 6).
#
# The light emission matches an **astronomical** solar-elevation model; the
# temperature emission additionally matches the GLORYS ocean field, so the repo is
# now fully self-contained (no sibling repository).
#
# Each recovered-PAT package holds the raw light data this study consumes:
#
# - `out-Archive.csv` — a dense raw **`Light Level`** time series (same cadence
#   as Depth + Temperature), plus the on-tag `One Minute`/`Smoothed` light.
# - `out-LightLoc.csv` — the on-tag detected **twilight events** (Dawn/Dusk)
#   with a tag-clock timestamp and a UTC `Time Offset`.
# - `*-GPE3.csv` — the proprietary baseline track (light + SST; not recomputable).
#
# Co-deployed SPOT packages hold the **Argos fixes** used as the accuracy referee.
#
# All downloads are **cached**: re-running skips files already on disk.

# %%
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import requests

# %%
RAW_DIR = Path("../data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
TAG_DIR = RAW_DIR / "tag_packages"
TAG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DOI = "10.24431/rw1k6c3"
CN_SOLR = "https://cn.dataone.org/cn/v2/query/solr/"
CN_RESOLVE = "https://cn.dataone.org/cn/v2/resolve/"

# %% [markdown]
# ## The seven recovered-PAT tags
#
# Same recovered-PAT deployments as the prior temperature chain. Four have a
# co-deployed SPOT tag that supplies the Argos accuracy referee; three are
# PAT-only. **Every PAT — including `02_01`, which the temperature chain dropped
# for lacking an external water-temperature sensor — records light**, so all
# seven enter the light download (the in-scope tag list is re-derived in
# notebook 02 from what each archive actually contains).
#
# Registry: `shark_id -> (PAT_DEPLOY_ID, SPOT_DEPLOY_ID or None, has_referee)`.

# %%
TAGS = {
    "07_05": ("07_05-66885", "07_05-77272", True),
    "08_01": ("08_01-40561", "08_01-77274", True),
    "08_02": ("08_02-55716", "08_02-77273", True),
    "08_09": ("08_09-83066", "08_09-83076", True),
    "02_01": ("02_01-18616", None, False),
    "06_10": ("06_10-40564", None, False),
    "07_01": ("07_01-64272", None, False),
}

# %% [markdown]
# ## Step 1 — resolve the DOI to a DataONE data package

# %%
def solr_query(q: str, fl: str, rows: int = 500) -> list[dict]:
    """Query the DataONE CN solr index and return the matching docs."""
    resp = requests.get(
        CN_SOLR,
        params={"q": q, "fl": fl, "rows": rows, "wt": "json"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]["docs"]


doi_token = DATASET_DOI.replace("/", "_")
meta_docs = solr_query(
    q=f"id:*{doi_token.split('_')[-1]}*",
    fl="identifier,formatId,title,resourceMap,dateUploaded",
)
meta_docs = [d for d in meta_docs if "isotc211" in d.get("formatId", "")]
meta_docs.sort(key=lambda d: d.get("dateUploaded", ""))
metadata = meta_docs[-1]
resource_map = metadata["resourceMap"][0]

print(f"Dataset DOI:    {DATASET_DOI}")
print(f"Title:          {metadata['title']}")
print(f"Resource map:   {resource_map}")

# %% [markdown]
# ## Step 2 — list every data object in the package

# %%
objs = solr_query(
    q=f'resourceMap:"{resource_map}"',
    fl="identifier,fileName,formatId,size",
)
manifest = pd.DataFrame(
    [
        {
            "identifier": o["identifier"],
            "fileName": o.get("fileName"),
            "formatId": o.get("formatId"),
            "size": o.get("size"),
            "dataUrl": CN_RESOLVE + o["identifier"],
        }
        for o in objs
        if "isotc211" not in o.get("formatId", "")
    ]
).sort_values("fileName", na_position="first").reset_index(drop=True)
manifest.to_csv(RAW_DIR / "package_manifest.csv", index=False)

zip_pkgs = manifest[manifest["formatId"] == "application/zip"]
print(f"Package objects: {len(manifest)}  (ZIP data packages: {len(zip_pkgs)})")

# %% [markdown]
# ## Step 3 — download the deployment metadata table

# %%
def resolve_object(pid: str) -> list[str]:
    """Return member-node download URLs for a DataONE PID (no redirect-follow)."""
    r = requests.get(CN_RESOLVE + pid, timeout=120, allow_redirects=False)
    urls = re.findall(r"<url>(.*?)</url>", r.text)
    return urls or [f"https://cn.dataone.org/cn/v2/object/{pid}"]


def download_object(pid: str, out_path: Path) -> Path:
    """Stream a DataONE object to disk if not already cached, trying MN mirrors."""
    if out_path.exists():
        print(f"  cached: {out_path.name}")
        return out_path
    last_err = None
    for u in resolve_object(pid):
        try:
            resp = requests.get(u, stream=True, timeout=900)
            resp.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
            print(f"  downloaded: {out_path.name} "
                  f"({out_path.stat().st_size} bytes)")
            return out_path
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"all mirrors failed for {out_path.name}: {last_err}")


WANTED_CSVS = {
    "JWS_metadata.csv",
    "JWS_metadata_README.csv",
    "PAT_programming_table.csv",
    "PAT_programming_table_README.csv",
}
print("Downloading metadata tables:")
for _, r in manifest[manifest["fileName"].isin(WANTED_CSVS)].iterrows():
    download_object(r["identifier"], RAW_DIR / r["fileName"])

# %% [markdown]
# ## Step 4 — download every tag's ZIP packages (recovered PAT + co-deployed SPOT)
#
# Each recovered-PAT package holds the raw light series (`out-Archive.csv`), the
# on-tag twilight events (`out-LightLoc.csv`) and the published `*-GPE3.csv`
# baseline track. Each co-deployed SPOT package holds the Argos referee fixes.

# %%
def download_zip(zip_name: str) -> Path:
    row = manifest[manifest["fileName"] == zip_name]
    if row.empty:
        raise FileNotFoundError(f"{zip_name} not in package manifest")
    pid = row.iloc[0]["identifier"]
    return download_object(pid, TAG_DIR / zip_name)


print("Downloading tag packages:")
for shark, (pat_dep, spot_dep, has_referee) in TAGS.items():
    print(f"  {shark}:")
    download_zip(f"{pat_dep}.zip")
    if spot_dep is not None:
        download_zip(f"{spot_dep}.zip")

# %% [markdown]
# ## Step 5 — regional bathymetry (NOAA NCEI ETOPO 2022 v1 15-arc-second)
#
# The fused HMM in notebook 03 adds a **land mask** and a **bathymetric depth
# floor** so tracks cannot cross land or run shallower than the tag's dive record.
# We pull a regional subset of NOAA NCEI **ETOPO 2022 v1 15-arc-second** (~450 m)
# global relief covering the NE-Pacific shark domain
# (lon −130…−105, lat 18…40 — the union of all tags' release/pop-up + GPE3
# envelopes with a pad). The source is the CoastWatch ERDDAP griddap dataset
# [`ETOPO_2022_v1_15s`](https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.html)
# (variable `z`, EGM2008 height, lon −180…180), accessed over OPeNDAP — **no
# credentials**. The subset is saved as NetCDF (never `.npz`, per DOMAIN.md).

# %%
BATHY_NC = RAW_DIR / "etopo2022_15s_nepac.nc"
BATHY_SOURCE = (
    "NOAA NCEI ETOPO 2022 v1 15 arc-second global relief model via CoastWatch "
    "ERDDAP griddap dataset ETOPO_2022_v1_15s")
BATHY_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s"
BATHY_DOMAIN = {"lon": (-130.0, -105.0), "lat": (18.0, 40.0)}

if BATHY_NC.exists():
    print(f"  cached: {BATHY_NC.name} ({BATHY_NC.stat().st_size} bytes)")
else:
    # Direct ERDDAP griddap NetCDF subset over plain HTTPS — NOT OPeNDAP. The
    # netCDF4 build on CI runners often lacks DAP support, so xr.open_dataset() on
    # the OPeNDAP endpoint fails with "NetCDF: I/O failure". ERDDAP serves the
    # server-side lat/lon subset directly as a downloadable .nc file.
    import requests  # local import: only this step needs it here

    lat0, lat1 = BATHY_DOMAIN["lat"]
    lon0, lon1 = BATHY_DOMAIN["lon"]
    query = f"z%5B({lat0}):1:({lat1})%5D%5B({lon0}):1:({lon1})%5D"
    nc_url = f"{BATHY_URL}.nc?{query}"
    print(f"Downloading ETOPO 2022 15s subset {BATHY_DOMAIN} (ERDDAP .nc)...")
    resp = requests.get(nc_url, timeout=600)
    resp.raise_for_status()
    BATHY_NC.write_bytes(resp.content)
    print(f"  downloaded: {BATHY_NC.name} ({BATHY_NC.stat().st_size} bytes); "
          f"source: {BATHY_SOURCE}")

# %% [markdown]
# ## Step 6 — GLORYS12V1 ocean reanalysis for the temperature emission
#
# The fused HMM in notebook 03 multiplies the light emission by a **temperature**
# emission that matches the tag's external-temperature dive record against
# **GLORYS12V1** (`thetao` 4-D temperature + `zos` sea-surface height). We download
# that ocean field here so the repo is self-contained — no sibling repo.
#
# **Baseline-free box (DOMAIN.md).** The GLORYS subset spans the juvenile-white-
# shark **species range**, `lon[-125, -106]`, `lat[22, 38]` (NE Pacific + Gulf of
# California), **not** a box derived from the GPE3/Argos tracks. Using the species
# range rather than the per-tag track envelope is what makes the analysis grid
# baseline-free: the state space is fixed by ecology, not by the baseline we judge
# against.
#
# One `thetao+zos` file is downloaded per **temperature tag** over that tag's
# deployment window (`DATE_START`..`PAT_END` from `JWS_metadata.csv`, +1 day so the
# daily time-bin bounds cover the pop-up day), plus one box-only **static**
# `deptho+mask` file shared by all tags. Depths `0..900 m`, daily, NESTED-agnostic
# (lat/lon grid; the HEALPix regrid happens in 03).
#
# This step needs a **Copernicus Marine** account. Credentials are read from
# `~/.copernicusmarine/.copernicusmarine-credentials`; in CI they are written from
# the base64 secret `COPERNICUS_CREDENTIALS_BASE64` (see README / DOMAIN.md). The
# download is **idempotent**: a tag whose `reference_model_<tag>_gulfext.nc` already
# exists, or whose raw GLORYS file is already on disk, is skipped.

# %%
GLORYS_DIR = RAW_DIR / "glorys"
GLORYS_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR = Path("../data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

GLORYS_THETAO = "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311"
GLORYS_STATIC = "cmems_mod_glo_phy_my_0.083deg_static_202311"
# Baseline-free species-range box (NOT GPE3/Argos derived) — see DOMAIN.md.
GLORYS_BOX = {"longitude": (-125.0, -106.0), "latitude": (22.0, 38.0)}
GLORYS_MAX_DEPTH = 900.0
# Tags that carry an external water-temperature sensor (the temperature factor).
# 02_01 is light-only (PAT2, no external thermistor), so it gets no GLORYS.
TEMPERATURE_TAGS = ["07_05", "08_01", "08_02", "08_09", "06_10", "07_01"]

static_nc = GLORYS_DIR / "glorys_static_gulfext.nc"


def glorys_window(shark: str) -> tuple[str, str]:
    """Deployment window (release..pop-up +1 day) as ISO strings from JWS metadata."""
    row = sharks_meta[sharks_meta["SHARK_ID"] == shark].iloc[0]
    t0 = pd.to_datetime(row["DATE_START"])
    t1 = pd.to_datetime(row["PAT_END"]) + pd.Timedelta(days=1)
    return t0.strftime("%Y-%m-%dT00:00:00"), t1.strftime("%Y-%m-%dT00:00:00")


# JWS metadata (already downloaded above) supplies each tag's deployment window.
sharks_meta = pd.read_csv(
    RAW_DIR / "JWS_metadata.csv", dtype=str, encoding="latin-1")
sharks_meta = sharks_meta.apply(
    lambda c: c.str.strip() if c.dtype == "object" else c)

# Any temperature tag still missing BOTH its reference model and its raw GLORYS
# file triggers the (slow) copernicusmarine download; otherwise the whole step is a
# no-op. Import copernicusmarine lazily so a credential-less run that only needs the
# already-cached reference models does not fail at import.
_need_download = [
    t for t in TEMPERATURE_TAGS
    if not (CLEAN_DIR / f"reference_model_{t}_gulfext.nc").exists()
    and not (GLORYS_DIR / f"glorys_thetao_{t}_gulfext.nc").exists()
]
_need_static = (not static_nc.exists()) and bool(_need_download)

if _need_download or _need_static:
    import copernicusmarine as cm  # lazy: only when an actual fetch is required

    if _need_static:
        print(f"Downloading GLORYS static deptho+mask {GLORYS_BOX} ...")
        cm.subset(
            dataset_id=GLORYS_STATIC, variables=["deptho", "mask"],
            minimum_longitude=GLORYS_BOX["longitude"][0],
            maximum_longitude=GLORYS_BOX["longitude"][1],
            minimum_latitude=GLORYS_BOX["latitude"][0],
            maximum_latitude=GLORYS_BOX["latitude"][1],
            output_filename=static_nc.name, output_directory=str(GLORYS_DIR),
            coordinates_selection_method="outside",
            overwrite=True, disable_progress_bar=True)
        print(f"  wrote {static_nc.name} ({static_nc.stat().st_size} bytes)")

    for shark in _need_download:
        t0, t1 = glorys_window(shark)
        out = GLORYS_DIR / f"glorys_thetao_{shark}_gulfext.nc"
        print(f"[{shark}] downloading GLORYS thetao+zos {GLORYS_BOX} {t0}..{t1} ...")
        cm.subset(
            dataset_id=GLORYS_THETAO, variables=["thetao", "zos"],
            minimum_longitude=GLORYS_BOX["longitude"][0],
            maximum_longitude=GLORYS_BOX["longitude"][1],
            minimum_latitude=GLORYS_BOX["latitude"][0],
            maximum_latitude=GLORYS_BOX["latitude"][1],
            start_datetime=t0, end_datetime=t1,
            minimum_depth=0.0, maximum_depth=GLORYS_MAX_DEPTH,
            output_filename=out.name, output_directory=str(GLORYS_DIR),
            coordinates_selection_method="outside",
            overwrite=True, disable_progress_bar=True)
        print(f"  wrote {out.name} ({out.stat().st_size} bytes)")
else:
    print("GLORYS: all temperature tags already have a reference model or raw "
          "subset on disk — skipping download.")

# %% [markdown]
# ## Step 7 — source log

# %%
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


checksums = {
    p.name: sha256(p)
    for p in sorted(RAW_DIR.glob("*.csv"))
    if p.name != "package_manifest.csv"
}

sources = {
    "biologging_archive": {
        "name": "A biologging database of juvenile white sharks from the "
                "Northeast Pacific, 2001-2020",
        "doi": DATASET_DOI,
        "url": f"https://doi.org/{DATASET_DOI}",
        "repository": "ATN DAC / Research Workspace (DataONE member node)",
        "license": "CC-BY-4.0",
        "resource_map": resource_map,
        "checksums_sha256": checksums,
        "tags": {
            shark: {
                "pat_zip": f"{pat_dep}.zip",
                "spot_zip": (f"{spot_dep}.zip" if spot_dep else None),
                "has_referee": has_referee,
            }
            for shark, (pat_dep, spot_dep, has_referee) in TAGS.items()
        },
    },
    "emission_model": {
        "kind": "astronomical solar-elevation / twilight",
        "note": "No ocean reanalysis (GLORYS) or Copernicus credentials are "
                "needed: the light emission matches an astronomical "
                "sunrise/sunset model computed per HEALPix cell with astropy.",
    },
    "bathymetry": {
        "name": "ETOPO 2022 v1 15 arc-second global relief model",
        "source": BATHY_SOURCE,
        "url": BATHY_URL + ".html",
        "variable": "z (EGM2008 height, metres)",
        "resolution": "15 arc-second (~450 m)",
        "domain": BATHY_DOMAIN,
        "file": BATHY_NC.name,
        "license": "Public domain (US Government work)",
        "note": "Regional NE-Pacific subset, fetched over OPeNDAP (no auth). "
                "Used in notebook 03 for the land mask + bathymetric depth floor.",
    },
    "glorys_temperature": {
        "name": "GLORYS12V1 global ocean physics reanalysis",
        "thetao_dataset_id": GLORYS_THETAO,
        "static_dataset_id": GLORYS_STATIC,
        "variables": ["thetao", "zos", "deptho", "mask"],
        "box_baseline_free": GLORYS_BOX,
        "max_depth_m": GLORYS_MAX_DEPTH,
        "source": "Copernicus Marine Service (E.U. Copernicus Marine "
                  "Environment Monitoring Service)",
        "credential": "~/.copernicusmarine/.copernicusmarine-credentials "
                      "(CI: base64 secret COPERNICUS_CREDENTIALS_BASE64)",
        "temperature_tags": TEMPERATURE_TAGS,
        "note": "Downloaded over the juvenile-white-shark SPECIES-RANGE box "
                "lon[-125,-106] lat[22,38] (NOT GPE3/Argos derived), one "
                "thetao+zos file per temperature tag over its deployment "
                "window, plus a shared static deptho+mask. Notebook 02 builds "
                "reference_model_<tag>_gulfext.nc from these; notebook 03 reads "
                "them. 02_01 is light-only and gets no GLORYS.",
    },
    "accessed_on": "2026-06-07",
}
with open(RAW_DIR / "sources.json", "w") as fh:
    json.dump(sources, fh, indent=2, default=list)

print(f"Logged source provenance to {RAW_DIR / 'sources.json'}")
