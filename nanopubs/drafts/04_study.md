# 04 — FORRT Replication Study

> Pre-flight done. Method field verified against `notebooks/03_analysis.py` (the
> baseline-free fused HMM) — not from memory. Scope = WHAT is reproduced; Method =
> HOW; results stay in the Outcome. See `docs/pico-study-outcome-levels.md`.

**Form heading:** *"FORRT Replication Study"*

## Documented field list (from `docs/forrt-form-fields.md` § FORRT Replication Study)

1. Short URI suffix for study ID — text input, **required**
2. Label/name of replication study — text input, **required**
3. Study type — dropdown, **required**
4. Search for a FORRT claim — search/select, **required** (the Claim URI from step 03)
5. Describe what part of the claim is reproduced/replicated (scope) — textarea, **required**
6. Describe how the claim is reproduced/replicated (method) — textarea, **required**
7. Describe any deviations from original methodology — textarea, *optional*
8. Search keywords (Wikidata) — multi-select, *optional*
9. Search discipline (Wikidata) — search, *optional*

## Field-by-field draft

### Short URI suffix for study ID (text input, required)

```
open-light-hmm-white-shark-geolocation-study
```

### Label/name of replication study (text input, required)

```
Open light+SST+bathymetry HMM replication of juvenile white-shark geolocation
```

### Study type (dropdown, required)

- [ ] Reproduction Study — same methodology, same tools.
- [x] **Replication Study** — different methodology / conditions.
- [ ] Reproduction/Replication Study — both.

Rationale: GPE3 is proprietary and non-recomputable, so a same-tools reproduction is
impossible; this re-derives the geolocations with an independent open method.

### Search for a FORRT claim (search/select, required)

URI of the Claim published in step 03. Pull from `nanopubs/PUBLISHED.md`.

```
_not yet published — fill from nanopubs/PUBLISHED.md step 03 before publishing this Study_
```

### Describe what part of the claim is reproduced/replicated — SCOPE (textarea, required)

```
The paper's computed daily PAT geolocations of juvenile white sharks. Specifically, whether
the daily positions GPE3 produced from the light series can be recovered by an independent,
fully open method, evaluated on the four sharks that carry a co-deployed SPOT tag, so that
the held-out Argos satellite fixes serve as an independent positional referee. In scope:
the daily geolocation track and its positional accuracy. Out of scope: the depth and
temperature time series themselves, the tag hardware, and any non-geolocation data product.
GPE3 is treated as a comparison baseline, never as ground truth; the Argos fixes are the
referee and are never used to fit the model.
```

### Describe how the claim is reproduced/replicated — METHOD (textarea, required)

```
An open hidden Markov geolocation model (pangeo-fish) on a HEALPix (NESTED, level 9) state
space spanning the documented NE-Pacific + Gulf-of-California species range (lon[-125,-106],
lat[22,38]) — a baseline-free grid fixed by ecology, not by the GPE3/Argos tracks. The daily
emission is the product of three independent factors: (1) an astronomical TWILIGHT emission
that scores each cell by how well its modelled sunrise/sunset time (astropy) matches the
tag's detected dawn/dusk times; (2) a TEMPERATURE emission matching the tag's external
depth-temperature record against the GLORYS12V1 thetao field; and (3) a BATHYMETRY factor
(GEBCO/ETOPO) that masks land and softly requires the seabed to be at least as deep as the
animal's observed dives. Movement uses a land-barrier Gaussian transition (a custom kernel
that forbids probability flow across land), so the model cannot move the animal over land.
The Brownian diffusion sigma is fit per tag by maximum likelihood; daily positions are taken
by maximum-a-posteriori decode and joined into a continuous, land-respecting track via a
constrained shortest-path decode. Release and pop-up coordinates are the only spatial
anchors. Each tag's track is compared to the held-out SPOT Argos fixes (quality classes 1/2/3)
by median great-circle distance, with GPE3 scored identically for reference.
```

### Describe any deviations from original methodology (textarea, optional)

```
- Open pangeo-fish HMM in place of the proprietary, non-recomputable GPE3 algorithm (same
  input ingredients per the paper: light + satellite SST + release/pop-up anchors + a
  movement prior), plus a bathymetry/land constraint GPE3 does not use.
- Only the twilight TIMES are taken from the light series (anti-circularity): the tag's own
  seed latitude/longitude and SST columns are not used.
- Baseline-free species-range grid (the GPE3 track is used for neither the fit nor the grid).
- Argos referee cleaned of physically impossible fixes (out-of-region positions; implied
  speeds far above a shark's), as the paper's Technical Validation instructs; the median
  metric is in any case robust to them.
```

### Search keywords (Wikidata) (multi-select, optional)

Labels (not QIDs):

- Carcharodon carcharias
- geolocation
- hidden Markov model
- biologging

### Search discipline (Wikidata) (search, optional)

- marine biology

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 04.
