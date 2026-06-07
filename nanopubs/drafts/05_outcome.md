# 05 — FORRT Replication Outcome

> Pre-flight done: verified numbers read from `results/summary_baselinefree.csv`,
> `results/aggregate_baselinefree.json`, and the per-tag `results/fused_track_*_gulfext.csv`
> (independently recomputed against `data/clean/argos_*.csv`). See `docs/verify-before-drafting.md`.

## Documented field list (from `docs/forrt-form-fields.md` § FORRT Replication Outcome)

1. Short URI suffix for outcome ID — text input, **required**
2. Plain-text label for the outcome — text input, **required**
3. Search for a FORRT replication study — search/select, **required**
4. Repository URL — text input, **required**
5. Completion date — date picker, **required**
6. Validation status — dropdown, **required** (Validated / PartiallySupported / Contradicted / Inconclusive / NotTested)
7. Confidence level — dropdown, **required** (VeryHighConfidence / HighConfidence / Moderate / LowConfidence / VeryLowConfidence)
8. Describe the overall conclusion about the original claim — textarea, **required**
9. Describe the evidence that supports your conclusion — textarea, **required**
10. Describe what limits the conclusions of the study — textarea, *optional*

## Field-by-field draft

### Short URI suffix for outcome ID (text input, required)

```
open-light-hmm-white-shark-geolocation-outcome
```

### Plain-text label for the outcome (text input, required)

```
Open light+SST+bathymetry HMM reproduces juvenile white-shark daily geolocation at 148 km median to held-out Argos (GPE3: 54 km) - physically consistent and fully open
```

### Search for a FORRT replication study (search/select, required)

URI of the Replication Study published in step 04. Pull from `nanopubs/PUBLISHED.md`.

```
_not yet published — fill from nanopubs/PUBLISHED.md step 04 before publishing this Outcome_
```

### Repository URL (text input, required)

```
https://github.com/annefou/white-shark-geolocation-light
```

### Completion date (date picker, required)

```
2026-06-07
```

### Validation status (dropdown, required)

- [ ] Validated
- [x] **PartiallySupported**
- [ ] Contradicted
- [ ] Inconclusive
- [ ] NotTested

Maps to CiTO `qualifies` in step 06. Rationale: the open method *reproduces the broad daily geolocations* (supporting that PAT light-level data yield usable tracks) but at materially lower accuracy than the proprietary GPE3 product — partial, not full, agreement.

### Confidence level (dropdown, required)

- [ ] VeryHighConfidence
- [ ] HighConfidence
- [x] **Moderate**
- [ ] LowConfidence
- [ ] VeryLowConfidence

Rationale: the evidence is methodologically strong (held-out Argos referee, baseline-free grid, multiple independent recomputations, robust median metric), but the referee sample is small (4 co-deployed PAT+SPOT tags) and the per-tag spread is large (62–206 km) — adequate evidence, partial agreement.

### Describe the overall conclusion about the original claim (textarea, required)

```
This replication PARTIALLY SUPPORTS the original claim that pop-up archival tag (PAT)
light-level data yield usable daily geolocations of juvenile white sharks. An open,
fully-reproducible Hidden Markov geolocation model — fusing an astronomical twilight
emission, a satellite-vs-onboard temperature emission, and a bathymetric depth/land
constraint, with a land-barrier movement model, on a baseline-free species-range grid —
reproduces the broad daily tracks of the four co-deployed (PAT + SPOT) sharks, reaching a
pooled median great-circle error of 148 km to the held-out SPOT Argos referee (per-tag
62–206 km). This is well below the 276 km temperature-only floor of the prior chain
(more than halving it), and within the range typically reported for open light-level
geolocation. It is larger than the proprietary Wildlife Computers GPE3 product (54 km) on
the same held-out fixes — by a factor of about 2.7 on the pooled median, but the gap is
tag-dependent rather than uniform: on the coastal-nursery tag 08_01 the open method comes
within 62 vs 30 km of GPE3, while the wide-ranging tags drive most of the difference. The chain's falsifiable prediction — that adding the light signal brings
the fitted Brownian movement sigma off its optimisation bound — is CONFIRMED on every
referee tag. The open method is thus a valid but less-accurate reproduction of the
manufacturer geolocation, with two compensating properties GPE3 lacks: it is physically
consistent (it never places the animal on land, where GPE3 does on 10 referee days, and it
respects bathymetry by construction) and it is fully open and recomputable, whereas GPE3 is
proprietary and non-recomputable.
```

### Describe the evidence that supports your conclusion (textarea, required)

```
Per-tag median great-circle error to held-out SPOT Argos fixes (quality classes 1/2/3),
open HMM vs GPE3: 07_05 = 161 vs 95 km; 08_01 = 62 vs 30 km; 08_02 = 206 vs 41 km;
08_09 = 136 vs 67 km. Pooled median-of-medians: 148 km (open HMM) vs 54 km (GPE3) vs
276 km (prior temperature-only chain). Fitted movement sigma = 0.0017–0.015 rad on all
four referee tags, all far below the 0.094 rad search upper bound (the falsifiable
sigma-off-bound prediction is confirmed). Physical consistency: 0 of ~580 open-HMM daily
positions fall on land (GPE3 places the shark on land on 10 of the four tags' days);
every open-HMM position respects bathymetry (cell seabed >= observed daily max dive depth)
by construction. The state-space grid is baseline-free — sized from the documented
NE-Pacific + Gulf-of-California species range (lon[-125,-106], lat[22,38]), so the GPE3
baseline touches neither the model fit nor the grid, and the Argos referee is held out
throughout. Referee robustness: the raw Argos data contains corrupt fixes (an out-of-region
Florida position; implied inter-fix speeds up to 1078 km/h) — exactly the outliers the
paper's Technical Validation warns users to filter — but the median metric absorbs them
(cleaning leaves both 54 km and 148 km unchanged). Fairness: the paper documents GPE3's
inputs as light levels + satellite SST + deployment/pop-up anchors + a swim-speed prior,
and NOT the SPOT Argos fixes — the same inputs the open method uses — so the comparison
against the held-out Argos is fair.
```

### Describe what limits the conclusions of the study (textarea, optional)

```
1. Small referee sample: only 4 of the 7 analysed tags carry a co-deployed SPOT Argos
   referee; the per-tag error spread is large (62–206 km).
2. The open method is ~2.7x less accurate than GPE3 — a partial, not full, accuracy
   reproduction.
3. Per-tag failure modes (which double as future-work levers): latitude error dominates
   08_09 and 08_01 (the intrinsic day-length->latitude weakness of light geolocation,
   worse near the equinox); longitude error dominates 08_02 and grows away from the
   equinox, consistent with onboard-clock drift (a degradation the paper itself flags);
   07_05 was held at the aquarium and released in Monterey Bay, north of its natural range
   (paper Usage Notes), and its baseline-free track retains 1 residual land-crossing
   segment of 133.
4. GPE3 is a proprietary black box: its exact swim-speed prior, any light-curve filtering,
   and internal tuning are unknown — only its documented input list could be matched.
```

*(Reproducibility is not a limitation here: the pipeline is self-contained — notebooks
01–04 download their own GLORYS + ETOPO and clean the depth-temperature logs natively, and
`snakemake --cores 1` reproduces the 148 km result on a fresh clone with no sibling repo.)*

## Chain consistency (all six drafts complete)

The upstream/downstream steps are drafted and aligned with this Outcome:
- **02 AIDA** — "Light-level data … yield daily geolocation estimates of juvenile white sharks …"
- **03 Claim** — type `model performance` (geolocation accuracy vs a referee).
- **04 Study** — type **Replication Study**; scope = the paper's daily PAT geolocations judged vs held-out SPOT Argos; method = the light × SST × bathymetry HMM in `notebooks/03_analysis.py` (now the canonical baseline-free pipeline).
- **06 CiTO** — intention `qualifies` (from this PartiallySupported status) → cites the paper DOI.

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 05.
