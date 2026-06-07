# 03 — FORRT Claim

> Pre-flight done. Claim type chosen per `docs/claim-type-vocabulary.md`.

**Form heading:** *"FORRT Claim — Declare an original claim according to FORRT, linking it to an AIDA sentence with a specific FORRT type."*

## Documented field list (from `docs/forrt-form-fields.md` § FORRT Claim)

1. Short URI suffix as claim ID — text input, **required**
2. Label of the claim — text input, **required**
3. Search for an AIDA sentence — search/select, **required** (the AIDA URI from step 02)
4. Type of FORRT claim — dropdown, **required**
5. Source URI — text input, *optional*

## Field-by-field draft

### Short URI suffix as claim ID (text input, required)

Slug, kebab-case.

```
pat-light-geolocation-juvenile-white-shark
```

### Label of the claim (text input, required)

A descriptive title (not a sentence).

```
PAT light-level geolocation of juvenile white sharks
```

### Search for an AIDA sentence (search/select, required)

URI of the AIDA published in step 02. Pull from `nanopubs/PUBLISHED.md`. (If the AIDA
was published via Nanodash — `w3id.org/np/...` — the search may not find it; paste the
URI manually.)

```
_not yet published — fill from nanopubs/PUBLISHED.md step 02 before publishing this Claim_
```

### Type of FORRT claim (dropdown, required)

See `docs/claim-type-vocabulary.md`.

- [ ] computational performance
- [ ] scalability
- [ ] data quality
- [ ] data governance
- [ ] descriptive pattern
- [x] **model performance**
- [ ] statistical significance

Rationale: the claim concerns the **accuracy of a geolocation method** (daily positions
recovered from light, judged against a held-out positional referee) — an accuracy/eval
metric, i.e. `model performance`. (A `data quality` framing — "light data yield usable
positions through processing" — is a defensible alternative; `model performance` is chosen
because the replication is judged on positional error.)

### Source URI (text input, optional)

Full URL form (NOT bare DOI).

```
https://doi.org/10.1038/s41597-022-01235-3
```

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 03.
