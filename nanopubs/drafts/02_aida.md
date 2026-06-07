# 02 — AIDA Sentence

> Pre-flight done. AIDA = one Atomic, Independent, Declarative, Absolute empirical
> claim — the paper's geolocation claim that this replication tests (NOT the
> replication's result; that lives in the Outcome).

**Form heading:** *"AIDA Sentence — Make structured scientific claims following the AIDA model"*

## Documented field list (from `docs/forrt-form-fields.md` § AIDA sentence)

1. AIDA sentence (ending with a full stop) — textarea, **required**
2. Select related topics/tags — dropdown, *optional*
3. Relates to this nanopublication — text input, **required** (the Quote URI from step 01)
4. Supported by datasets — repeatable group, *optional*
5. Supported by other publications — repeatable group, *optional*

## Field-by-field draft

### AIDA sentence (textarea, required)

One empirical finding; no "and" linking two findings; ends with a full stop.

```
Pop-up archival tag light-level data yield accurate daily geolocations of juvenile white sharks in the northeast Pacific.
```

### Select related topics/tags (dropdown, optional)

Pick from the platform's predefined vocabulary. Suggested labels if available:

```
geolocation; biologging; marine animal tracking
```

*(skip if none of these are in the dropdown — optional field)*

### Relates to this nanopublication (text input, required)

URI of the Quote-with-comment published in step 01. Pull from `nanopubs/PUBLISHED.md`.

```
_not yet published — fill from nanopubs/PUBLISHED.md step 01 before publishing this AIDA_
```

### Supported by datasets (repeatable group, optional)

The CC-BY biologging archive that grounds the claim:

- DOI 1: `https://doi.org/10.24431/rw1k6c3`

### Supported by other publications (optional)

*(skip — the source paper is already cited via the Quote in step 01; per the known platform bug note, leaving this empty also avoids the datasets+publications publish failure)*

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 02.
