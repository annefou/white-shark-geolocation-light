# 01 — Quote-with-comment (paper-rooted chains)

> Run the pre-flight checklist in `docs/forrt-form-fields.md` § Pre-flight checklist before drafting.
>
> If this is a question-rooted chain, use `01_pico.md` or `01_pcc.md` instead — see `docs/chain-decision-tree.md`.
>
> **After choosing the chain shape, delete the two step-1 alternates you aren't using.** Once you've decided this chain is paper-rooted and keep `01_quote.md`, run:
> ```bash
> rm nanopubs/drafts/01_pico.md nanopubs/drafts/01_pcc.md
> ```

**Form heading:** *"Annotate a paper quotation — Annotating a paper quotation with personal interpretation"*

## Documented form fields (from `docs/forrt-form-fields.md`)

| Field label | Field type | Notes |
|---|---|---|
| Cited DOI | text input | bare DOI starting with `10.` (not the `https://doi.org/...` form) |
| Quote whole text (less than 500 characters) | radio button (default) | Quoted Text must be ≤ 500 chars |
| Quote start/end | radio button (alternative) | start-phrase + end-phrase for longer spans |
| Quoted Text | textarea, **required** | verbatim, character-for-character, ≤ 500 chars in whole-text mode |
| Comment | textarea, **required** | interpretation / why the quote is relevant; target ≤ 500 chars |

## Field-by-field draft

### Cited DOI (text input)

Format: starts with `10.` — bare DOI, **NOT** `https://doi.org/...` form.

```
10.1038/s41597-022-01235-3
```

### Quote mode (radio button)

- [x] **Quote whole text (less than 500 characters)**
- [ ] Quote start/end *(use this if the quote exceeds 500 chars)*

### Quoted Text (textarea, required)

Verbatim from the paper PDF in `paper/`. Character-for-character. ≤ 500 chars in whole-text mode.

> Source: Abstract, page 1 of `paper/osullivan-2022.pdf`. Verified character-for-character against the PDF on 2026-06-06.

```
Here we report the full data records from 59 pop-up archival (PAT) and 20 smart position and temperature transmitting (SPOT) tags that variously recorded pressure, temperature, and light-level data, and computed depth and geolocations for 63 individuals.
```

Character count: 254 / 500.

### Comment (textarea, required)

Subtitle: *"Our interpretation or explanation of why this quotation is relevant."*

Why this quote matters and what the replication tests. Connect the paper's claim to the work this repo does. Don't repeat the quote.

> Drafted to reflect the light-extension framing. Leave to the user to finalise their own interpretation if they prefer different wording.

```
The archive's "computed geolocations" come from GPE3, a proprietary Wildlife Computers algorithm whose dominant signal is the recorded light level (with satellite SST). Because GPE3 is closed and cannot be recomputed, this study tests whether an OPEN light-level geolocation method, applied to the same CC-BY light series and judged against held-out Argos fixes, can reproduce comparable positions — extending a prior chain that found a temperature-only method ~5x less accurate with its diffusion bound saturated.
```

## Alternative candidate quote (light-specific, not selected)

If a reviewer prefers a sentence that states explicitly that the geolocations rely on light, use this verbatim passage instead. It is the GPE3 description, **page 5**, `paper/osullivan-2022.pdf` (length 566 chars — two sentences, OVER the 500-char cap, so it requires **Quote start/end** mode, not whole-text mode):

```
To further refine these geolocation estimates, the light-level data are processed through a proprietary geolocation algorithm on the Wildlife Computer portal called GPE3. The user inputs an estimate of the average swimming speed of the tagged animal, and the GPE3 process employs a discretized Hidden Markov model that uses light levels, sea surface temperatures from satellites to compare with the onboard temperature recordings, and any known locations (such as the deployment and pop-up locations) to reduce the uncertainty around each daily geolocation estimate.
```

> Recommendation: **keep the abstract sentence as the anchor.** It is atomic (one sentence), names light-level data AND the computed geolocations together, and lives in the abstract (preferred per the paper-analyst procedure). The GPE3 detail above is better carried in the Comment field than used as the primary quote.

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 01.
