# 06 — CiTO Citation

> Pre-flight done. Citation intention derived from the Outcome's validation status
> (PartiallySupported → `qualifies`). This is the final, apex step of the chain.

**Description:** *"Declare citations between papers or other works, using Citation Typing Ontology"*

## Documented field list (from `docs/forrt-form-fields.md` § Citation with CiTO)

1. Identifier for the citing creative work — text input, **required** (the Outcome URI from step 05)
2. List citations — repeatable group, **required ≥1**
   - Citation Type — dropdown
   - DOI or other URL of the cited work — text input

## Field-by-field draft

### Identifier for the citing creative work (text input, required)

URI of the Outcome published in step 05. Pull from `nanopubs/PUBLISHED.md`.

```
_not yet published — fill from nanopubs/PUBLISHED.md step 05 before publishing this Citation_
```

### List citations (repeatable group, required ≥1)

#### Citation 1 — back to the original paper

##### Citation Type (dropdown)

From the Outcome's `PartiallySupported` status:

- [ ] confirms (Validated)
- [x] **qualifies** (PartiallySupported)
- [ ] disputes (Contradicted)

##### DOI or other URL of the cited work (text input)

```
https://doi.org/10.1038/s41597-022-01235-3
```

#### Additional citations (optional)

The upstream FORRT chain this study extends (the temperature-only replication) — cite it so
the two chains are linked. Use `qualifies` or `citesAsRelated` as the platform allows:

- Type: `qualifies` (or `citesAsRelated`) → URL: `https://w3id.org/sciencelive/np/RAnqtFUZHfmW7Dtmf3bcTQtjDAfrq5IGV4xQ8guW8L3vY`

The upstream biologging dataset (optional):

- Type: `citesAsDataSource` → URL: `https://doi.org/10.24431/rw1k6c3`

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 06. This
completes the six-step FORRT chain.

Optional next layer: **Research Software** (`drafts/07_research_software.md`) — only if the
repo produces a reusable, installable tool (it currently produces a one-off replication, so
this is likely not needed).
