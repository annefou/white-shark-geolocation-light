# Paper summary

> This is a working scratchpad for the paper-analysis phase. The output of this file feeds the Quote / AIDA / Claim drafts. It is not itself a nanopub.

**Reference paper:** A biologging database of juvenile white sharks from the northeast Pacific

**DOI:** 10.1038/s41597-022-01235-3

**Authors:** John O'Sullivan, Christopher G. Lowe, Oscar Sosa-Nishizaki, Salvador J. Jorgensen, James M. Anderson, Thomas J. Farrugia, Emiliano García-Rodríguez, Kady Lyons, Megan K. McKinzie, Erick C. Oñate-González, Kevin Weng, Connor F. White, Chuck Winkler & Kyle S. Van Houtan

**Year:** 2022 (Scientific Data 9:142; published 1 April 2022)

## Headline claim

The single sentence in the paper that this replication tests. Should be one of the paper's core empirical assertions, not a definition or framing statement.

> "Here we report the full data records from 59 pop-up archival (PAT) and 20 smart position and temperature transmitting (SPOT) tags that variously recorded pressure, temperature, and light-level data, and computed depth and geolocations for 63 individuals." (Abstract, page 1)

This is the anchor for the Quote nanopub (see `01_quote.md`). It explicitly names **light-level data** and the **computed geolocations** — the raw material and the target of this light-extension replication.

## Methodology summary

This is a **data descriptor**, not a hypothesis paper — it documents the released biologging archive, not a statistical test.

- **Study system / coverage:** Juvenile white sharks (*Carcharodon carcharias*) in the northeast Pacific (Southern California Bight and the Bahía Sebastián Vizcaíno nursery, Baja California Sur), tagged 2001–2020 by the Monterey Bay Aquarium's Juvenile White Shark Project.
- **Tag platforms / sensors:** Two biotelemetry platforms. (1) **PAT** (pop-up archival transmitting; Wildlife Computers MK10/MiniPAT/PAT2/PAT4) — every PAT model recorded **wet/dry, light level, pressure (depth), and temperature**; archive on a pre-set date, float to surface, and transmit a subset to Argos (full archive recoverable from physically retrieved tags). (2) **SPOT5** (smart position and temperature transmitting; Wildlife Computers) — transmit Argos Doppler locations when the fin-mounted tag surfaces.
- **Geolocation method (the proprietary baseline this study targets):** PAT light-level data yield two light curves per day; the onboard UTC clock compares local dawn/dusk to estimate **longitude**, and day length estimates **latitude**. These are then refined through **GPE3**, a proprietary Wildlife Computers algorithm: a discretized Hidden Markov model using **light levels, satellite sea-surface temperature compared with onboard temperature, a user swim-speed prior, and known release/pop-up anchor locations** to produce a daily geolocation track (page 4–5).
- **Sample sizes:** 79 electronic tags total (59 PAT + 20 SPOT) across **63 individual sharks**; **70 successful deployments** (n = 19 SPOT + 51 PAT) form the public archive; **26** PAT tags were physically recovered (full archive + GPE3 processing). Figure 2 reports the manuscript visualizes geolocation/temperature/depth from 58 PAT + 20 SPOT sharks; 39.7% (25/64) of tagging operations involved commercial fishery collaborations.
- **Headline numerical facts:** 79 tags / 63 individuals / 70 successful deployments / 2001–2020 deployment span; deployments concentrated on neonates and young-of-year (<1.75 m TBL); two sharks each travelled a linear distance of nearly 2,000 km.
- **Data archive (CC-BY):** Public via the US Animal Telemetry Network (ATN) Data Assembly Center and the DataONE Research Workspace member node. Standalone dataset DOI: **https://doi.org/10.24431/rw1k6c3** (CC-BY). The paper's own code (reference 23 / Code availability) covers data visualization only; the GPE3 geolocations themselves are produced by proprietary, non-recomputable manufacturer software.

## Replication design choice

Which of the three FORRT Study Types fits this replication?

- [ ] **Reproduction Study** — direct reproduction: same methodology, same tools.
- [x] **Replication Study** — replication with different methodology or conditions.
- [ ] **Reproduction/Replication Study** — both.

Brief justification for the choice (one paragraph).

GPE3 is a closed, proprietary Wildlife Computers algorithm; its geolocations cannot be recomputed, so a same-tools Reproduction is impossible by construction. This study is therefore a **Replication Study**: it re-derives daily positions with an **open light-level geolocation method** applied to the light series already in the CC-BY archive, judged against the held-out Argos SPOT fixes as referee (GPE3 remains a comparison baseline, never ground truth). It **extends** the prior FORRT chain (CiTO `extends`), which established a temperature-at-depth-only floor (~276 km median error to Argos, with the fitted Brownian σ railing at its bound). The only intended moving part relative to that prior Study is the added light emission — keeping the same tag set / referee logic and HEALPix NESTED level 9 grid so the test of the falsifiable prediction (σ comes off its bound, median error drops) is clean.

## Notes for downstream drafts

- **Quote discrepancy fixed:** the prior chain's quote read "smart position and **transmitting** (SPOT)"; the PDF reads "smart position and **temperature transmitting** (SPOT)". The verbatim quote in `01_quote.md` uses the PDF wording.
- **Light-specific alternative quote** (page 5, GPE3 paragraph) is recorded in `01_quote.md` as an alternative candidate — it states explicitly that GPE3 uses light levels + SST. Recommendation: keep the abstract sentence as the anchor (it names light-level data AND the computed geolocations in one atomic sentence and sits in the abstract); cite the GPE3 detail in the Comment field instead.
- **Tag-set caveat for the new Study:** the prior temperature chain dropped tag 02_01 (no external temperature sensor). Because every PAT records light, 02_01 may re-enter the light-based tag set — re-derive the in-scope list from the archive metadata, do not copy the prior list blindly.
- Dataset DOI for CITATION.cff and the Study Methodology field: 10.24431/rw1k6c3 (the upstream telemetry archive, CC-BY).
