# S05 Experiment Integrity Audit

**Date:** 2026-11-23

**Auditor:** fresh GPT-6.1-Sol ultra reviewer, same-family provisional

**Overall verdict:** **WARN**
**Integrity status:** `warn`

Deterministic evidence checks passed. No fabricated GT, score self-normalization, phantom result, duplicate-action inflation, or GPU0 use was found. The warning is about scope and claim strength, not a failed integrity check.

## Checks

### A. Ground-truth provenance: PASS

S05-A freezes tracker files before reading DanceTrack GT and runs the pinned official TrackEval HOTA/CLEAR/Identity implementation. S05-B generates M/G/V candidates without GT; a separate offline stage uses dataset GT with the explicitly labelled IoU-0.5 stable-owner proxy. No model-derived reference was presented as ground truth.

### B. Score normalization: PASS

Official ratio metrics are projected to 0–100 points without prediction-dependent normalization. S05-B uses integer `g`, `h`, and `g-h`; the gate is a raw component count.

### C. Result existence and exact values: PASS

The reviewer independently read 152 JSON files, 335,407 tracker rows, all 25,328 candidate trace rows, and all 242 component records. It independently reran official TrackEval and exactly reproduced all 27 condition × aggregate/per-sequence rows.

- G: HOTA 68.90273986139066, AssA 52.82377438598485, IDSW 270.
- L: HOTA 68.81504013879633, AssA 52.77110310801603, IDSW 250.
- M: HOTA 69.51869387798489, AssA 53.56446597941373, IDSW 235.
- M−G: +0.6159540165942303 HOTA, +0.7406915934288847 AssA, IDSW −35; IDF1 is essentially unchanged.
- S05-B: 242 components, 126 fully labelable, 41 net-profitable component-frames, 40 unique frames, 17 videos, and 69 involved video-track IDs.

The 69 IDs are all tracks involved in profitable components, not recovered identities. Exactly 43 unique video-track IDs gain a proxy-correct pair.

### D. Executed metrics and engineering gates: PASS with limits

All 24 instrumented M trajectory hashes match authoritative S01 outputs. Candidate assignments and reconstructed union components are one-to-one. Selecting M reconstructs factual first-round assignment on every trace. The full 0027 replay reproduced both tracker and trace SHA256 exactly; deterministic second replay was checked for one video, not all 24.

The first S05-A attempt failed before tracking because `PYTHONPATH` lacked `src`; attempt 2 records the environment-only correction. A directory glob also attempted `coverage_attempt1/source_snapshot/` and retained a launcher error without scientific output. The first pytest invocation hit an unrelated user Dash/Jinja2 plugin; the clean plugin-disabled run passed.

### E. Scope and chronology: WARN

S05-A covers eight previously used development videos, not an independent test. M beats G on HOTA in 5/8 videos and loses on 3/8; the worst video difference is −7.138784661157743. S05-B covers reused fit24 states and its owner labels are diagnostic proxies. Appearance pretraining overlap remains unknown.

All checked locks use protocol SHA256 `b35aef525d6c6b43564d036ea090dd9edf6f6c868c8f2b45c29a6cd18d20cbc3`, and local times place the protocol before execution locks. This supports local pre-execution freezing, not external preregistration.

### F. Evaluation classification: PASS

- S05-A: `real_gt`, official full-sequence TrackEval.
- S05-B: `real_gt` source with a derived offline owner-proxy diagnostic; it is not official HOTA/IDSW and not a complete true-error census.

## Resolved report correction

The first report exporter incorrectly aliased “correct alternative exists” to “net-profitable.” They are now separate:

- Any alternative with `g>0`: 46 components.
- Any unique alternative with `g-h>0`: 41 components, which is the frozen resource gate.
- Five components have `g=1,h=1`; they add a correct proxy pair while causing equal collateral harm.

This was a report-only correction after results; immutable raw records and the 41-component gate did not change.

## Claim impact

- **M development8 stability:** supported with qualifiers. Aggregate improvement is exact and reproducible, but not uniform or independently generalized.
- **S05 coverage gate:** supported for the frozen owner-proxy resource gate. It permits consideration of a separately frozen S06; it does not show learnability, closed-loop gain, novelty, or automatic permission to start S06.

## Independent evidence

- `/media/imc/data/mot-data/audits/s05_integrity_20261123/independent_audit.json`
- `/media/imc/data/mot-data/audits/s05_integrity_20261123/independent_audit.py`
- `/media/imc/data/mot-data/audits/s05_integrity_20261123/official_reproduction1/reproduction_check.json`
- `/media/imc/data/mot-data/audits/s05_integrity_20261123/official_reproduction1/full_raw_result.json`

This audit is same-family provisional assurance, not a cross-family independent review.
