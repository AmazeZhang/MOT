# S05 visual decision coverage

Status: **COMPLETED / CONTINUE TO SEPARATELY FROZEN S06 SEED 13 ONLY**. This is a reused-development stability check plus a proxy-labelled resource census, not independent validation or a learned-method result.

## One-page result

S05-A ran G and L on the eight reused development videos and reused M only after source/input/output checks. Official combined TrackEval is:

| condition | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| G | 68.90273986139066 | 52.82377438598485 | 67.32730333178326 | 270 |
| L | 68.81504013879633 | 52.77110310801603 | 66.97274144322336 | 250 |
| M | 69.51869387798489 | 53.56446597941373 | 67.32726439190517 | 235 |

M minus G is `+0.61595401659423` HOTA and `+0.74069159342888` AssA with 35 fewer ID switches. M wins HOTA on 5/8 videos, loses on 3/8, and the worst delta is `-7.1388` on `dancetrack0074`; the unweighted per-video HOTA/AssA means are negative. The pooled result is not uniform improvement.

S05-B inspected every one of 25,328 frames in M's reused fit24 causal trajectories. A GT-free same-state M/G/V generator found 242 differing union components in 239 frames. Offline stable-owner proxy labels cover 126 components; 41 have a legal deduplicated G/V action with `g-h>0`, across 40 unique frames and 17 videos. The frozen 40-component/4-video resource gate passes by one component. Forty-six components have any `g>0`, but five also destroy an M-correct pair. The reported 69 involved video-track IDs are not recovered identities; 43 unique video-track IDs gain a proxy-correct pair.

The fresh same-family provisional audit independently reproduced all 27 TrackEval rows and all 242 components. Verdict is `WARN` for reused data, proxy labels, unknown ReID-pretraining overlap and limited scope—not for a deterministic mismatch. Three fully labelable owner-collision components exist; none is profitable.

## Attempts and deviations

- S05-A attempt 1 failed before tracking because `PYTHONPATH` omitted the local `src` package. Attempt 2 changed only the environment path and completed under a new immutable run ID.
- The S05-B launcher glob also addressed a `source_snapshot` directory, causing one retained launcher error with no scientific output. All 24 intended sequence runs completed.
- A full deterministic replay of `dancetrack0027` reproduced both candidate-trace and tracker-output SHA256 exactly.
- All S05 tracking, evaluation, analysis and audit tasks used `CUDA_VISIBLE_DEVICES=''`; physical GPU0 and every other GPU were unused.

## Interpretation and next action

The supported claim is only that the frozen candidate resource gate passes and M remains a useful pooled development baseline. S05 does not show that the proxy action is learnable, improves natural closed-loop metrics, is novel, or generalizes.

The result-to-claim route is `implement_and_run_seed13_only`. The separately frozen protocol is `manifests/protocols/s06_component_selector_seed13_20261123.yaml` (SHA256 `290a95e3e36e96dbc62cd20e7661a6b1c80b1627a2849586cb37c412f2afdf51`). Seeds 37/73 remain blocked unless seed 13 passes every frozen gate clause. CAMELTrack is Apache-2.0 and provides official DanceTrack assets, but its TrackLab pipeline, detections, state files and training path are not direct same-input drop-ins; any later comparison must distinguish an adapted control from its official full system.

Full-precision aggregate and per-sequence data are in `aggregate_metrics.csv` and `per_sequence_metrics.csv`; component evidence is in `diagnostics.json`, `component_results.jsonl`, and `cases.jsonl`. Large raw outputs remain under `/media/imc/data/mot-data/runs/s05_visual_decision_coverage_20261123/`.
