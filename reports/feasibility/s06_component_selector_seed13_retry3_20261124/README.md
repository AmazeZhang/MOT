# S06 seed-13 component selector

Status: **COMPLETED / STOP_S06_NO_SEEDS_37_73**. The frozen primary Context+risk claim is not supported. This is a one-seed experiment on reused DanceTrack train development videos, not independent validation.

## One-page result

The experiment used 18 training videos, six internal checkpoint-selection videos and eight one-time development-evaluation videos. All are sequence-disjoint. Training supervision is a current-frame offline stable-owner proxy; inference uses only causal observable features. Local and Context were trained with seed 13, and their risk-policy checkpoints were selected on internal-dev6 by official AssA, then HOTA, then earlier epoch: Local epoch 5 and Context epoch 30.

| condition | HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| M | 69.51869387798489 | 53.56446597941373 | 67.32726439190517 | 235 |
| G | 68.90273986139066 | 52.82377438598485 | 67.32730333178326 | 270 |
| V | 70.43674964119137 | 55.035986318998795 | 68.39296122386163 | 246 |
| Local, no extra risk | 69.89846225123074 | 54.21411095355416 | 68.06824507973933 | 231 |
| Local, risk | 69.51869387798489 | 53.56446597941373 | 67.32726439190517 | 235 |
| Context, no extra risk | 70.99879509239292 | 55.768069195774885 | 69.18588278441055 | 226 |
| Context, risk | 69.61326332254522 | 53.70720180699403 | 67.46348811317048 | 239 |

Frozen Context+risk versus M is only `+0.094569 HOTA / +0.142736 AssA`, with IDSW `+4` and positive HOTA in 1/8 videos. It trails fixed V by `−0.823486 HOTA / −1.328785 AssA`. Only 2/8 primary clauses pass, so seeds 37/73 are not run.

Context-risk changes 3/77 observed components: the two affected videos move in opposite directions (`dancetrack0049` HOTA `−5.362653`; `dancetrack0074` `+11.685903`), with six unchanged. Local-risk changes 0/78 and is identical to M.

The planned Context-no-extra-risk secondary condition is descriptively positive: versus M `+1.480101 HOTA / +2.203603 AssA / −9 IDSW`, and versus V `+0.562045 HOTA / +0.732083 AssA / −20 IDSW`. It improves HOTA in four videos, worsens two and ties two. It cannot replace the frozen primary policy after inspection or establish a validated context mechanism.

## Integrity, deviations and failed attempts

Fresh same-family provisional audit is `WARN`. It independently reproduced all 693 final TrackEval metric cells exactly and all 378 proxy action labels with zero mismatch. No fabricated GT, score normalization fraud or phantom result was found.

The audit found two material qualifications discovered after results:

- The protocol says batches of 128 components; execution expands to action examples and steps at 128 actions plus a final 80 actions, with that final batch still scaled by `/128`.
- Both models allocate 25,794 parameters, but 8,192 Local neighbor-input weights are structurally inactive. Equal total parameters do not mean equal effective capacity.

These weaken protocol fidelity and Local/Context mechanism attribution; they do not change the official metrics or authorize a repaired rerun.

Retained earlier failures include two empty-tracker instrumentation failures, a training-lock relative-path invocation failure, and deterministic CUDA training stopped before its first optimizer step because `CUBLAS_WORKSPACE_CONFIG` was missing. Retry3 uses a new protocol, source snapshot, run ID and deterministic environment. Nothing was overwritten.

## Hardware and evidence

Training used `CUDA_VISIBLE_DEVICES=1`, with process `cuda:0` mapped to physical GPU1, for about 132.38 seconds total model-training wall time (~0.0368 device-hours). Instrumentation, tracking and TrackEval used `CUDA_VISIBLE_DEVICES=''`. Physical GPU0 was never exposed to project jobs. There is no archived independent prelaunch GPU-allocation snapshot, so this is supported by locks/code/runtime records rather than historical telemetry.

Full-precision result tables are `aggregate_metrics.csv` and `per_sequence_metrics.csv`; curves, checkpoint selection, gate clauses, decision/case evidence and audit are adjacent. Large artifacts remain under `/media/imc/data/mot-data/runs/s06_component_selector_seed13_retry2_20261123/` and `/media/imc/data/mot-data/runs/s06_component_selector_seed13_retry3_20261124/`.

## Decision

Result-to-claim is `no`, confidence high: stop S06, preserve all results, and keep seeds 37/73 blocked. No new performance run is automatically warranted. A future no-extra-risk study would require a new question and protocol, corrected component batching, an effective-capacity control and an untouched evaluation design; it would not be a continuation or repair of this gate.
