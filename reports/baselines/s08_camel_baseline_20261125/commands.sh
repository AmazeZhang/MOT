#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
PY=/media/imc/data/mot-data/envs/s08-camel/bin/python
SCRIPT=/home/imc/yzy/MOT/scripts/run_s08_camel.py

# Stage A: asset audit & runtime lock (CPU)
CUDA_VISIBLE_DEVICES='' $PY /home/imc/yzy/MOT/scripts/prepare_s08_camel_assets.py
CUDA_VISIBLE_DEVICES='' $PY $SCRIPT resolve-configs

# Stage B: P_OFFICIAL online boundary verification on dancetrack0027 (single physical GPU 1..7, GPU 0 forbidden)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-b --run-id b_p_official_online_check_v2

# Stage C: T_SCRATCH_SMOKE 100 steps with step-50 save & new-process resume + reload check
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-c-part1 --run-id c_t_scratch_smoke_v5
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-c-part2 --run-id c_t_scratch_smoke_v5
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-c-reload-check --run-id c_t_scratch_smoke_v5

# Stage D: T_SCRATCH full 20-epoch training on motion_train32 (new dir, isolated cache)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-d --run-id d_t_scratch_seed13_e20_v1

# Stage E: Independent closed-loop evaluation on motion_dev8 (P_OFFICIAL and T_SCRATCH)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-e --condition P_OFFICIAL --run-id e_p_official_dev8_v2
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-e --condition T_SCRATCH --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --run-id e_t_scratch_dev8_v1

# Stage F: Bounded natural failure summary on T_SCRATCH dev8 predictions + report bundle
CUDA_VISIBLE_DEVICES='' $PY /home/imc/yzy/MOT/scripts/export_s08_failures.py
CUDA_VISIBLE_DEVICES='' $PY /home/imc/yzy/MOT/scripts/export_s08_report.py
