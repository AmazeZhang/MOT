#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
PY=/media/imc/data/mot-data/envs/s08-camel/bin/python
S10=/home/imc/yzy/MOT/scripts/run_s10_camel.py
T2_CKPT=/media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt

# Stage A
CUDA_VISIBLE_DEVICES='' $PY $S10 stage-a-lock --p-iid 0.07688032935262179
CUDA_VISIBLE_DEVICES='' $PY -m pytest /home/imc/yzy/MOT/tests/test_s10_camel.py -q

# Stage B: D2_SKIP_ONLY + C0 zero-retrain baselines
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-skip-only --ckpt $T2_CKPT --run-id b_d2_skip_only_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-c0-write-gate --ckpt $T2_CKPT --tau-write 0.60 --run-id b_c0_tau060_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-c0-write-gate --ckpt $T2_CKPT --tau-write 0.65 --run-id b_c0_tau065_v1

# Stage C: Controlled 20-epoch training + dev8 online evaluation
CUDA_VISIBLE_DEVICES=1 $PY $S10 train-c --condition C1_MATCHED_IID_SWAP --p-iid 0.07688032935262179 --run-id c1_matched_iid_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-online --condition C1_MATCHED_IID_SWAP --ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c1_matched_iid_seed13_e20_v1/CAMEL/last.ckpt --run-id c_eval_c1_th050_v1

CUDA_VISIBLE_DEVICES=1 $PY $S10 train-c --condition C2_CONTIGUOUS_TAKEOVER --run-id c2_contiguous_takeover_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-online --condition C2_CONTIGUOUS_TAKEOVER --ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c2_contiguous_takeover_seed13_e20_v1/CAMEL/last.ckpt --run-id c_eval_c2_th050_v1

CUDA_VISIBLE_DEVICES=1 $PY $S10 train-c --condition C3_CONSISTENCY_GATED_TE --run-id c3_consistency_gated_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-online --condition C3_CONSISTENCY_GATED_TE --ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c3_consistency_gated_seed13_e20_v1/CAMEL/last.ckpt --consistency-model --run-id c_eval_c3_th050_v1

# Stage D: Soft write filter + offline AUROC on real T_TWOSTAGE cross-ID writes
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-online --condition C3_WITH_SOFT_WRITE_FILTER --ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c3_consistency_gated_seed13_e20_v1/CAMEL/last.ckpt --consistency-model --soft-write-filter-w 0.50 --run-id d_c3_soft_write_filter_w050_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 stage-d-real-pollution-auroc --t2-ckpt $T2_CKPT --c3-ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c3_consistency_gated_seed13_e20_v1/CAMEL/last.ckpt --run-id d_real_pollution_auroc_v1

# Stage E
CUDA_VISIBLE_DEVICES='' $PY /home/imc/yzy/MOT/scripts/export_s10_report.py
