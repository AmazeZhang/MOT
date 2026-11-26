#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
PY=/media/imc/data/mot-data/envs/s08-camel/bin/python
SCRIPT=/home/imc/yzy/MOT/scripts/run_s09_camel.py

# Stage A: asset verification, runtime lock & unit tests
CUDA_VISIBLE_DEVICES='' $PY $SCRIPT stage-a-lock
CUDA_VISIBLE_DEVICES='' $PY -m pytest /home/imc/yzy/MOT/tests/test_s09_camel.py -q

# Stage B: Zero-retrain threshold calibration on dev8 (4 runs)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_SCRATCH_TH046 --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.46 --seed 13 --run-id b_t_scratch_th046_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_SCRATCH_TH050 --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.50 --seed 13 --run-id b_t_scratch_th050_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_SCRATCH_TH_COMP --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.5197290778160095 --seed 13 --run-id b_t_scratch_th_comp_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition P_OFFICIAL_TH050 --ckpt /media/imc/data/mot-data/weights/camel/camel_bbox_app_kps_dancetrack.ckpt --sim-threshold 0.50 --seed NA --run-id b_p_official_th050_v1

# Stage C1: Single-cue TE pretraining on motion_train32 (3 runs)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT train-c1 --cue app_encoder --run-id c1_app_encoder_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT train-c1 --cue kp_encoder --run-id c1_kp_encoder_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT train-c1 --cue bbox_encoder --run-id c1_bbox_encoder_seed13_e20_v1

# Stage C2: Two-stage joint GAFFE training on motion_train32
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT train-c2 --run-id c2_twostage_joint_seed13_e20_v1 --app-ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c1_app_encoder_seed13_e20_v1/CAMEL/last.ckpt --kp-ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c1_kp_encoder_seed13_e20_v1/CAMEL/last.ckpt --bbox-ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c1_bbox_encoder_seed13_e20_v1/CAMEL/last.ckpt

# Stage C3: T_TWOSTAGE dev8 closed-loop eval (0.50 and epoch20 computed_sim_threshold)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_TWOSTAGE_TH050 --ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.50 --seed 13 --run-id c3_t_twostage_th050_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_TWOSTAGE_TH_COMP --ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold-from-summary /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/stage_c2_summary.json --seed 13 --run-id c3_t_twostage_th_comp_v1

# Stage D1: Similarity margin trace on T_TWOSTAGE_TH050 and T_SCRATCH_TH_COMP
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-d1-trace --condition T_TWOSTAGE_TH050 --ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.50 --ref-run-id c3_t_twostage_th050_v1 --run-id d1_t_twostage_th050_trace_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-d1-trace --condition T_SCRATCH_TH_COMP --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.5197290778160095 --ref-run-id b_t_scratch_th_comp_v1 --run-id d1_t_scratch_th_comp_trace_v1

# Stage D2: Privileged GT-routed clean history probe on T_TWOSTAGE_TH050
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-d2-clean-history --ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.50 --run-id d2_t_twostage_th050_clean_history_v3

# Stage E: Standard report export
CUDA_VISIBLE_DEVICES='' $PY /home/imc/yzy/MOT/scripts/export_s09_report.py
