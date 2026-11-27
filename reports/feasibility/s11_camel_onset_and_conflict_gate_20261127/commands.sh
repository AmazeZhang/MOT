#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
PY=/media/imc/data/mot-data/envs/s08-camel/bin/python
S11=/home/imc/yzy/MOT/scripts/run_s11_camel.py
CKPT=/media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt

# Stage A
CUDA_VISIBLE_DEVICES='' $PY $S11 stage-a-lock
CUDA_VISIBLE_DEVICES='' $PY -m pytest /home/imc/yzy/MOT/tests/test_s11_camel.py -v

# Stage B: lower-threshold calibration sweep
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-th --condition T2_TH010 --sim-threshold 0.10 --ckpt $CKPT --run-id b_t2_th010_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-th --condition T2_TH030 --sim-threshold 0.30 --ckpt $CKPT --run-id b_t2_th030_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-th --condition T2_TH040 --sim-threshold 0.40 --ckpt $CKPT --run-id b_t2_th040_v1

# Stage C: 149-onset anatomy & anchor drift probe on T_TWOSTAGE_TH050
CUDA_VISIBLE_DEVICES=1 $PY $S11 stage-c-onset-anatomy --ckpt $CKPT --run-id c_onset_anatomy_v1

# Stage D: zero-retrain conflict-gated & early-pin5 closed-loop evals (sim_threshold=0.50)
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-conflict-gate --condition G1_CONFLICT_IOU030_TAU065 --ckpt $CKPT --run-id d_g1_conflict_iou030_tau065_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-conflict-gate --condition G2_MARGIN_M010_TAU070 --ckpt $CKPT --run-id d_g2_margin_m010_tau070_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-conflict-gate --condition G3_HYBRID_GATE_NO_PIN --ckpt $CKPT --run-id d_g3_hybrid_gate_no_pin_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-conflict-gate --condition G4_HYBRID_GATE_EARLY_PIN5 --ckpt $CKPT --run-id d_g4_hybrid_gate_early_pin5_v1
