"""Targeted unit tests for S12 causal extrapolation, kin+DOA geometry, step-0 equivalence, M0>=M2 params, M3 loss, and B1/B2."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch
from hydra.utils import instantiate

sys.path.insert(0, "/home/imc/yzy/MOT/scripts")
import run_s09_camel as s09  # noqa: E402
import run_s12_camel as s12  # noqa: E402
from cameltrack.camel import CAMEL, Detections, Tracklets  # noqa: E402
from cameltrack.cameltrack import Tracklet  # noqa: E402


def test_causal_extrapolation_constant_velocity_and_dropout_nan_safety() -> None:
    # Track 0 moves linearly dx=+0.02, dy=+0.01 per frame at ages [1, 2, 3, 4] -> t=[-1,-2,-3,-4]
    # At age=0 (t=0), extrapolated center must match x_last + dx = (0.50 + 0.02) = 0.52!
    B, T, S = 1, 3, 4
    bbox = torch.zeros((B, T, S, 4), dtype=torch.float32)
    age = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, S, 1).repeat(B, T, 1, 1)
    for s in range(S):
        cx = 0.50 - 0.02 * s
        cy = 0.40 - 0.01 * s
        bbox[0, 0, s] = torch.tensor([cx - 0.05, cy - 0.10, 0.10, 0.20])
    # Track 1 has NaN and drop_bbox==1 on slot 0 -> must gracefully disable velocity / mask drop
    bbox[0, 1, 0] = torch.tensor([float("nan"), 0.2, 0.1, 0.2])
    bbox[0, 1, 1] = torch.tensor([0.3, 0.3, 0.1, 0.2])
    masks = torch.tensor([[[True, True, True, True], [True, True, False, False], [False, False, False, False]]])
    drop_bbox = torch.zeros((B, T, S, 1), dtype=torch.float32)
    drop_bbox[0, 1, 1, 0] = 1.0  # dropped slot -> Track 1 becomes fully dropped safely

    out = s12.compute_causal_extrapolation(bbox, age, masks, drop_bbox=drop_bbox)
    assert out["has_vel"][0, 0].item() is True
    assert abs(float(out["cx_hat"][0, 0]) - 0.52) < 1e-4
    assert abs(float(out["cy_hat"][0, 0]) - 0.41) < 1e-4
    assert out["trk_valid"][0, 1].item() is False
    assert out["trk_valid"][0, 2].item() is False
    for v in out.values():
        assert torch.isfinite(v).all()


def test_step0_equivalence_m0_ge_m2_params_and_m3_finite_grads(tmp_path) -> None:
    s09.apply_s09_runtime_compat_patches()
    train32, dev8 = s09.load_split_lists()
    dm_cache = s12.ensure_shared_dm_cache()

    cfg_base = s09.compose_s09_train_config(
        tmp_path, train32, dev8,
        s09.S08_STATES_DIR / "states_train32.pklz",
        s09.S08_STATES_DIR / "states_dev8_train_ref.pklz",
        dm_cache, stage1_cue=None, te_checkpoints=s12.S09_C1_CKPTS,
    )
    m_base = instantiate(cfg_base.modules.track.CAMEL, _recursive_=False).eval()

    models = {}
    extras = {}
    for cond in ("M0_NODE_CAP_CTRL", "M1_C_KIN_EDGE", "M2_C_KIN_DOA_EDGE", "M3_C_PRIME_PLUS_A_LOSS"):
        cfg_c = s12.compose_s12_train_config(tmp_path / cond, train32, dev8, dm_cache, cond)
        m = instantiate(cfg_c.modules.track.CAMEL, _recursive_=False).eval()
        m.load_state_dict(m_base.state_dict(), strict=False)
        models[cond] = m
        extras[cond] = s12.extra_param_summary(m)["s12_extra_params"]["total_params"]

    assert extras["M0_NODE_CAP_CTRL"] >= extras["M2_C_KIN_DOA_EDGE"] > extras["M1_C_KIN_EDGE"] > 0

    B, T, D, S = 1, 3, 3, 4
    feats_t = {
        "age": torch.arange(1, S + 1, dtype=torch.float32).view(1, 1, S, 1).repeat(B, T, 1, 1),
        "bbox_ltwh": torch.tensor([100.0, 100.0, 50.0, 120.0]).view(1, 1, 1, 4).repeat(B, T, S, 1),
        "bbox_conf": torch.ones(B, T, S, 1),
        "keypoints_xyc": torch.rand(B, T, S, 17, 3),
        "embeddings": torch.randn(B, T, S, 6, 128),
        "visibility_scores": torch.ones(B, T, S, 6),
        "im_width": torch.full((B, T, S, 1), 1920.0),
        "im_height": torch.full((B, T, S, 1), 1080.0),
    }
    # Shift tracks & include 1 empty padded tracklet (T=2) and 1 unmatched track (T=1, target=99)
    feats_t["bbox_ltwh"][0, 1, :, 0] += 10.0
    feats_d = {k: v[:, :D, :1].clone() for k, v in feats_t.items()}
    feats_d["bbox_ltwh"][0, 1, 0, 0] += 12.0  # overlaps track 0 and track 1

    t_targets = torch.tensor([[[10.0, 10.0, 10.0, float("nan")], [99.0, 99.0, float("nan"), float("nan")], [float("nan")] * S]])
    d_targets = torch.tensor([[[10.0], [20.0], [float("nan")]]])
    b_base = {
        "track_feats": {k: v.clone() for k, v in feats_t.items()},
        "track_targets": t_targets.clone(),
        "det_feats": {k: v.clone() for k, v in feats_d.items()},
        "det_targets": d_targets.clone(),
    }
    t_b, d_b = m_base.train_val_preprocess(b_base)
    _, _, sim_base = m_base.forward(t_b, d_b)
    for cond, m in models.items():
        # Copy identical normalized input batch so base_model norm_coords in-place scaling doesn't double-divide!
        b_copy = {
            "track_feats": {k: v.clone() for k, v in feats_t.items()},
            "track_targets": t_targets.clone(),
            "det_feats": {k: v.clone() for k, v in feats_d.items()},
            "det_targets": d_targets.clone(),
        }
        t_c, d_c = m.train_val_preprocess(b_copy)
        _, _, sim_c = m.forward(t_c, d_c)
        assert torch.allclose(sim_base, sim_c, atol=1e-5), f"{cond} step-0 diff: {(sim_base - sim_c).abs().max().item()}"
        assert torch.isfinite(sim_c[sim_c > -1e8]).all()

    # Check backward & finite grads including new heads on M0, M1, M2, and M3 (ensuring zero dead parameters)
    for cond in ("M0_NODE_CAP_CTRL", "M1_C_KIN_EDGE", "M2_C_KIN_DOA_EDGE", "M3_C_PRIME_PLUS_A_LOSS"):
        m = models[cond].train()
        b_copy = {
            "track_feats": {k: v.clone() for k, v in feats_t.items()},
            "track_targets": t_targets.clone(),
            "det_feats": {k: v.clone() for k, v in feats_d.items()},
            "det_targets": d_targets.clone(),
        }
        t_c, d_c = m.train_val_preprocess(b_copy)
        t_c, d_c, sim_c = m.forward(t_c, d_c)
        loss = m.compute_loss(t_c, d_c, sim_c)
        assert torch.isfinite(loss)
        loss.backward()
        for pname, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Nonfinite grad in {cond}:{pname}"
        # Verify new heads receive non-zero gradient immediately at step 0 across ALL conditions!
        if cond == "M0_NODE_CAP_CTRL":
            assert m.node_t_mlp[-1].weight.grad.abs().sum().item() > 0.0
        else:
            assert m.edge_sim_mlp[-1].weight.grad.abs().sum().item() > 0.0
            assert m.edge_attn_td[-1].weight.grad.abs().sum().item() > 0.0


def test_b1_dual_softmax_and_b2_2cycle_defer() -> None:
    # 1. B1 Dustbin filter rejects below 0.40 and matches mutual best above 0.40
    sim = torch.tensor([[0.85, 0.45], [0.20, 0.35]])  # row 1 max 0.35 < dustbin 0.40
    matched, u_t, u_d, st = s12.dual_softmax_associate(sim, tau_ds=0.10, dustbin_score=0.40)
    assert matched.tolist() == [[0, 0]]
    assert u_t == [1] and u_d == [1]

    # 2. B2 Steps 1 (2-cycle kinematic swap) and 2 (extreme ambiguity defer)
    def _d(idx: int, bbox: list[float], age: float = 0.0) -> SimpleNamespace:
        return SimpleNamespace(
            index=torch.tensor(idx),
            bbox_ltwh=torch.tensor(bbox, dtype=torch.float32),
            age=torch.tensor([age], dtype=torch.float32),
        )

    Tracklet.count = 1
    # Track 0 moving right at x=10 -> x=20 (expected at x=30); Track 1 moving left at x=40 -> x=30 (expected at x=20)
    t0 = Tracklet(_d(0, [10.0, 0.0, 40.0, 80.0], 2.0), 50)
    t0.update(_d(1, [20.0, 0.0, 40.0, 80.0], 1.0))
    t0.hits = 4
    t0.state = "active"
    t1 = Tracklet(_d(2, [40.0, 0.0, 40.0, 80.0], 2.0), 50)
    t1.update(_d(3, [30.0, 0.0, 40.0, 80.0], 1.0))
    t1.hits = 4
    t1.state = "active"
    # Det 0 at x=30 (true pos for t0), Det 1 at x=20 (true pos for t1); sim wrongly slightly prefers crossed pairs with small margin < 0.10
    d0 = _d(10, [30.0, 0.0, 40.0, 80.0])
    d1 = _d(11, [20.0, 0.0, 40.0, 80.0])
    sim_cross = torch.tensor([[0.68, 0.72], [0.71, 0.67]])
    m_fixed, ut2, ud2, st2 = s12.apply_b2_margin_2cycle_defer(
        [t0, t1], [d0, d1], np.array([[0, 1], [1, 0]]), [], [], sim_cross
    )
    assert st2["twocycle_swaps"] == 1
    assert set(map(tuple, m_fixed.tolist())) == {(0, 0), (1, 1)}

    # Step 2 defer: mature active match with margin < 0.05, IoU >= 0.30, sim < 0.65 -> deferred!
    t_def = Tracklet(_d(0, [10.0, 0.0, 50.0, 100.0], 1.0), 50)
    t_def.hits = 4
    t_def.state = "active"
    d_a = _d(10, [12.0, 0.0, 50.0, 100.0])
    d_b = _d(11, [22.0, 0.0, 50.0, 100.0])
    sim_ambig = torch.tensor([[0.58, 0.56]])  # margin = 0.02 < 0.05, IoU >= 0.30, sim 0.58 < 0.65
    m_def, ut3, ud3, st3 = s12.apply_b2_margin_2cycle_defer(
        [t_def], [d_a, d_b], np.array([[0, 0]]), [], [1], sim_ambig
    )
    assert st3["deferred_matches"] == 1
    assert len(m_def) == 0 and ut3 == [0] and set(ud3) == {0, 1}
