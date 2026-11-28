"""Stage A unit tests for S13: early anchor >50-frame FIFO persistence, occ_drop, hist_pollute, step-0 equivalence & calibrated B1."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from hydra.utils import instantiate

sys.path.insert(0, "/home/imc/yzy/MOT/scripts")
import run_s09_camel as s09  # noqa: E402
import run_s12_camel as s12  # noqa: E402
import run_s13_camel as s13  # noqa: E402
from cameltrack.camel import Detections, Tracklets  # noqa: E402
from cameltrack.cameltrack import Tracklet  # noqa: E402


def _make_dummy_det(idx: int, box: list[float], conf: float, emb_seed: float) -> SimpleNamespace:
    emb = torch.full((6, 128), emb_seed, dtype=torch.float32)
    vis = torch.ones((6,), dtype=torch.float32)
    return SimpleNamespace(
        index=torch.tensor(idx),
        bbox_ltwh=torch.tensor(box, dtype=torch.float32),
        bbox_conf=torch.tensor([conf], dtype=torch.float32),
        embeddings=emb,
        visibility_scores=vis,
        age=torch.tensor([0.0]),
    )


def test_early_anchor_persistence_beyond_50_frames_and_slot_ordering() -> None:
    wrapper = s13.S13CAMELTrackWrapper(base_mod=SimpleNamespace(device="cpu", max_track_gallery_size=50), early_anchor_k=4)
    # Create birth det at t=0 and update 65 times (exceeding FIFO size 50), shifting embeddings to drift=99.0
    d0 = _make_dummy_det(0, [100.0, 100.0, 40.0, 100.0], 0.95, 1.0)
    trk = Tracklet(d0, max_gallery_size=50)
    wrapper._record_early_anchor(trk, d0, max_other_iou=0.05)
    for t in range(1, 66):
        trk.forward()
        dt = _make_dummy_det(t, [100.0 + t, 100.0, 40.0, 100.0], 0.92, float(t + 1) if t < 10 else 99.0)
        trk.update(dt)
        wrapper._record_early_anchor(trk, dt, max_other_iou=0.05 if t < 4 else 0.80)

    # Tracklet FIFO evicted birth frames 0..15 (only retains frames 16..65 = 50 items)
    assert len(trk.detections) == 50
    ov = wrapper._build_anchor_override([trk], "cpu")
    assert ov is not None and ov["masks"].shape == (1, 1, 4) and ov["masks"].all()
    # Early anchors MUST retain birth seeds 1.0..4.0 despite FIFO eviction!
    anchor_means = ov["embeddings"][0, 0, :, 0, 0].tolist()
    assert anchor_means == [1.0, 2.0, 3.0, 4.0]


def test_occ_drop_and_hist_pollute_invariants() -> None:
    torch.manual_seed(7)
    B, T, D, S = 1, 2, 2, 8
    embs_t = torch.randn(B, T, S, 6, 128)
    embs_t_orig = embs_t.clone()
    # Ages sorted ascending: index 0 is newest (age=1), index 7 is oldest (age=8)
    ages = torch.arange(1, S + 1, dtype=torch.float32).view(1, 1, S, 1).repeat(B, T, 1, 1)
    boxes_d = torch.tensor([[[[100.0, 100.0, 50.0, 120.0]], [[110.0, 105.0, 50.0, 120.0]]]]) / 1920.0
    feats_t = {
        "age": ages,
        "bbox_ltwh": torch.tensor([100.0, 100.0, 50.0, 120.0]).view(1, 1, 1, 4).repeat(B, T, S, 1) / 1920.0,
        "embeddings": embs_t,
        "visibility_scores": torch.ones(B, T, S, 6),
        "drop_app": torch.zeros(B, T, S, 1),
    }
    feats_d = {
        "bbox_ltwh": boxes_d,
        "embeddings": torch.randn(B, D, 1, 6, 128),
        "visibility_scores": torch.ones(B, D, 1, 6),
        "drop_app": torch.zeros(B, D, 1, 1),
    }
    t_tgts = torch.tensor([[[10.0] * S, [20.0] * S]])
    d_tgts = torch.tensor([[[10.0], [20.0]]])

    tracks = Tracklets(feats_t, ~t_tgts.isnan(), t_tgts.clone())
    dets = Detections(feats_d, ~d_tgts.isnan(), d_tgts.clone())

    n_pol = s13.inject_recent_history_pollution_(tracks, dets, p_pollute=1.0, k_anchor_keep=4, min_valid_slots=6)
    assert n_pol == 2
    # Oldest 4 slots (indices 4..7, largest age) MUST remain bitwise identical!
    assert torch.equal(tracks.feats["embeddings"][:, :, 4:], embs_t_orig[:, :, 4:])
    # At least newest 2 slots (indices 0..1) were polluted with neighbor det embeddings
    assert not torch.equal(tracks.feats["embeddings"][:, :, :2], embs_t_orig[:, :, :2])

    # Check targeted occ_drop masks only selected track's positive det
    recs = s13.apply_occlusion_drop_(tracks, dets, p_occ_drop=1.0, occ_conflict_iou=0.25)
    assert len(recs) == 1  # When first det drops, remaining det becomes singleton with no distractor -> never drops both!
    assert dets.masks[0].sum().item() == 1
    assert torch.isnan(dets.targets).sum().item() == 1


def test_n1_n2_n3_step0_equivalence_and_2step_updates(tmp_path) -> None:
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
    for cond in ("N1_KIN_SCALE_ANCHORED", "N2_KIN_OCC_DROP_SUP", "N3_DUAL_ANCHOR_CAMEL"):
        cfg_c = s13.compose_s13_train_config(tmp_path / cond, train32, dev8, dm_cache, cond)
        m = instantiate(cfg_c.modules.track.CAMEL, _recursive_=False).eval()
        m.load_state_dict(m_base.state_dict(), strict=False)
        models[cond] = m

    assert s13.extra_param_summary(models["N3_DUAL_ANCHOR_CAMEL"])["s13_extra_params"] > s13.extra_param_summary(models["N1_KIN_SCALE_ANCHORED"])["s13_extra_params"] == 146305

    B, T, D, S = 1, 3, 3, 6
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
    feats_t["bbox_ltwh"][0, 1, :, 0] += 15.0
    feats_d = {k: v[:, :D, :1].clone() for k, v in feats_t.items()}
    # Include padded slots (NaN target -> -inf in td_sim) to verify zero -inf*0=NaN bugs!
    t_targets = torch.tensor([[[10.0] * S, [20.0] * S, [float("nan")] * S]])
    d_targets = torch.tensor([[[10.0], [20.0], [float("nan")]]])

    def _mk_batch():
        return {
            "track_feats": {k: v.clone() for k, v in feats_t.items()},
            "track_targets": t_targets.clone(),
            "det_feats": {k: v.clone() for k, v in feats_d.items()},
            "det_targets": d_targets.clone(),
        }

    t_b, d_b = m_base.train_val_preprocess(_mk_batch())
    _, _, sim_base = m_base.forward(t_b, d_b)
    for cond, m in models.items():
        t_c, d_c = m.train_val_preprocess(_mk_batch())
        _, _, sim_c = m.forward(t_c, d_c)
        assert torch.allclose(sim_base, sim_c, atol=1e-5), f"{cond} step-0 diff: {(sim_base - sim_c).abs().max().item()}"

    # 2 training steps on N3 verify ALL params (including new 4D columns [:, 13:17] of edge_sim_mlp) receive finite updates
    m3 = models["N3_DUAL_ANCHOR_CAMEL"].train()
    w0 = m3.edge_sim_mlp[0].weight[:, 13:17].detach().clone()
    opt = torch.optim.AdamW(m3.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        out = m3.training_step(_mk_batch(), 0)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        for pname, p in m3.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Nonfinite grad in {pname}"
        opt.step()
    assert (m3.edge_sim_mlp[0].weight[:, 13:17] - w0).abs().sum().item() > 0.0
    assert s13.compute_val_derived_threshold(0.6869) == 0.55
    assert s13.compute_val_derived_threshold(0.5460) == 0.41
    # Verify calibrated B1 dustbin=0.62 rejects low-score match <= 0.50
    sim_low = torch.tensor([[0.50, 0.10], [0.10, 0.50]])
    m_ds, _, _, st = s12.dual_softmax_associate(sim_low, tau_ds=0.10, dustbin_score=0.62)
    assert len(m_ds) == 0 and st["dustbin_rejected"] == 2
