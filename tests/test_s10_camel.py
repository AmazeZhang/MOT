"""Targeted unit tests for S10 skip-only probe, C0 write gate, C1/C2 swap transforms, and C3 causal consistency."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate

sys.path.insert(0, "/home/imc/yzy/MOT/scripts")
import run_s09_camel as s09  # noqa: E402
import run_s10_camel as s10  # noqa: E402
from cameltrack.camel import Tracklets  # noqa: E402
from cameltrack.cameltrack import Tracklet  # noqa: E402


def _make_det(idx: int, bbox: list[float], tag: float) -> SimpleNamespace:
    return SimpleNamespace(
        index=torch.tensor(idx),
        image_id=torch.tensor(10),
        bbox_ltwh=torch.tensor(bbox, dtype=torch.float32),
        bbox_conf=torch.tensor(0.95, dtype=torch.float32),
        age=torch.tensor(0.0, dtype=torch.float32),
        similarity=0.0,
        second_similarity=0.0,
        tag=tag,
    )


def test_d2_skip_only_never_injects_on_miss_and_c0_gates_low_sim() -> None:
    Tracklet.count = 1
    det_a0 = _make_det(0, [0.0, 0.0, 50.0, 100.0], 1.0)
    det_b0 = _make_det(1, [200.0, 0.0, 50.0, 100.0], 2.0)
    trk_a = Tracklet(det_a0, max_gallery_size=50)
    trk_a.hits = 4
    trk_a.s10_gt_anchor = 101
    trk_b = Tracklet(det_b0, max_gallery_size=50)
    trk_b.hits = 4
    trk_b.s10_gt_anchor = 102

    # Frame 10: trk_a misses (no match), trk_b wrongly matched to det_a1 (GT 101 != 102)
    det_a1 = _make_det(2, [2.0, 0.0, 50.0, 100.0], 1.1)

    class StubBase:
        input_columns = ["bbox_conf"]
        output_columns = ["track_id"]
        level = "image"
        batch_size = 1
        collate_fn = None
        min_init_det_conf = 0.9
        max_track_gallery_size = 50
        min_num_hits = 0
        frame_count = 1
        CAMEL = SimpleNamespace(sim_threshold=0.5)

        def __init__(self) -> None:
            self.tracklets = [trk_a, trk_b]

        def reset(self) -> None:
            pass

        def associate_dets_to_trks(self, tracklets, detections):
            sim = torch.tensor([[0.2], [0.55]])
            return np.array([[1, 0]]), [0], [], sim

        def update_state(self, trk) -> None:
            trk.state = "active"

    gt_by_image_id = {10: (np.array([101, 102]), np.array([[2.0, 0.0, 50.0, 100.0], [202.0, 0.0, 50.0, 100.0]]))}
    skip_probe = s10.SkipOnlyCleanGalleryCAMELTrack(StubBase(), gt_by_image_id)
    out = skip_probe.process([det_a1], pd.DataFrame(index=[2]), pd.DataFrame(index=[10]))
    assert out.loc[2, "track_id"] == trk_b.id
    # Both trk_a (missed) and trk_b (cross-ID) must remain untouched; zero GT injection!
    assert [d.tag for d in trk_a.detections] == [1.0]
    assert [d.tag for d in trk_b.detections] == [2.0]
    assert skip_probe.routing_stats["cross_id_matches_skipped"] == 1
    assert skip_probe.routing_stats["gt_injected_on_miss"] == 0

    # C0 simple write gate (tau_write = 0.60): sim 0.55 < 0.60 with hits=4 must emit match but skip write
    Tracklet.count = 1
    trk_b2 = Tracklet(_make_det(1, [200.0, 0.0, 50.0, 100.0], 2.0), max_gallery_size=50)
    trk_b2.hits = 4
    stub_c0 = StubBase()
    stub_c0.tracklets = [trk_b2]
    stub_c0.associate_dets_to_trks = lambda t, d: (np.array([[0, 0]]), [], [], torch.tensor([[0.55]]))
    c0_mod = s10.SimpleWriteGateCAMELTrack(stub_c0, tau_write=0.60, min_hits_to_gate=3)
    out_c0 = c0_mod.process([det_a1], pd.DataFrame(index=[2]), pd.DataFrame(index=[10]))
    assert out_c0.loc[2, "track_id"] == trk_b2.id
    assert trk_b2.time_wo_hits == 0
    assert [d.tag for d in trk_b2.detections] == [2.0]
    assert c0_mod.gate_stats["write_skipped"] == 1


def test_contiguous_takeover_constraints_and_exact_corrupted_mask() -> None:
    rng = np.random.default_rng(13)
    rows = []
    # Two co-occurring tracklets over 25 frames in same video (overlapping boxes)
    for t in range(25):
        age = 25 - t
        for tid, offset in [(101, 0.0), (202, 5.0)]:
            rows.append({
                "video_id": 1,
                "image_id": t + 1,
                "track_id": tid,
                "age": age,
                "bbox_ltwh": np.array([100.0 + offset, 100.0, 40.0, 80.0], dtype=np.float32),
                "bbox_conf": np.float32(0.95),
                "keypoints_xyc": np.full((17, 3), float(tid), dtype=np.float32),
                "embeddings": np.full((6, 128), float(tid), dtype=np.float32),
                "visibility_scores": np.ones(6, dtype=np.float32),
                "im_width": np.float32(1920.0),
                "im_height": np.float32(1080.0),
            })
    df = pd.DataFrame(rows).sort_values("age", ascending=True).reset_index(drop=True)
    xform = s10.ContiguousTakeoverTransform(p_takeover=1.0, min_len=2, max_len=6, tail_window=10, max_length=50)
    xform.set_rng(rng)
    out_df, _ = xform(df.copy(), None)

    for tid in (101, 202):
        sub = out_df[out_df["track_id"] == tid].sort_values("age", ascending=False)
        corr = sub["corrupted_mask"].to_numpy(dtype=bool)
        assert corr.any(), "Expected takeover corruption under p_takeover=1.0"
        pos = np.where(corr)[0]
        L = len(pos)
        assert 2 <= L <= 6
        assert np.array_equal(pos, np.arange(pos[0], pos[0] + L)), f"Takeover must be contiguous, got {pos}"
        assert pos[0] >= 2 * L, f"Clean prefix length {pos[0]} < 2*L ({2*L})"
        assert (len(sub) - 10) <= pos[-1] <= (len(sub) - 1)


def test_c3_causal_consistency_equivalence_and_finite_grads(tmp_path) -> None:
    s10.apply_s10_runtime_compat_patches()
    train32, dev8 = s09.load_split_lists()
    cfg = s10.compose_s10_train_config(
        run_dir=tmp_path,
        train_vids=train32,
        val_vids=dev8,
        dm_cache_dir=s10.ensure_shared_dm_cache(),
        condition="C3_CONSISTENCY_GATED_TE",
    )
    model = instantiate(cfg.modules.track.CAMEL, _recursive_=False)
    assert isinstance(model, s10.S10ConsistencyCAMEL)
    cons_params = sum(p.numel() for p in model.consistency_head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    assert cons_params / total_params < 0.02

    # 1. Exact equivalence of weighted_temporal_encode when obs_weights == 1.0
    model.eval()
    B, N, S = 1, 3, 6
    feats_t = {
        "age": torch.arange(1, S + 1, dtype=torch.float32).view(1, 1, S, 1).repeat(B, N, 1, 1),
        "bbox_ltwh": torch.rand(B, N, S, 4),
        "bbox_conf": torch.ones(B, N, S, 1),
        "keypoints_xyc": torch.rand(B, N, S, 17, 3),
        "embeddings": torch.randn(B, N, S, 6, 128),
        "visibility_scores": torch.ones(B, N, S, 6),
        "im_width": torch.full((B, N, S, 1), 1920.0),
        "im_height": torch.full((B, N, S, 1), 1080.0),
        "corrupted_mask": torch.zeros(B, N, S, 1),
    }
    # Row N=2 is an empty padded tracklet (0 valid obs) to verify zero NaNs
    masks_t = torch.tensor([[[True, True, True, True, False, False], [True, True, False, False, False, False], [False] * S]])
    trk_obj = Tracklets(feats_t, masks_t, torch.tensor([[[1.0] * 4 + [float("nan")] * 2, [2.0] * 2 + [float("nan")] * 4, [float("nan")] * S]]))
    te = model.temp_encs["bbox_encoder"]
    raw_tok = te.det_tokenizer(trk_obj)
    out_orig = s10.weighted_temporal_encode(te, trk_obj, raw_tok, obs_weights=None)
    out_ones = s10.weighted_temporal_encode(te, trk_obj, raw_tok, obs_weights=torch.ones(B, N, S))
    assert torch.allclose(out_orig, out_ones, atol=1e-5)
    assert torch.isfinite(out_ones).all()

    # 2. Strict causality: perturbing newer observation (idx 0, smallest age) CANNOT alter older consistency weights (idx >= 1)
    cat_tok = torch.cat([m.det_tokenizer(trk_obj) for m in model.temp_encs.values()], dim=-1)
    w_base, _ = model.consistency_head(cat_tok, trk_obj.feats["age"], masks_t)
    cat_pert = cat_tok.clone()
    cat_pert[:, :, 0, :] += 50.0
    w_pert, _ = model.consistency_head(cat_pert, trk_obj.feats["age"], masks_t)
    older_diff = (w_base[:, :2, 1:4] - w_pert[:, :2, 1:4]).abs().max().item()
    assert older_diff < 1e-6, f"Causal leakage detected: older weights changed by {older_diff}"

    # 3. Finite forward + backward with mixed corrupted_mask including empty tracklet slot
    model.train()
    feats_t["corrupted_mask"][0, 0, 0, 0] = 1.0
    feats_d = {k: v[:, :, :1].clone() for k, v in feats_t.items() if k != "corrupted_mask"}
    feats_d["corrupted_mask"] = torch.zeros(B, N, 1, 1)
    batch = {
        "track_feats": feats_t,
        "track_targets": torch.tensor([[[1.0] * 4 + [float("nan")] * 2, [2.0] * 2 + [float("nan")] * 4, [float("nan")] * S]]),
        "det_feats": feats_d,
        "det_targets": torch.tensor([[[1.0], [2.0], [float("nan")]]]),
    }
    tracks, dets = model.train_val_preprocess(batch)
    tracks, dets, td_sim = model.forward(tracks, dets)
    loss = model.compute_loss(tracks, dets, td_sim)
    assert torch.isfinite(loss)
    loss.backward()
    for pname, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Nonfinite grad in {pname}"
