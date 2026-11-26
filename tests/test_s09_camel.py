"""Targeted unit tests for S09 Stage A runner, Stage 1/2 checkpoint loading, cache equivalence, and D1/D2 probes."""

from __future__ import annotations

import pickle
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from hydra.utils import instantiate

sys.path.insert(0, "/home/imc/yzy/MOT/scripts")
import run_s09_camel as s09  # noqa: E402
from cameltrack.cameltrack import Tracklet  # noqa: E402
from cameltrack.train.dataset import CAMELDataset  # noqa: E402


def test_stage1_single_cue_forward_and_stage2_partial_load(tmp_path: Path) -> None:
    s09.apply_s09_runtime_compat_patches()
    train32, dev8 = s09.load_split_lists()

    te_ckpts: dict[str, Path] = {}
    for cue in s09.TE_CUES:
        cfg_c1 = s09.compose_s09_train_config(
            run_dir=tmp_path / cue,
            train_vids=train32,
            val_vids=dev8,
            train_state=s09.S08_STATES_DIR / "states_train32.pklz",
            val_state=s09.S08_STATES_DIR / "states_dev8_train_ref.pklz",
            dm_cache_dir=tmp_path / f"dm_{cue}",
            stage1_cue=cue,
            max_epochs=20,
        )
        model = instantiate(cfg_c1.modules.track.CAMEL, _recursive_=False)
        assert list(model.temp_encs.keys()) == [cue]

        # Synthetic batch forward + loss + backward
        B, N, T = 1, 4, 3
        feats_t = {
            "age": torch.zeros(B, N, T, 1),
            "bbox_ltwh": torch.tensor([[[[10.0, 20.0, 50.0, 100.0]] * T] * N]),
            "bbox_conf": torch.ones(B, N, T, 1),
            "keypoints_xyc": torch.ones(B, N, T, 17, 3) * 0.5,
            "embeddings": torch.randn(B, N, T, 6, 128),
            "visibility_scores": torch.ones(B, N, T, 6),
            "im_width": torch.full((B, N, T, 1), 1920.0),
            "im_height": torch.full((B, N, T, 1), 1080.0),
        }
        feats_d = {k: v[:, :, :1].clone() for k, v in feats_t.items()}
        batch = {
            "track_feats": feats_t,
            "track_targets": torch.tensor([[[1.0] * T, [2.0] * T, [3.0] * T, [4.0] * T]]),
            "det_feats": feats_d,
            "det_targets": torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        }
        tracks, dets = model.train_val_preprocess(batch)
        tracks, dets, td_sim = model.forward(tracks, dets)
        assert td_sim.shape == (B, N, N)
        loss = model.compute_loss(tracks, dets, td_sim)
        assert torch.isfinite(loss)
        loss.backward()

        ckpt_file = tmp_path / f"{cue}.ckpt"
        torch.save({"state_dict": model.state_dict()}, ckpt_file)
        te_ckpts[cue] = ckpt_file

    # Stage 2 partial loading from all 3 Stage 1 checkpoints + random GAFFE
    cfg_c2 = s09.compose_s09_train_config(
        run_dir=tmp_path / "c2",
        train_vids=train32,
        val_vids=dev8,
        train_state=s09.S08_STATES_DIR / "states_train32.pklz",
        val_state=s09.S08_STATES_DIR / "states_dev8_train_ref.pklz",
        dm_cache_dir=tmp_path / "dm_c2",
        stage1_cue=None,
        te_checkpoints=te_ckpts,
        max_epochs=20,
    )
    joint_model = instantiate(cfg_c2.modules.track.CAMEL, _recursive_=False)
    for cue in s09.TE_CUES:
        c1_sd = torch.load(te_ckpts[cue], map_location="cpu", weights_only=False)["state_dict"]
        j_sd = joint_model.state_dict()
        prefix = f"temp_encs.{cue}."
        matched_keys = [k for k in c1_sd if k.startswith(prefix)]
        assert len(matched_keys) > 0
        for k in matched_keys:
            assert torch.equal(c1_sd[k], j_sd[k])


def test_cached_load_pickle_exact_equivalence() -> None:
    cache = Path("/media/imc/data/mot-data/cache/s08_camel/dm_d_t_scratch_seed13_e20_v1_isolated")
    if not cache.exists():
        pytest.skip("S08 cache not present")
    vids = sorted([n for n in zipfile.ZipFile(cache / "camel_train.pklz").namelist() if n.endswith(".pkl")])[:2]
    orig_fn = CAMELDataset._load_pickle
    try:
        ds_orig = CAMELDataset(cache / "camel_train.pklz", cache / "camel_train_tracklets.json", video_ids=vids)
        sample = ds_orig.samples[0]
        idx = [(sample["video_id"], 0, sample["image_id"][0])]
        out_orig = ds_orig[idx]

        s09.apply_s09_runtime_compat_patches()
        ds_cached = CAMELDataset(cache / "camel_train.pklz", cache / "camel_train_tracklets.json", video_ids=vids)
        out_cached1 = ds_cached[idx]
        out_cached2 = ds_cached[idx]
        for k in out_orig[0]["track_feats"]:
            np.testing.assert_array_equal(out_orig[0]["track_feats"][k], out_cached1[0]["track_feats"][k])
            np.testing.assert_array_equal(out_orig[0]["track_feats"][k], out_cached2[0]["track_feats"][k])
    finally:
        CAMELDataset._load_pickle = orig_fn


def test_clean_history_probe_routing() -> None:
    # Minimal mock CAMELTrack to verify D2 keeps gallery GT-pure while emitting CAMEL's predicted assignment
    def make_det(idx: int, bbox: list[float], tag: float) -> SimpleNamespace:
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

    Tracklet.count = 1
    det_a0 = make_det(0, [0.0, 0.0, 50.0, 100.0], 1.0)
    det_b0 = make_det(1, [200.0, 0.0, 50.0, 100.0], 2.0)
    trk_a = Tracklet(det_a0, max_gallery_size=50)
    trk_a.s09_gt_anchor = 101
    trk_b = Tracklet(det_b0, max_gallery_size=50)
    trk_b.s09_gt_anchor = 102

    # Next frame: det_a1 is GT 101, det_b1 is GT 102; CAMEL mistakenly swaps them (trk_a -> det_b1, trk_b -> det_a1)
    det_a1 = make_det(2, [2.0, 0.0, 50.0, 100.0], 1.1)
    det_b1 = make_det(3, [202.0, 0.0, 50.0, 100.0], 2.1)

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
            # Forced swap: trk 0 -> det 1 (det_b1), trk 1 -> det 0 (det_a1)
            sim = torch.tensor([[0.4, 0.8], [0.8, 0.4]])
            return np.array([[0, 1], [1, 0]]), [], [], sim

        def update_state(self, trk) -> None:
            trk.state = "active"

    gt_by_image_id = {
        10: (np.array([101, 102]), np.array([[2.0, 0.0, 50.0, 100.0], [202.0, 0.0, 50.0, 100.0]]))
    }
    stub = StubBase()
    probe = s09.CleanHistoryCAMELTrack(stub, gt_by_image_id)
    dets_df = pd.DataFrame(index=[2, 3])
    metas_df = pd.DataFrame(index=[10])
    out_df = probe.process([det_a1, det_b1], dets_df, metas_df)

    # Frame output must reflect CAMEL's swapped prediction (no current-frame cheating!)
    assert out_df.loc[3, "track_id"] == trk_a.id
    assert out_df.loc[2, "track_id"] == trk_b.id
    # Gallery must remain GT-pure (det_a1 in trk_a, det_b1 in trk_b)
    assert [d.tag for d in trk_a.detections] == [1.0, 1.1]
    assert [d.tag for d in trk_b.detections] == [2.0, 2.1]
    assert probe.routing_stats["cross_id_matches_prevented_from_gallery"] == 2
    # Duplicate sibling sharing same GT anchor must NOT receive a duplicate update or shared object ref
    trk_a_dup = Tracklet(make_det(4, [1.0, 0.0, 50.0, 100.0], 1.2), max_gallery_size=50)
    trk_a_dup.s09_gt_anchor = 101
    trk_a.hits = 10
    stub.tracklets = [trk_a, trk_b, trk_a_dup]

    class StubBase3(StubBase):
        def associate_dets_to_trks(self, tracklets, detections):
            sim = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.1, 0.1]])
            return np.array([[0, 0], [1, 1]]), [2], [], sim

    stub3 = StubBase3()
    stub3.tracklets = [trk_a, trk_b, trk_a_dup]
    probe3 = s09.CleanHistoryCAMELTrack(stub3, gt_by_image_id)
    det_a2 = make_det(5, [3.0, 0.0, 50.0, 100.0], 1.3)
    det_b2 = make_det(6, [203.0, 0.0, 50.0, 100.0], 2.3)
    probe3.process([det_a2, det_b2], pd.DataFrame(index=[5, 6]), metas_df)
    assert probe3.routing_stats["gt_routed_gallery_updates"] == 2
    assert [d.tag for d in trk_a.detections] == [1.0, 1.1, 1.3]
    assert [d.tag for d in trk_a_dup.detections] == [1.2]
    assert trk_a.detections[-1] is not det_a2
    assert torch.is_tensor(trk_a.detections[-1].age)
    assert float(trk_a.detections[-1].age.item()) == 0.0
