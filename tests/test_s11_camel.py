"""Unit tests for S11 conflict IoU, two-sided margin, 2-cycle detection, skip-write, and early-pin5 invariants."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/imc/yzy/MOT/scripts")
import run_s11_camel as s11  # noqa: E402
from cameltrack.cameltrack import Tracklet  # noqa: E402


def _make_det(idx: int, bbox: list[float], tag: float, emb_seed: float = 0.0) -> SimpleNamespace:
    vec = torch.zeros((6, 128), dtype=torch.float32)
    vec[:, 0] = 1.0
    vec[:, 1] = float(emb_seed)
    return SimpleNamespace(
        index=torch.tensor(idx),
        image_id=torch.tensor(10),
        bbox_ltwh=torch.tensor(bbox, dtype=torch.float32),
        bbox_conf=torch.tensor(0.95, dtype=torch.float32),
        age=torch.tensor([0.0], dtype=torch.float32),
        embeddings=vec,
        visibility_scores=torch.ones((6,), dtype=torch.float32),
        similarity_with_tracklet=None,
        similarities=None,
        tag=tag,
    )


def test_conflict_iou_and_two_sided_margin_exclude_self() -> None:
    Tracklet.count = 1
    d0 = _make_det(0, [0.0, 0.0, 100.0, 100.0], 0.0)
    d1 = _make_det(1, [20.0, 0.0, 100.0, 100.0], 1.0)
    d2 = _make_det(2, [500.0, 500.0, 100.0, 100.0], 2.0)

    t0 = Tracklet(_make_det(10, [0.0, 0.0, 100.0, 100.0], 10.0), max_gallery_size=50)
    t0.state = "active"
    t1 = Tracklet(_make_det(11, [20.0, 0.0, 100.0, 100.0], 11.0), max_gallery_size=50)
    t1.state = "active"
    t2_lost = Tracklet(_make_det(12, [0.0, 0.0, 100.0, 100.0], 12.0), max_gallery_size=50)
    t2_lost.state = "lost"  # Lost tracks must NOT inflate trk_max_other_iou

    sim = torch.tensor([
        [0.80, 0.72, 0.10],
        [0.40, 0.85, 0.20],
        [0.05, 0.05, 0.95],
    ])
    sig = s11.compute_frame_conflict_signals([t0, t1, t2_lost], [d0, d1, d2], sim)
    # IoU([0,0,100,100], [20,0,100,100]) = 80*100 / (10000+10000-8000) = 8000/12000 = 2/3
    assert abs(sig["det_max_other_iou"][0] - (2.0 / 3.0)) < 1e-6
    assert sig["det_max_other_iou"][2] == 0.0
    assert abs(sig["trk_max_other_iou"][0] - (2.0 / 3.0)) < 1e-6
    assert abs(sig["trk_max_other_iou"][1] - (2.0 / 3.0)) < 1e-6

    margin0, other_t0, other_d0 = s11.compute_two_sided_margin(sig["sim_np"], 0, 0)
    assert abs(other_t0 - 0.40) < 1e-6
    assert abs(other_d0 - 0.72) < 1e-6
    assert abs(margin0 - (0.80 - 0.72)) < 1e-6

    # Single-track, single-det edge case -> margin must be +inf
    m_single, ot, od = s11.compute_two_sided_margin(np.array([[0.62]]), 0, 0)
    assert math.isinf(m_single) and m_single > 0
    assert math.isinf(ot) and ot < 0
    assert math.isinf(od) and od < 0


def test_detect_two_cycles_same_frame_and_temporal() -> None:
    # 1. Same-frame mutual exchange: trk 1 (anchor 101 -> det 202) and trk 2 (anchor 202 -> det 101)
    sw1 = s11.detect_two_cycles([], [(1, 101, 202), (2, 202, 101), (3, 303, 303)])
    assert sw1 == {1, 2}

    # 2. Temporal t-1 -> t exchange: at t-1 trk 4 matched 10 & trk 5 matched 20; at t trk 4 matched 20 & trk 5 matched 10
    sw2 = s11.detect_two_cycles([(4, 10), (5, 20)], [(4, 99, 20), (5, 88, 10)])
    assert sw2 == {4, 5}


def test_g1_g3_skip_leaves_detections_untouched_and_g4_early_pin5_invariants() -> None:
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

        def __init__(self, tracklets: list[Any], sim_fn: Any) -> None:
            self.tracklets = tracklets
            self._sim_fn = sim_fn

        def reset(self) -> None:
            pass

        def associate_dets_to_trks(self, tracklets, detections):
            return self._sim_fn(tracklets, detections)

        def update_state(self, trk) -> None:
            trk.state = "active"

    # Case A: G1 skips write when det overlaps (IoU=0.67 >= 0.30) and sim=0.60 < 0.65
    Tracklet.count = 1
    trk_g1 = Tracklet(_make_det(0, [0.0, 0.0, 100.0, 100.0], 0.0), max_gallery_size=50)
    trk_g1.hits = 4
    trk_g1.state = "active"
    det_main = _make_det(10, [1.0, 0.0, 100.0, 100.0], 9.9)
    det_comp = _make_det(11, [20.0, 0.0, 100.0, 100.0], 8.8)

    stub_g1 = StubBase(
        [trk_g1],
        lambda t, d: (np.array([[0, 0]]), [], [1], torch.tensor([[0.60, 0.20]])),
    )
    g1_mod = s11.ConflictGatedCAMELTrack(stub_g1, "G1_CONFLICT_IOU030_TAU065", min_hits_to_gate=3)
    out_g1 = g1_mod.process([det_main, det_comp], pd.DataFrame(index=[10, 11]), pd.DataFrame(index=[1]))
    assert out_g1.loc[10, "track_id"] == trk_g1.id
    assert trk_g1.hits == 5
    assert trk_g1.time_wo_hits == 0
    assert [d.tag for d in trk_g1.detections] == [0.0], "G1 skip must NOT alter trk.detections"
    assert g1_mod.gate_stats["write_skipped"] == 1

    # Case B: G3 does NOT skip isolated fast-motion match (IoU=0, margin=+inf, sim=0.60)
    Tracklet.count = 1
    trk_g3 = Tracklet(_make_det(0, [0.0, 0.0, 100.0, 100.0], 0.0), max_gallery_size=50)
    trk_g3.hits = 4
    trk_g3.state = "active"
    stub_g3 = StubBase(
        [trk_g3],
        lambda t, d: (np.array([[0, 0]]), [], [], torch.tensor([[0.60]])),
    )
    g3_mod = s11.ConflictGatedCAMELTrack(stub_g3, "G3_HYBRID_GATE_NO_PIN", min_hits_to_gate=3)
    g3_mod.process([det_main], pd.DataFrame(index=[10]), pd.DataFrame(index=[1]))
    assert [d.tag for d in trk_g3.detections] == [0.0, 9.9]
    assert g3_mod.gate_stats["write_skipped"] == 0

    # Case C: G4_HYBRID_GATE_EARLY_PIN5 pins earliest 5 mature unconflicted slots across 80 updates with len <= 50
    Tracklet.count = 1
    trk_g4 = Tracklet(_make_det(0, [0.0, 0.0, 100.0, 100.0], 0.0), max_gallery_size=50)
    trk_g4.state = "active"
    stub_g4 = StubBase(
        [trk_g4],
        lambda t, d: (np.array([[0, 0]]), [], [], torch.tensor([[0.90]])),
    )
    g4_mod = s11.ConflictGatedCAMELTrack(stub_g4, "G4_HYBRID_GATE_EARLY_PIN5", min_hits_to_gate=3, early_pin_k=5)
    for step in range(1, 81):
        d_step = _make_det(step, [0.0, 0.0, 100.0, 100.0], float(step))
        g4_mod.process([d_step], pd.DataFrame(index=[step]), pd.DataFrame(index=[step]))

    assert len(trk_g4.detections) == 50
    pinned_tags = [d.tag for d in trk_g4.detections[:5]]
    # At birth hits=0; steps 1..3 increment hits 0->1->2->3 (unpinned); steps 4..8 see hits>=3 -> tags 4..8 pinned!
    assert pinned_tags == [4.0, 5.0, 6.0, 7.0, 8.0], f"Expected pinned [4..8], got {pinned_tags}"
    assert [d.tag for d in trk_g4.detections[5:]] == list(map(float, range(36, 81)))
    # Verify forward() still ages pinned detections normally
    assert float(trk_g4.detections[0].age.item()) == float(80 - 4)
