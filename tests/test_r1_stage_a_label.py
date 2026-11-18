import numpy as np

from run_r1_stage_a import anchor_label


def row(x1, track_id):
    return np.asarray([[x1, 0, x1 + 10, 10, track_id]], dtype=np.float64)


def test_label_uses_original_id_only_on_visible_gt_frames():
    gt = {
        1: {7: np.asarray([0, 0, 10, 10], dtype=np.float64)},
        2: {7: np.asarray([0, 0, 10, 10], dtype=np.float64)},
        3: {},
        4: {7: np.asarray([0, 0, 10, 10], dtype=np.float64)},
    }
    reference = {1: row(0, 12)}
    branches = {
        "a0": [row(20, 12), row(0, 99), row(20, 12)],
        "a1": [row(0, 12), row(0, 99), row(0, 12)],
    }
    label = anchor_label(gt, reference, branches, 2, 3)
    assert label["anchor_gt_to_original_tracker_id"] == {"7": 12}
    assert label["anchor_visible_frames"] == 2
    assert label["no_gt_frames"] == 1
    assert label["correct_frames"] == {"a0": 0, "a1": 2}
    assert label["net_correct_frame_delta"] == 2
    assert label["benefited_anchors"] == 1
