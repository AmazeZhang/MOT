"""CPU checks for the S01 source detection ID provenance hook."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/OC_SORT"))
import trackers.ocsort_tracker.ocsort as ocsort_module


class SourceDetectionProvenanceTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            self.fail("integration tests require CUDA_VISIBLE_DEVICES='' ")
        self.tracker = ocsort_module.OCSort(det_thresh=.6, use_byte=False)
        self.events = []
        self.tracker._s01_record = lambda tid, did, stage: self.events.append(
            (int(tid), int(did), stage))

    def update(self, rows, ids):
        self.tracker._s01_source_det_ids = ids
        return self.tracker.update(np.asarray(rows, dtype=np.float64).reshape(-1, 5),
                                   (100, 100), (100, 100))

    def test_duplicate_boxes_keep_distinct_source_ids(self):
        box = [10, 10, 30, 40, .9]
        self.update([box, box], [7, 11])
        self.assertEqual([(did, stage) for _, did, stage in self.events],
                         [(7, "birth"), (11, "birth")])
        self.assertEqual(len({tid for tid, _, _ in self.events}), 2)

    def test_high_low_interleaving_preserves_high_ids(self):
        self.update([[10, 10, 30, 40, .9], [35, 10, 55, 40, .2],
                     [60, 10, 80, 40, .95]], [5, 6, 7])
        self.assertEqual([did for _, did, _ in self.events], [5, 7])

    def test_later_round_update_keeps_source_id(self):
        box = [10, 10, 30, 40, .9]
        self.update([box], [1])
        self.events.clear()
        self.update([box], [2])
        self.assertEqual([stage for _, _, stage in self.events], ["first"])
        self.events.clear()
        original = ocsort_module.associate
        ocsort_module.associate = lambda dets, trks, *_: (
            np.empty((0, 2), dtype=int), np.arange(len(dets)), np.arange(len(trks)))
        try:
            self.update([box], [3])
        finally:
            ocsort_module.associate = original
        self.assertEqual([(did, stage) for _, did, stage in self.events], [(3, "later")])

    def test_source_id_length_mismatch_fails_before_assignment(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            self.update([[10, 10, 30, 40, .9]], [1, 2])
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
