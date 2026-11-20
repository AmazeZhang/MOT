"""Protocol-critical invariants; runnable with unittest or pytest, CPU only."""
import unittest

import numpy as np

from mot_research.association.appearance_core import (
    CausalAppearanceMemory, fuse_appearance_score, normalize_features,
    select_high_detections,
)


class AppearanceCoreTests(unittest.TestCase):
    def test_interleaved_scores_preserve_detection_identity(self):
        rows = np.array([[0, 0, 2, 3, s] for s in [.2, .9, .6, .8]])
        ids, high, indices = select_high_detections([40, 11, 91, 7], rows)
        self.assertEqual(ids, (11, 7))
        np.testing.assert_array_equal(indices, [1, 3])
        np.testing.assert_array_equal(high, rows[[1, 3]])
        high[0, 0] = 99
        self.assertEqual(rows[1, 0], 0)

    def test_bad_detection_identity_fails(self):
        rows = np.array([[0, 0, 2, 3, .9]] * 2)
        for ids in ([1, 1], [1, 1.5], [True, 2]):
            with self.assertRaises(ValueError):
                select_high_detections(ids, rows)

    def test_invalid_and_large_features(self):
        normalized, valid = normalize_features(
            np.array([[0, 0], [np.nan, 1], [np.inf, 1], [1e300, 1e300]])
        )
        np.testing.assert_array_equal(valid, [False, False, False, True])
        np.testing.assert_array_equal(normalized[:3], np.zeros((3, 2)))
        np.testing.assert_allclose(normalized[3], [2**-.5, 2**-.5])

    def test_zero_weight_bypasses_unusable_features(self):
        base = np.array([[.2, -.1], [.5, .4]])
        fused, used = fuse_appearance_score(base, np.array(np.nan),
                                            np.array(np.nan), base > 0, weight=0)
        np.testing.assert_array_equal(fused, base)
        self.assertFalse(used.any())
        fused[0, 0] = 10
        self.assertEqual(base[0, 0], .2)

    def test_identity_evidence_changes_ambiguous_pair_order(self):
        base = np.full((2, 2), .5)
        # Detection 0 corresponds to track 1; detection 1 to track 0.
        fused, used = fuse_appearance_score(base, np.eye(2), np.eye(2)[::-1],
                                            np.ones((2, 2), dtype=bool))
        self.assertTrue(used.all())
        self.assertGreater(fused[0, 1] + fused[1, 0], fused[0, 0] + fused[1, 1])

    def test_invalid_or_ineligible_edges_do_not_change(self):
        base = np.full((2, 2), .5)
        eligible = np.array([[False, True], [True, True]])
        fused, used = fuse_appearance_score(base, np.array([[1, 0], [0, 0]]),
                                            np.eye(2), eligible)
        np.testing.assert_array_equal(used, [[False, True], [False, False]])
        np.testing.assert_array_equal(fused, base)  # only usable cosine is zero

    def test_memory_reads_only_prior_frames_and_owns_snapshots(self):
        mem = CausalAppearanceMemory(2)
        features, frames = mem.begin_frame(1, [])
        self.assertEqual(features.shape, (0, 2))
        mem.finish_frame(1, [8], {8: 55}, [55], np.array([[3, 4]]))
        features, frames = mem.begin_frame(2, [8, 9])
        np.testing.assert_array_equal(frames, [1, -1])
        np.testing.assert_allclose(features[0], [.6, .8])
        features[0] = 0
        with self.assertRaises(ValueError):
            mem.begin_frame(2, [8])
        mem.finish_frame(2, [8], {}, [], np.empty((0, 2)))
        features, frames = mem.begin_frame(3, [8])
        np.testing.assert_allclose(features[0], [.6, .8])
        self.assertEqual(frames[0], 1)

    def test_final_mapping_must_be_one_to_one_and_retry_is_atomic(self):
        mem = CausalAppearanceMemory(2)
        mem.begin_frame(1, [])
        with self.assertRaises(ValueError):
            mem.finish_frame(1, [1, 2], {1: 4, 2: 4}, [4], np.array([[1, 0]]))
        with self.assertRaises(ValueError):
            mem.finish_frame(1, [1], {1: 99}, [4], np.array([[1, 0]]))
        mem.finish_frame(1, [1], {1: 4}, [4], np.array([[1, 0]]))
        features, frames = mem.begin_frame(2, [1, 2])
        np.testing.assert_array_equal(frames, [1, -1])
        np.testing.assert_array_equal(features, [[1, 0], [0, 0]])

    def test_invalid_write_and_deleted_track(self):
        mem = CausalAppearanceMemory(2)
        mem.begin_frame(1, [])
        mem.finish_frame(1, [1, 2], {1: 4, 2: 5}, [4, 5], np.eye(2))
        mem.begin_frame(2, [1, 2])
        mem.finish_frame(2, [1], {1: 6}, [6], np.array([[np.nan, 0]]))
        features, frames = mem.begin_frame(3, [1, 2])
        np.testing.assert_array_equal(frames, [1, -1])
        np.testing.assert_array_equal(features, [[1, 0], [0, 0]])

    def test_ema_math_and_no_same_frame_reentry(self):
        mem = CausalAppearanceMemory(2, mode="ema", alpha=.9)
        mem.begin_frame(1, [])
        mem.finish_frame(1, [1], {1: 1}, [1], np.array([[1, 0]]))
        mem.begin_frame(2, [1])
        mem.finish_frame(2, [1], {1: 2}, [2], np.array([[0, 1]]))
        with self.assertRaises(ValueError):
            mem.begin_frame(2, [1])
        features, frames = mem.begin_frame(3, [1])
        np.testing.assert_allclose(features[0], np.array([.9, .1]) / np.sqrt(.82))
        self.assertEqual(frames[0], 2)

    def test_empty_matrices_are_supported(self):
        normalized, valid = normalize_features(np.empty((0, 2)))
        self.assertEqual(normalized.shape, (0, 2))
        self.assertEqual(valid.shape, (0,))
        fused, used = fuse_appearance_score(np.empty((0, 3)), np.empty((0, 2)),
                                            np.eye(3, 2), np.empty((0, 3), dtype=bool))
        self.assertEqual(fused.shape, (0, 3))
        self.assertFalse(used.any())


if __name__ == "__main__":
    unittest.main()
