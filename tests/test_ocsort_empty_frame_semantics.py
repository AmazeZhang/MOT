import sys
from pathlib import Path

import numpy as np


OC_SORT_ROOT = Path(__file__).parents[1] / "third_party" / "OC_SORT"
sys.path.insert(0, str(OC_SORT_ROOT))

from trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402


def make_tracker() -> OCSort:
    return OCSort(
        det_thresh=0.6,
        iou_threshold=0.3,
        asso_func="iou",
        delta_t=3,
        inertia=0.2,
        use_byte=False,
    )


def test_none_skips_time_but_empty_array_advances_time() -> None:
    detection = np.array([[10.0, 10.0, 20.0, 20.0, 0.9]])
    image_info = (100, 100)
    input_size = (100, 100)

    none_tracker = make_tracker()
    none_tracker.update(detection.copy(), image_info, input_size)
    assert none_tracker.frame_count == 1
    assert none_tracker.trackers[0].age == 0
    none_tracker.update(None, image_info, input_size)
    assert none_tracker.frame_count == 1
    assert none_tracker.trackers[0].age == 0

    empty_tracker = make_tracker()
    empty_tracker.update(detection.copy(), image_info, input_size)
    assert empty_tracker.frame_count == 1
    assert empty_tracker.trackers[0].age == 0
    empty_tracker.update(np.empty((0, 5)), image_info, input_size)
    assert empty_tracker.frame_count == 2
    assert empty_tracker.trackers[0].age == 1
    assert empty_tracker.trackers[0].time_since_update == 1
