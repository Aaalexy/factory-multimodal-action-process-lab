import numpy as np

from src.person_tracker import PersonTracker, TrackerConfig
from src.pose_postprocess import PoseDetection


def _person(x, y=20, confidence=0.8, width=80, height=160):
    bbox = np.array([x, y, x + width, y + height], np.float32)
    keypoints = np.zeros((17, 3), np.float32)
    for index in range(17):
        keypoints[index] = [x + width * (0.35 + 0.02 * index), y + 10 + index * 8, 0.9]
    return PoseDetection(bbox, confidence, keypoints)


def test_tracker_prefers_continuity_over_single_frame_largest_box():
    tracker = PersonTracker()
    first = tracker.update([_person(20), _person(300, confidence=0.5, width=50)])
    assert first.detection is not None
    assert first.detection.bbox[0] == 20

    continuous = _person(25, confidence=0.65)
    distractor = _person(300, confidence=0.99, width=140, height=220)
    second = tracker.update([distractor, continuous])
    assert second.state == "tracked"
    assert second.detection is continuous
    assert second.track_id == first.track_id


def test_tracker_does_not_force_far_person_after_target_disappears():
    tracker = PersonTracker(TrackerConfig(max_lost_frames=2))
    first = tracker.update([_person(10)])
    result = tracker.update([_person(600, confidence=0.99)])
    assert result.state == "uncertain"
    assert result.detection is None
    assert result.track_id == first.track_id


def test_tracker_has_finite_lost_window_and_new_track_afterward():
    tracker = PersonTracker(TrackerConfig(max_lost_frames=1))
    first = tracker.update([_person(10)])
    assert tracker.update([]).state == "uncertain"
    lost = tracker.update([])
    assert lost.state == "lost"
    restarted = tracker.update([_person(500)])
    assert restarted.state == "tracked"
    assert restarted.track_id != first.track_id


def test_torso_consistency_prevents_extremity_only_match():
    tracker = PersonTracker()
    original = _person(50)
    tracker.update([original])
    good = _person(54, confidence=0.6)
    bad = _person(54, confidence=0.99)
    bad.keypoints[[5, 6, 11, 12], 0] += 180
    selected = tracker.update([bad, good])
    assert selected.detection is good
