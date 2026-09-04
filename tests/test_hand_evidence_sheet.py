from __future__ import annotations

from scripts.build_hand_evidence_sheet import select_representative_frames


def _hand(frame: int, state: str) -> dict[str, object]:
    return {
        "frame_index": frame,
        "observation_state": state,
    }


def test_representative_sheet_selection_prioritizes_all_real_hand_states() -> None:
    payload = {
        "pose_frames": [
            {
                "source_frame_index": index,
                "timestamp": index / 8,
            }
            for index in range(15)
        ],
        "hand_pose_frames": [
            *(_hand(index, "detected") for index in range(5)),
            *(_hand(index, "uncertain") for index in range(5, 10)),
            *(_hand(index, "missing") for index in range(10, 15)),
        ],
    }
    selected = select_representative_frames(payload, per_state_limit=2)
    assert [item["hand_state"] for item in selected] == [
        "detected",
        "detected",
        "uncertain",
        "uncertain",
        "missing",
        "missing",
    ]
    assert all(
        item["hand_records"]
        for item in selected
        if item["hand_state"] != "missing"
    )
