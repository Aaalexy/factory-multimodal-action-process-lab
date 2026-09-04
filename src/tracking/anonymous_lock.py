"""Expose every track change and require an explicit relock across identities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from src.legacy_pose.manual_selection import ManualSelectionSeed
from src.legacy_pose.person_tracker import PersonTracker, TrackerConfig


@dataclass
class LockObservation:
    person_ref: str
    lock_epoch: int
    track_id: int | None
    track_state: str
    lock_state: str
    candidate_person_count: int
    switch_exposed: bool
    awaiting_manual_relock: bool
    usable_pose: bool
    raw_result: Any


class AnonymousPersonLock:
    """Session-local anonymous lock; it performs no cross-video ReID."""

    def __init__(self, tracker_config: TrackerConfig | None = None) -> None:
        self.tracker = PersonTracker(tracker_config or TrackerConfig())
        self.person_sequence = 0
        self.lock_epoch = 0
        self.person_ref = "unlocked"
        self.active_track_id: int | None = None
        self.pending_track_id: int | None = None
        self.awaiting_manual_relock = False
        self._manual_reselection_pending = False
        self._manual_reselection_from_track_id: int | None = None
        self.switch_events: list[dict[str, Any]] = []

    def _new_reference(self) -> None:
        self.person_sequence += 1
        self.lock_epoch += 1
        self.person_ref = f"person-{self.person_sequence:03d}"

    def confirm_relock(self) -> None:
        """Legacy internal-candidate confirmation retained for compatibility."""

        if not self.awaiting_manual_relock or self.pending_track_id is None:
            raise RuntimeError("No pending anonymous candidate to relock")
        previous = self.active_track_id
        self.active_track_id = self.pending_track_id
        self.pending_track_id = None
        self.awaiting_manual_relock = False
        self._new_reference()
        self.switch_events.append(
            {
                "event": "manual_relock_confirmed",
                "from_track_id": previous,
                "to_track_id": self.active_track_id,
                "person_ref": self.person_ref,
                "lock_epoch": self.lock_epoch,
            }
        )

    def select_candidate(self, seed: ManualSelectionSeed) -> None:
        """Reset all Body tracking state and await this explicit anonymous seed."""

        if seed.selection_source != "manual" or not seed.manual_reselection:
            raise ValueError("Manual relock requires an explicit reselection seed")
        previous = self.active_track_id
        next_config = replace(
            self.tracker.config,
            selection_mode="manual",
            manual_selection_seed=seed,
        )
        self.tracker = PersonTracker(next_config)
        self.active_track_id = None
        self.pending_track_id = None
        self.awaiting_manual_relock = True
        self._manual_reselection_pending = True
        self._manual_reselection_from_track_id = previous
        self.switch_events.append(
            {
                "event": "manual_relock_requested",
                "from_track_id": previous,
                "selected_candidate_id": seed.candidate_id,
                "selection_frame_index": seed.selection_frame_index,
                "person_ref": self.person_ref,
                "lock_epoch": self.lock_epoch,
            }
        )

    def consume_result(self, result: Any) -> LockObservation:
        candidate_count = len(result.candidate_scores)
        current_track_id = result.track_id
        switch_exposed = False

        if self.active_track_id is None and current_track_id is not None:
            self.active_track_id = current_track_id
            self._new_reference()
            event_name = (
                "manual_relock_confirmed"
                if self._manual_reselection_pending
                else "initial_anonymous_lock"
            )
            self.switch_events.append(
                {
                    "event": event_name,
                    "from_track_id": self._manual_reselection_from_track_id,
                    "to_track_id": current_track_id,
                    "person_ref": self.person_ref,
                    "lock_epoch": self.lock_epoch,
                }
            )
            self._manual_reselection_pending = False
            self._manual_reselection_from_track_id = None
            self.awaiting_manual_relock = False
        elif (
            current_track_id is not None
            and self.active_track_id is not None
            and current_track_id != self.active_track_id
        ):
            if self.pending_track_id != current_track_id:
                self.pending_track_id = current_track_id
                self.awaiting_manual_relock = True
                switch_exposed = True
                self.switch_events.append(
                    {
                        "event": "candidate_switch_requires_manual_relock",
                        "from_track_id": self.active_track_id,
                        "candidate_track_id": current_track_id,
                        "person_ref": self.person_ref,
                        "lock_epoch": self.lock_epoch,
                    }
                )

        effective_lock_state = result.lock_state
        effective_track_state = result.state
        usable_pose = bool(
            result.state == "tracked"
            and result.smoothed_pose is not None
            and current_track_id == self.active_track_id
            and not self.awaiting_manual_relock
        )
        if self.awaiting_manual_relock:
            effective_lock_state = "awaiting_manual_relock"
            effective_track_state = "lost"
            usable_pose = False

        return LockObservation(
            person_ref=self.person_ref,
            lock_epoch=self.lock_epoch,
            track_id=self.active_track_id,
            track_state=effective_track_state,
            lock_state=effective_lock_state,
            candidate_person_count=candidate_count,
            switch_exposed=switch_exposed,
            awaiting_manual_relock=self.awaiting_manual_relock,
            usable_pose=usable_pose,
            raw_result=result,
        )

    def update(self, detections: list[Any], frame_shape: tuple[int, ...]) -> LockObservation:
        return self.consume_result(self.tracker.update(detections, frame_shape))
