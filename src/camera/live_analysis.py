"""Real frame-local Body, Hand, anonymous-lock and conservative action analysis."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from src.action_segmentation import (
    CausalCoarseActionClassifier,
    FrameActionStabilityConfig,
    stabilize_coarse_frames,
)
from src.action_segmentation.coarse import CoarseFrame
from src.hand_pose import DisabledHandBackend, MediaPipeHandLandmarkerBackend
from src.legacy_pose.manual_selection import ManualSelectionSeed
from src.pose_core import PoseRuntime
from src.tracking import AnonymousPersonLock


def _safe_points(values: np.ndarray | None) -> list[list[float | None]]:
    if values is None:
        return []
    result: list[list[float | None]] = []
    for row in np.asarray(values):
        result.append(
            [
                round(float(value), 3) if np.isfinite(value) else None
                for value in row
            ]
        )
    return result


def _candidate_payload(
    detections: list[Any],
    *,
    frame_index: int,
    timestamp: float,
    width: int,
    height: int,
    mirror_horizontal: bool,
) -> list[dict[str, Any]]:
    """Create non-identifying, frame-bound selection evidence."""

    candidates: list[dict[str, Any]] = []
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    for index, detection in enumerate(detections):
        bbox = np.asarray(detection.bbox, dtype=np.float32)
        torso = np.asarray(detection.keypoints[[5, 6, 11, 12]], dtype=np.float32)
        normalized_torso = torso.copy()
        normalized_torso[:, 0] /= max(1, width)
        normalized_torso[:, 1] /= max(1, height)
        fingerprint_source = np.concatenate(
            [bbox, torso[:, :2].reshape(-1)]
        ).astype(np.float32).tobytes()
        size = bbox[2:] - bbox[:2]
        candidates.append(
            {
                "candidate_id": f"camera-candidate-{index + 1:03d}",
                "bbox": [round(float(value), 3) for value in bbox],
                "center": [
                    round(float((bbox[0] + bbox[2]) * 0.5), 3),
                    round(float((bbox[1] + bbox[3]) * 0.5), 3),
                ],
                "size": [round(float(value), 3) for value in size],
                "torso_keypoints": [
                    [round(float(value), 6) for value in point]
                    for point in torso
                ],
                "normalized_bbox": [
                    round(float(value), 8) for value in bbox / scale
                ],
                "normalized_torso_keypoints": [
                    [round(float(value), 8) for value in point]
                    for point in normalized_torso
                ],
                "confidence": round(float(detection.confidence), 6),
                "source_frame_index": int(frame_index),
                "timestamp": round(float(timestamp), 6),
                "source_width": int(width),
                "source_height": int(height),
                "mirror_horizontal": bool(mirror_horizontal),
                "candidate_fingerprint": hashlib.sha256(
                    fingerprint_source
                ).hexdigest(),
            }
        )
    return candidates


class LiveFrameAnalyzer:
    """Stateful real inference adapter. It contains no mock evidence."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        body_model_path: str,
        hand_model_path: str,
        hand_enabled: bool,
        session_id: str,
        analysis_fps: float,
        body_provider_policy: str = "prefer_cuda",
        mirror_horizontal: bool = False,
        candidate_history_size: int = 32,
    ) -> None:
        root = Path(project_root).resolve()
        body_path = (root / body_model_path).resolve()
        hand_path = (root / hand_model_path).resolve()
        self.pose = PoseRuntime(
            body_path,
            provider_policy=body_provider_policy,
        )
        self.lock = AnonymousPersonLock()
        self.classifier = CausalCoarseActionClassifier()
        self.analysis_fps = float(analysis_fps)
        self.session_id = str(session_id)
        self.mirror_horizontal = bool(mirror_horizontal)
        self.candidate_history_size = max(4, int(candidate_history_size))
        self._candidate_history: OrderedDict[
            int, list[dict[str, Any]]
        ] = OrderedDict()
        self.manual_relock_events: list[dict[str, Any]] = []
        self.frames: list[CoarseFrame] = []
        self.action_sample_count = 0
        self._next_action_due: float | None = None
        self._last_action_boundary: tuple[str, int, str, str] | None = None
        self._last_action_payload: dict[str, Any] | None = None
        self.stream_evidence_hash = hashlib.sha256(
            f"local-usb-session:{session_id}".encode("utf-8")
        ).hexdigest()
        self.hand_error: str | None = None
        if hand_enabled and hand_path.is_file():
            try:
                self.hand = MediaPipeHandLandmarkerBackend(
                    hand_path,
                    model_version=f"mediapipe_hand_landmarker:{hand_path.name}",
                )
            except Exception as exc:
                self.hand = DisabledHandBackend()
                self.hand_error = (
                    f"hand_backend_initialization_failed:{type(exc).__name__}"
                )
                self.hand.reason = self.hand_error
        else:
            self.hand = DisabledHandBackend()
            self.hand_error = (
                "hand_model_missing" if hand_enabled else "hand_disabled"
            )
            self.hand.reason = self.hand_error

    def close(self) -> None:
        self.hand.close()

    def confirm_relock(self, selection: dict[str, Any]) -> dict[str, Any]:
        """Revalidate one controller-authorized candidate and reset all lanes."""

        if selection.get("session_id") != self.session_id:
            raise RuntimeError("Candidate does not belong to this Camera session")
        frame_index = int(selection.get("source_frame_index", -1))
        fingerprint = str(selection.get("candidate_fingerprint", ""))
        candidate_id = str(selection.get("candidate_id", ""))
        candidates = self._candidate_history.get(frame_index)
        if not candidates:
            raise RuntimeError("Candidate frame is no longer in worker history")
        matches = [
            item
            for item in candidates
            if item["candidate_id"] == candidate_id
            and item["candidate_fingerprint"] == fingerprint
        ]
        if len(matches) != 1:
            raise RuntimeError("Candidate disappeared or failed revalidation")
        candidate = matches[0]
        seed = ManualSelectionSeed(
            candidate_id=candidate_id,
            video_path=f"local-usb-session:{self.session_id}",
            selection_timestamp=float(candidate["timestamp"]),
            selection_frame_index=frame_index,
            bbox=tuple(float(value) for value in candidate["bbox"]),
            center=tuple(float(value) for value in candidate["center"]),
            size=tuple(float(value) for value in candidate["size"]),
            torso_keypoints=tuple(
                tuple(float(value) for value in point)
                for point in candidate["torso_keypoints"]
            ),
            person_confidence=float(candidate["confidence"]),
            selection_source="manual",
            manual_reselection=True,
            source_width=int(candidate["source_width"]),
            source_height=int(candidate["source_height"]),
            mirror_horizontal=bool(candidate["mirror_horizontal"]),
            normalized_bbox=tuple(
                float(value) for value in candidate["normalized_bbox"]
            ),
            normalized_torso_keypoints=tuple(
                tuple(float(value) for value in point)
                for point in candidate["normalized_torso_keypoints"]
            ),
            camera_backend="local_usb",
            selected_candidate_fingerprint=fingerprint,
        )
        prior_person_ref = self.lock.person_ref
        prior_lock_epoch = self.lock.lock_epoch
        self.lock.select_candidate(seed)
        self.hand.reset()
        self.classifier = CausalCoarseActionClassifier()
        self.frames.clear()
        self._next_action_due = None
        self._last_action_boundary = None
        self._last_action_payload = None
        event = {
            "event": "manual_relock_selection_accepted",
            "candidate_id": candidate_id,
            "source_frame_index": frame_index,
            "prior_person_ref": prior_person_ref,
            "prior_lock_epoch": prior_lock_epoch,
            "body_smoother_reset": True,
            "hand_state_reset": True,
            "temporal_action_state_reset": True,
            "status": "proposed",
            "training_eligible": False,
        }
        self.manual_relock_events.append(event)
        return event

    def _action_sample_due(
        self,
        *,
        timestamp: float,
        person_ref: str,
        lock_epoch: int,
        track_state: str,
        lock_state: str,
    ) -> bool:
        """Keep action sampling at 8 FPS while forcing true boundaries through."""

        boundary = (
            person_ref,
            int(lock_epoch),
            track_state,
            lock_state,
        )
        boundary_changed = (
            self._last_action_boundary is not None
            and boundary != self._last_action_boundary
        )
        self._last_action_boundary = boundary
        interval = 1.0 / self.analysis_fps
        if self._next_action_due is None or boundary_changed:
            self._next_action_due = float(timestamp) + interval
            return True
        if float(timestamp) + 1e-9 < self._next_action_due:
            return False
        while self._next_action_due <= float(timestamp) + 1e-9:
            self._next_action_due += interval
        return True

    @staticmethod
    def _pose_quality(
        *,
        track_state: str,
        keypoints: np.ndarray | None,
        statuses: np.ndarray | list[str] | None,
    ) -> dict[str, Any]:
        if track_state != "tracked" or keypoints is None or statuses is None:
            return {
                "observation_state": "lost",
                "detected_ratio": 0.0,
                "predicted_ratio": 0.0,
                "interpolated_ratio": 0.0,
                "missing_ratio": 1.0,
                "direction_clear": False,
                "required_joints_reliable": False,
            }
        points = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
        states = np.asarray(statuses, dtype="<U16").reshape(17)
        ratios = {
            name: float(np.count_nonzero(states == name)) / 17.0
            for name in ("detected", "predicted", "interpolated", "missing")
        }
        ratios["missing"] += float(
            np.count_nonzero(np.isin(states, ["uncertain", "rejected"]))
        ) / 17.0
        arm_indices = [5, 6, 7, 8, 9, 10]
        reliable = np.isin(
            states[arm_indices],
            ["detected", "predicted", "interpolated"],
        )
        finite = np.isfinite(points[arm_indices, :2]).all(axis=1)
        return {
            "observation_state": (
                "detected"
                if ratios["detected"] >= 0.65
                else "interpolated"
                if ratios["interpolated"] + ratios["predicted"] > 0
                else "missing"
            ),
            "detected_ratio": ratios["detected"],
            "predicted_ratio": ratios["predicted"],
            "interpolated_ratio": ratios["interpolated"],
            "missing_ratio": min(1.0, ratios["missing"]),
            "direction_clear": False,
            "required_joints_reliable": bool(
                np.count_nonzero(reliable & finite) >= 4
            ),
        }

    def _sample_action(
        self,
        *,
        timestamp: float,
        frame_index: int,
        locked: Any,
        detections: list[Any],
        keypoints: np.ndarray | None,
        statuses: np.ndarray | list[str] | None,
    ) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        action, anatomical_side, evidence = self.classifier.classify(
            timestamp=timestamp,
            person_ref=locked.person_ref,
            lock_epoch=locked.lock_epoch,
            track_state=locked.track_state,
            lock_state=locked.lock_state,
            keypoints=keypoints,
            statuses=statuses,
        )
        coarse = CoarseFrame(
            timestamp=float(timestamp),
            source_frame_index=int(frame_index),
            person_ref=locked.person_ref,
            lock_epoch=locked.lock_epoch,
            track_state=locked.track_state,
            lock_state=locked.lock_state,
            candidate_person_count=len(detections),
            action=action,
            anatomical_side=anatomical_side,
            observation_state=str(evidence["observation_state"]),
            detected_ratio=float(evidence["detected_ratio"]),
            predicted_ratio=float(evidence["predicted_ratio"]),
            interpolated_ratio=float(evidence["interpolated_ratio"]),
            missing_ratio=float(evidence["missing_ratio"]),
            direction_clear=bool(evidence["direction_clear"]),
            required_joints_reliable=bool(
                evidence["required_joints_reliable"]
            ),
            keypoints=_safe_points(keypoints),
            keypoint_statuses=(
                [str(value) for value in statuses]
                if statuses is not None
                else []
            ),
        )
        self.frames.append(coarse)
        self.action_sample_count += 1
        horizon = max(32, int(np.ceil(5.0 * self.analysis_fps)))
        self.frames = self.frames[-horizon:]
        stabilized = stabilize_coarse_frames(
            self.frames,
            FrameActionStabilityConfig(
                start_confirmation_seconds=0.5,
                stop_confirmation_seconds=0.5,
                temporal_context_seconds=2.5,
                bounded_uncertain_gap_seconds=0.375,
            ),
        )["frames"]
        current = stabilized[-1]
        run = [current]
        for previous in reversed(stabilized[:-1]):
            if (
                previous.action != current.action
                or previous.person_ref != current.person_ref
                or previous.lock_epoch != current.lock_epoch
                or previous.anatomical_side != current.anatomical_side
                or previous.hard_boundary
            ):
                break
            run.append(previous)
        stable_duration = (
            float(current.timestamp) - float(run[-1].timestamp)
            + 1.0 / self.analysis_fps
        )
        stable_eligible = bool(
            current.action not in {"lost", "unknown", "transition"}
            and stable_duration >= 1.2 - 1e-9
            and not current.hard_boundary
        )
        payload = {
            "action": current.action if stable_eligible else "transition",
            "raw_action": action,
            "anatomical_side": current.anatomical_side,
            "duration_seconds": round(stable_duration, 6),
            "display_eligible": stable_eligible,
            "status": "proposed" if stable_eligible else "uncertain",
            "training_eligible": False,
            "source_frame_indices": [
                item.source_frame_index for item in reversed(run)
            ],
            "temporal_reason": current.temporal_reason,
            "action_sampled": True,
            "held_for_display": False,
            "source_action_frame_index": int(frame_index),
        }
        self._last_action_payload = payload
        return action, anatomical_side, evidence, payload

    def analyze(
        self,
        image: np.ndarray,
        *,
        frame_index: int,
        timestamp: float,
    ) -> dict[str, Any]:
        detections = self.pose.detect(image)
        anonymous_candidates = _candidate_payload(
            detections,
            frame_index=frame_index,
            timestamp=timestamp,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            mirror_horizontal=self.mirror_horizontal,
        )
        self._candidate_history[int(frame_index)] = anonymous_candidates
        while len(self._candidate_history) > self.candidate_history_size:
            self._candidate_history.popitem(last=False)
        locked = self.lock.update(detections, image.shape)
        raw = locked.raw_result
        pose = raw.smoothed_pose if locked.usable_pose else None
        keypoints = pose.keypoints if pose is not None else None
        statuses = pose.statuses if pose is not None else None
        action_sampled = self._action_sample_due(
            timestamp=timestamp,
            person_ref=locked.person_ref,
            lock_epoch=locked.lock_epoch,
            track_state=locked.track_state,
            lock_state=locked.lock_state,
        )
        if action_sampled:
            action, anatomical_side, evidence, stable_action = (
                self._sample_action(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    locked=locked,
                    detections=detections,
                    keypoints=keypoints,
                    statuses=statuses,
                )
            )
        else:
            evidence = self._pose_quality(
                track_state=locked.track_state,
                keypoints=keypoints,
                statuses=statuses,
            )
            prior = self._last_action_payload
            action = (
                str(prior["raw_action"])
                if prior is not None
                else "transition"
            )
            anatomical_side = (
                str(prior["anatomical_side"])
                if prior is not None
                else "bilateral"
            )
            stable_action = (
                {
                    **prior,
                    "action_sampled": False,
                    "held_for_display": True,
                    "display_timestamp": round(float(timestamp), 6),
                }
                if prior is not None
                else {
                    "action": "transition",
                    "raw_action": "transition",
                    "anatomical_side": "bilateral",
                    "duration_seconds": 0.0,
                    "display_eligible": False,
                    "status": "uncertain",
                    "training_eligible": False,
                    "source_frame_indices": [],
                    "temporal_reason": "awaiting_first_action_sample",
                    "action_sampled": False,
                    "held_for_display": True,
                    "source_action_frame_index": None,
                    "display_timestamp": round(float(timestamp), 6),
                }
            )
        hands = self.hand.infer_frame(
            image,
            body_keypoints=keypoints,
            body_keypoint_statuses=statuses,
            person_ref=locked.person_ref,
            lock_epoch=locked.lock_epoch,
            frame_index=frame_index,
            timestamp=timestamp,
            source_video_sha256=self.stream_evidence_hash,
            recording_group_id=f"local_usb_session_{self.stream_evidence_hash[:12]}",
            track_state=locked.track_state,
            lock_state=locked.lock_state,
        )
        bbox = (
            [round(float(value), 2) for value in raw.detection.bbox]
            if locked.usable_pose and raw.detection is not None
            else None
        )
        frame = {
            "timestamp": round(float(timestamp), 6),
            "source_frame_index": int(frame_index),
            "person_ref": locked.person_ref,
            "lock_epoch": locked.lock_epoch,
            "track_state": locked.track_state,
            "lock_state": locked.lock_state,
            "candidate_person_count": len(detections),
            "action": action,
            "anatomical_side": anatomical_side,
            "observation_state": evidence["observation_state"],
            "detected_ratio": evidence["detected_ratio"],
            "predicted_ratio": evidence["predicted_ratio"],
            "interpolated_ratio": evidence["interpolated_ratio"],
            "missing_ratio": evidence["missing_ratio"],
            "direction_clear": evidence["direction_clear"],
            "required_joints_reliable": evidence[
                "required_joints_reliable"
            ],
            "keypoints": _safe_points(keypoints),
            "keypoint_statuses": (
                [str(value) for value in statuses] if statuses is not None else []
            ),
            "bbox": bbox,
            "person_confidence": (
                round(float(raw.detection.confidence), 5)
                if locked.usable_pose and raw.detection is not None
                else None
            ),
            "anonymous_candidates": anonymous_candidates,
            "switch_exposed": locked.switch_exposed,
            "awaiting_manual_relock": locked.awaiting_manual_relock,
            "evidence_kind": "real_yolov8_pose_local_usb",
            "action_sampled": action_sampled,
            "action_source_frame_index": stable_action[
                "source_action_frame_index"
            ],
        }
        return {
            "frame": frame,
            "hand_pose_frames": hands,
            "stable_action": stable_action,
            "body_model": {
                "path": str(Path(self.pose.model_path).name),
                "providers": self.pose.providers,
                "provider_status": self.pose.provider_status,
            },
            "hand_model": {
                "backend_state": self.hand.availability_status,
                "model_version": self.hand.model_version,
                "provider": "CPU",
                "reason": self.hand.reason,
            },
        }
