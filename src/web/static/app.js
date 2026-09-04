const COCO_EDGES = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15],
  [12, 14], [14, 16]
];

const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

const HAND_COLORS = {
  left: "#35b6d4",
  right: "#9c8cff",
};

const NORMAL_ACTIONS = new Set([
  "idle", "reach", "retract", "lift", "lower", "move", "carry",
  "place", "hold", "release", "rotate", "push", "pull",
]);

const state = {
  analysis: null,
  frames: [],
  handFrames: [],
  poseSegments: [],
  suppressedEvidence: [],
  evidenceTimeline: [],
  evidenceTimelineSource: "unavailable",
  allActionEvents: [],
  actionEvents: [],
  processSteps: [],
  timelineStart: 0,
  timelineEnd: 1,
  timelineDuration: 1,
  showBodyPose: true,
  showHandPose: true,
  bodyRendererMode: "classic",
  mode: "video",
  cameraEvidence: null,
  cameraSequence: null,
  cameraPendingSequence: null,
  cameraTransportEpoch: 0,
  cameraPollInFlight: false,
  cameraLastStatusCheck: 0,
  cameraState: "stopped",
  cameraPollTimer: null,
  cameraSessionId: null,
  cameraSelectionMode: false,
  selectedCameraCandidate: null,
  videoFrameCallbackId: null,
  videoAnimationFrameId: null,
  upload: null,
  preview: null,
  selectedPreviewCandidate: null,
  jobPollTimer: null,
  loadedJobId: null,
};

const CAMERA_ACTIVE_POLL_INTERVAL_MS = 50;
const CAMERA_IDLE_POLL_INTERVAL_MS = 400;
const CAMERA_STATUS_INTERVAL_MS = 250;

const $ = (id) => document.getElementById(id);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const escapeText = (value) => String(value ?? "");

function eventDuration(event) {
  if (!event) return 0;
  const stored = Number(event.duration_seconds);
  return Number.isFinite(stored)
    ? stored
    : Math.max(0, Number(event.end_time) - Number(event.start_time));
}

function normalizeHandState(record) {
  const value = String(record?.observation_state || "missing").toLowerCase();
  if (value === "detected") return "detected";
  if (["uncertain", "predicted", "interpolated"].includes(value)) return "uncertain";
  if (["lost", "off_frame"].includes(value)) return "lost";
  if (["unavailable", "not_configured"].includes(value)) return "unavailable";
  return "missing";
}

function currentHandBackendState() {
  const handLayer = layer("hand_pose") || layer("hand_pose_estimation");
  return String(
    state.analysis?.hand_model?.backend_state
    || handLayer?.status
    || "unavailable",
  ).toLowerCase();
}

function handObservationState(record, frame) {
  if (record) return normalizeHandState(record);
  if (frame && frame.track_state !== "tracked") return "lost";
  if (currentHandBackendState() !== "available") return "unavailable";
  return "missing";
}

function currentHandActionFeatureUse() {
  const explicit = (
    state.analysis?.hand_action_feature_use
    ?? state.analysis?.runtime?.hand_action_feature_use
    ?? state.analysis?.hand_model?.action_feature_use
  );
  const normalized = String(explicit ?? "").toLowerCase();
  const explicitlyConsumed = (
    explicit === true
    || normalized === "used"
    || normalized === "consumed"
    || normalized === "consumed_by_current_action_naming"
  );
  return explicitlyConsumed
    ? "consumed_by_current_action_naming"
    : "not_consumed_by_current_action_naming";
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const rest = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(3).padStart(6, "0")}`;
}

function layer(name) {
  return (state.analysis.layer_states || []).find((item) => item.layer === name);
}

function setChip(node, value) {
  node.textContent = value;
  node.className = `state-chip ${value}`;
}

function timelineGeometry(item, minimumWidth = 1.5) {
  const rawStart = Number(item?.start_time);
  const rawEnd = Number(item?.end_time);
  if (!Number.isFinite(rawStart) || !Number.isFinite(rawEnd)) return null;
  const start = clamp(rawStart, state.timelineStart, state.timelineEnd);
  const end = clamp(rawEnd, state.timelineStart, state.timelineEnd);
  if (end <= start) return null;
  const left = clamp(
    (start - state.timelineStart) / state.timelineDuration * 100,
    0,
    100,
  );
  const available = Math.max(0, 100 - left);
  const width = Math.min(
    available,
    Math.max(minimumWidth, (end - start) / state.timelineDuration * 100),
  );
  return {left, width};
}

function jumpToTime(timestamp) {
  const video = $("video");
  const requested = Number(timestamp);
  if (!Number.isFinite(requested)) return;
  const maximum = Number.isFinite(video.duration)
    ? video.duration
    : Math.max(state.timelineEnd, requested);
  video.currentTime = clamp(requested, 0, maximum);
  const playRequest = video.play();
  if (playRequest?.catch) playRequest.catch(() => {});
}

function activeMediaElement() {
  return state.mode === "camera" ? $("camera-view") : $("video");
}

function renderProcessTimeline() {
  const host = $("process-timeline");
  const processLayer = layer("process_reasoning");
  const layerStatus = processLayer?.status || "unavailable";
  setChip($("process-state"), layerStatus);
  $("process-reason").textContent = processLayer?.reason || "No process evidence.";
  host.replaceChildren();
  if (!state.processSteps.length) {
    const placeholder = document.createElement("div");
    placeholder.className = "process-placeholder";
    placeholder.textContent = "真实零件 / 交互 / 时序证据尚不完整，本轮不生成工序步骤。";
    host.appendChild(placeholder);
    return;
  }
  state.processSteps.forEach((step) => {
    const button = document.createElement("button");
    button.className = "process-step";
    button.dataset.start = step.start_time;
    button.dataset.end = step.end_time;
    button.textContent = `${step.process_name} · ${step.status}`;
    const geometry = timelineGeometry(step, 2);
    if (!geometry) return;
    button.style.width = `${geometry.width}%`;
    button.style.left = `${geometry.left}%`;
    button.addEventListener("click", () => jumpToTime(step.start_time));
    host.appendChild(button);
  });
}

function normalizeEvidenceState(item) {
  const explicit = String(
    item?.evidence_state || item?.timeline_state || item?.state || "",
  ).toLowerCase();
  if (["normal", "transition", "unknown", "uncertain", "lost"].includes(explicit)) {
    return explicit;
  }
  const action = String(item?.action || item?.action_name || "unknown").toLowerCase();
  const trackState = String(item?.track_state || "").toLowerCase();
  const lockState = String(item?.lock_state || "").toLowerCase();
  const observationState = String(item?.observation_state || "").toLowerCase();
  const gapKind = String(item?.gap_kind || "").toLowerCase();
  if (
    action === "lost"
    || ["lost", "off_frame", "awaiting_manual_relock"].includes(trackState)
    || ["lost", "off_frame", "awaiting_manual_relock"].includes(lockState)
    || ["lost", "off_frame"].includes(observationState)
  ) return "lost";
  if (gapKind === "bounded_uncertain_gap" || observationState === "missing") {
    return "uncertain";
  }
  if (action === "transition") return "transition";
  if (action === "unknown") return "unknown";
  if (NORMAL_ACTIONS.has(action)) return "normal";
  return "uncertain";
}

function deriveEvidenceTimeline(segments) {
  return segments.map((segment, index) => ({
    ...segment,
    evidence_interval_id: `pose-segment-fallback-${index + 1}`,
    evidence_state: normalizeEvidenceState(segment),
    source_segment_ids: asList(
      segment.source_segment_ids || segment.segment_id,
    ),
    stabilization_reason: (
      segment.stabilization_reason
      || "display_fallback_from_original_pose_segment"
    ),
    display_source: "pose_segments_fallback",
  }));
}

function renderEvidenceTimeline() {
  const host = $("evidence-timeline");
  host.replaceChildren();
  $("evidence-track-source").textContent = state.evidenceTimelineSource;
  if (!state.evidenceTimeline.length) {
    const placeholder = document.createElement("div");
    placeholder.className = "process-placeholder";
    placeholder.textContent = "No continuous evidence intervals are available.";
    host.appendChild(placeholder);
    return;
  }
  state.evidenceTimeline.forEach((interval, index) => {
    const geometry = timelineGeometry(interval, 0.7);
    if (!geometry) return;
    const evidenceState = normalizeEvidenceState(interval);
    const button = document.createElement("button");
    button.className = `evidence-interval ${evidenceState}`;
    button.dataset.evidenceId = (
      interval.evidence_interval_id
      || interval.timeline_event_id
      || `evidence-${index + 1}`
    );
    const action = String(
      interval.action || interval.action_name || evidenceState,
    ).toLowerCase();
    button.textContent = action === evidenceState
      ? evidenceState
      : `${evidenceState} · ${action}`;
    button.title = [
      `${evidenceState} / ${action}`,
      `${formatTime(interval.start_time)}–${formatTime(interval.end_time)}`,
      interval.stabilization_reason || interval.gap_kind || "observed evidence",
    ].join(" · ");
    button.style.width = `${geometry.width}%`;
    button.style.left = `${geometry.left}%`;
    button.addEventListener("click", () => jumpToTime(interval.start_time));
    host.appendChild(button);
  });
}

function renderActionTimeline() {
  const host = $("action-timeline");
  host.replaceChildren();
  if (!state.actionEvents.length) {
    const placeholder = document.createElement("div");
    placeholder.className = "process-placeholder";
    placeholder.textContent = "No stable action events in this sampled window.";
    host.appendChild(placeholder);
    return;
  }
  state.actionEvents.forEach((event) => {
    const button = document.createElement("button");
    button.className = `action-event ${event.action}`;
    button.dataset.eventId = event.action_event_id;
    button.textContent = `${event.action}\n${eventDuration(event).toFixed(1)}s`;
    button.title = `${event.action} ${formatTime(event.start_time)}–${formatTime(event.end_time)}`;
    const geometry = timelineGeometry(event);
    if (!geometry) return;
    button.style.width = `${geometry.width}%`;
    button.style.left = `${geometry.left}%`;
    button.addEventListener("click", () => jumpToTime(event.start_time));
    host.appendChild(button);
  });
}

function renderLayerRows() {
  const entries = [
    ["object-layer", "Object perception", layer("object_perception")],
    ["interaction-layer", "Interaction fusion", layer("interaction_fusion")],
  ];
  entries.forEach(([id, label, item]) => {
    const host = $(id);
    host.replaceChildren();
    const name = document.createElement("span");
    name.textContent = label;
    const chip = document.createElement("span");
    setChip(chip, item?.status || "unavailable");
    host.append(name, chip);
  });

  renderProviderStates();
  const host = $("model-states");
  host.replaceChildren();
  (state.analysis.layer_states || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "model-row";
    const copy = document.createElement("span");
    copy.textContent = escapeText(item.layer);
    const detail = document.createElement("small");
    const version = item.model_version ? ` · ${item.model_version}` : "";
    detail.textContent = `${escapeText(item.reason)}${version}`;
    copy.appendChild(detail);
    const chip = document.createElement("span");
    setChip(chip, item.status);
    row.append(copy, chip);
    host.appendChild(row);
  });

  const handLayer = layer("hand_pose") || layer("hand_pose_estimation");
  const handVersion = handLayer?.model_version
    || state.analysis.hand_model?.version
    || state.analysis.hand_model?.model_version
    || "not_configured";
  $("hand-model-version").textContent = handVersion;
  const handModel = state.analysis.hand_model || {};
  setChip($("hand-backend-state"), currentHandBackendState());
  const runtime = state.analysis.runtime || {};
  $("hand-runtime-provider").textContent = (
    handModel.provider || runtime.hand_provider || "unavailable"
  );
  $("hand-gpu-status").textContent = (
    handModel.hand_gpu_status
    || runtime.hand_gpu_status
    || "not_configured"
  );
  $("hand-backend-mode").textContent = handModel.backend_mode || "not_configured";
  $("hand-quality-gate-version").textContent = (
    handModel.quality_gate_version || "not_configured"
  );
  const mean = Number(runtime.mean_hand_inference_ms);
  const p50 = Number(runtime.hand_inference_p50_ms);
  const p95 = Number(runtime.hand_inference_p95_ms);
  $("hand-inference-percentiles").textContent = [mean, p50, p95].every(
    Number.isFinite,
  )
    ? `${mean.toFixed(2)} / ${p50.toFixed(2)} / ${p95.toFixed(2)} ms`
    : "not_evaluable";
  $("hand-action-feature-use").textContent = currentHandActionFeatureUse();
}

function updateCurrentHandEvidenceSummary(hands) {
  const records = [hands?.left, hands?.right].filter(Boolean);
  const drawable = records.filter((record) => (
    ["detected", "uncertain"].includes(normalizeHandState(record))
    && Array.isArray(record.landmarks)
    && record.landmarks.length === 21
  ));
  $("hand-drawable-geometry").textContent = drawable.length
    ? `${drawable.length} side(s) with 21 real points`
    : "missing / no drawable 21-point geometry";
  const warnings = records.flatMap((record) => (
    Array.isArray(record.association_warnings)
      ? record.association_warnings
      : record.association_warning
      ? [record.association_warning]
      : []
  )).filter(Boolean);
  $("hand-association-warning").textContent = warnings.length
    ? [...new Set(warnings.map(String))].join(", ")
    : "none observed";
}

function providerRow(label, provider, fallbackActive = false, fallbackReason = null) {
  const row = document.createElement("div");
  row.className = "model-row provider-row";
  const copy = document.createElement("span");
  copy.textContent = label;
  const detail = document.createElement("small");
  detail.textContent = fallbackActive
    ? `fallback · ${fallbackReason || "reason unavailable"}`
    : "active runtime";
  copy.appendChild(detail);
  const chip = document.createElement("span");
  setChip(chip, provider || "unavailable");
  row.append(copy, chip);
  return row;
}

function renderProviderStates(bodyModel = null, handModel = null) {
  const host = $("provider-states");
  if (!host) return;
  const runtime = state.analysis?.runtime || {};
  const bodyStatus = bodyModel?.provider_status
    || runtime.pose_provider_status
    || {};
  const bodyProvider = bodyStatus.active_provider
    || bodyModel?.providers?.[0]
    || runtime.pose_providers?.[0]
    || "unavailable";
  const currentHand = handModel || state.analysis?.hand_model || {};
  const handProvider = currentHand.provider
    || currentHand.runtime_provider
    || (currentHandBackendState() === "available" ? "CPU" : "unavailable");
  host.replaceChildren(
    providerRow(
      "Body Pose provider",
      bodyProvider,
      bodyStatus.fallback_active === true,
      bodyStatus.fallback_reason,
    ),
    providerRow("Hand Pose provider", handProvider, false, null),
  );
}

function nearestFrame(time) {
  if (!state.frames.length) return null;
  let best = state.frames[0];
  let delta = Math.abs(best.timestamp - time);
  for (const frame of state.frames) {
    const candidate = Math.abs(frame.timestamp - time);
    if (candidate < delta) {
      best = frame;
      delta = candidate;
    }
  }
  const sampleFps = Number(
    state.analysis.source_video?.analysis_window?.sample_fps
    || state.analysis.runtime?.analysis_fps
    || 2
  );
  const windowSeconds = 0.75 / Math.max(0.1, sampleFps);
  return delta <= Math.max(0.35, windowSeconds) ? best : null;
}

function currentEvent(time) {
  return state.actionEvents.find((item) => time >= item.start_time && time < item.end_time) || null;
}

function currentEvidence(time) {
  return state.evidenceTimeline.find(
    (item) => time >= Number(item.start_time) && time < Number(item.end_time),
  ) || null;
}

function nearestHands(frame, time) {
  const result = {left: null, right: null};
  if (!frame) return result;
  const frameIndex = Number(frame.frame_index);
  const hasFrameIndex = Number.isInteger(frameIndex) && frameIndex >= 0;
  const sampleFps = Number(
    state.analysis.source_video?.analysis_window?.sample_fps
    || state.analysis.runtime?.analysis_fps
    || 2
  );
  const tolerance = Math.max(0.08, 0.75 / Math.max(0.1, sampleFps));
  ["left", "right"].forEach((side) => {
    let best = null;
    let bestDelta = Number.POSITIVE_INFINITY;
    state.handFrames.forEach((record) => {
      const recordSide = record.anatomical_side || record.side;
      if (recordSide !== side) return;
      if (String(record.person_ref) !== String(frame.person_ref)) return;
      if (String(record.lock_epoch) !== String(frame.lock_epoch)) return;
      const recordFrameIndex = Number(record.frame_index);
      if (
        hasFrameIndex
        && Number.isInteger(recordFrameIndex)
        && recordFrameIndex !== frameIndex
      ) return;
      const delta = Math.abs(Number(record.timestamp) - time);
      if (delta < bestDelta) {
        best = record;
        bestDelta = delta;
      }
    });
    if (best && bestDelta <= tolerance) result[side] = best;
  });
  return result;
}

function landmarkPoint(raw) {
  if (Array.isArray(raw)) return [Number(raw[0]), Number(raw[1])];
  if (!raw || typeof raw !== "object") return null;
  const x = raw.source_x ?? raw.x;
  const y = raw.source_y ?? raw.y;
  if (!Number.isFinite(Number(x)) || !Number.isFinite(Number(y))) return null;
  return [Number(x), Number(y)];
}

function drawHand(context, record, side, point) {
  const observation = normalizeHandState(record);
  if (["missing", "lost", "unavailable"].includes(observation)) return;
  const landmarks = Array.isArray(record?.landmarks) ? record.landmarks : [];
  if (!landmarks.length) return;
  const qualityState = String(record?.quality_state || "not_observed").toLowerCase();
  const qualified = (
    observation === "detected"
    && qualityState === "qualified"
    && record?.action_feature_eligible === true
  );
  const color = HAND_COLORS[side];
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = qualified ? 2 : 1.5;
  context.globalAlpha = qualified ? 0.95 : 0.5;
  context.setLineDash(qualified ? [] : [4, 3]);
  HAND_EDGES.forEach(([a, b]) => {
    const left = landmarkPoint(landmarks[a]);
    const right = landmarkPoint(landmarks[b]);
    if (!left || !right) return;
    const [x1, y1] = point(left);
    const [x2, y2] = point(right);
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
  });
  context.setLineDash([]);
  landmarks.forEach((raw) => {
    const sourcePoint = landmarkPoint(raw);
    if (!sourcePoint) return;
    const [x, y] = point(sourcePoint);
    context.beginPath();
    context.arc(x, y, qualified ? 2.5 : 2, 0, Math.PI * 2);
    context.fill();
  });
  context.globalAlpha = 1;
}

function drawEvidenceBodyPose(context, frame, point) {
  let segmentCount = 0;
  (frame.anonymous_candidates || []).forEach((candidate) => {
    const [x1, y1] = point(candidate.bbox);
    const [x2, y2] = point([candidate.bbox[2], candidate.bbox[3]]);
    context.setLineDash([5, 5]);
    context.strokeStyle = "rgba(231,180,90,.50)";
    context.lineWidth = 1;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  });
  context.setLineDash([]);
  if (frame.track_state !== "tracked" || !frame.keypoints?.length) return 0;
  if (frame.bbox) {
    const [x1, y1] = point(frame.bbox);
    const [x2, y2] = point([frame.bbox[2], frame.bbox[3]]);
    context.strokeStyle = "#35d49a";
    context.lineWidth = 2;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  COCO_EDGES.forEach(([a, b]) => {
    const left = frame.keypoints[a];
    const right = frame.keypoints[b];
    const leftState = frame.keypoint_statuses[a];
    const rightState = frame.keypoint_statuses[b];
    if (!left || !right || left[0] === null || right[0] === null) return;
    if (["missing", "uncertain", "rejected"].includes(leftState) ||
        ["missing", "uncertain", "rejected"].includes(rightState)) return;
    const [x1, y1] = point(left);
    const [x2, y2] = point(right);
    const auxiliary = [leftState, rightState].some(
      (value) => ["predicted", "interpolated"].includes(value),
    );
    context.strokeStyle = auxiliary ? "#e7b45a" : "#35d49a";
    context.lineWidth = auxiliary ? 2 : 3;
    context.setLineDash(auxiliary ? [6, 4] : []);
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
    segmentCount += 1;
  });
  context.setLineDash([]);
  frame.keypoints.forEach((raw, index) => {
    if (
      !raw
      || raw[0] === null
      || ["missing", "uncertain", "rejected"].includes(frame.keypoint_statuses[index])
    ) return;
    const [x, y] = point(raw);
    context.fillStyle = frame.keypoint_statuses[index] === "detected"
      ? "#eafdf6"
      : "#e7b45a";
    context.beginPath();
    context.arc(x, y, 3, 0, Math.PI * 2);
    context.fill();
  });
  return segmentCount;
}

function drawFrame(frame, hands = {left: null, right: null}) {
  const canvas = $("pose-canvas");
  const renderMetrics = {
    body_segment_count: 0,
    body_raw_point_count: 0,
    body_derived_visual_only_point_count: 0,
    body_geometry_state: "missing",
  };
  const media = activeMediaElement();
  const rect = media.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, rect.width, rect.height);
  if (!frame) {
    canvas.dataset.bodySegmentCount = "0";
    canvas.dataset.bodyGeometryState = "missing";
    return renderMetrics;
  }

  const sourceWidth = state.mode === "camera"
    ? Number(state.cameraEvidence?.width) || 1
    : state.analysis.source_video.width;
  const sourceHeight = state.mode === "camera"
    ? Number(state.cameraEvidence?.height) || 1
    : state.analysis.source_video.height;
  const sourceRatio = sourceWidth / sourceHeight;
  const displayRatio = rect.width / rect.height;
  let shownWidth = rect.width;
  let shownHeight = rect.height;
  let offsetX = 0;
  let offsetY = 0;
  if (sourceRatio > displayRatio) {
    shownHeight = rect.width / sourceRatio;
    offsetY = (rect.height - shownHeight) / 2;
  } else {
    shownWidth = rect.height * sourceRatio;
    offsetX = (rect.width - shownWidth) / 2;
  }
  const sx = shownWidth / sourceWidth;
  const sy = shownHeight / sourceHeight;
  const point = (raw) => [offsetX + raw[0] * sx, offsetY + raw[1] * sy];

  if (state.showBodyPose) {
    if (
      state.bodyRendererMode === "classic"
      && window.ClassicBodyPoseRenderer
    ) {
      const classicMetrics = window.ClassicBodyPoseRenderer.render(
        context,
        frame,
        point,
      );
      renderMetrics.body_segment_count = classicMetrics.segment_count;
      renderMetrics.body_raw_point_count = classicMetrics.raw_point_count;
      renderMetrics.body_derived_visual_only_point_count = (
        classicMetrics.derived_visual_only_point_count
      );
    } else {
      renderMetrics.body_segment_count = drawEvidenceBodyPose(
        context,
        frame,
        point,
      );
    }
  }
  renderMetrics.body_geometry_state = (
    frame.track_state !== "tracked"
      ? "lost"
      : (
        !state.showBodyPose
          ? "disabled"
          : (
            renderMetrics.body_segment_count > 0
              ? "visible"
              : "insufficient_visible_body_geometry"
          )
      )
  );
  canvas.dataset.bodySegmentCount = String(renderMetrics.body_segment_count);
  canvas.dataset.bodyGeometryState = renderMetrics.body_geometry_state;
  if (
    state.mode === "camera"
    && (state.cameraSelectionMode || frame.awaiting_manual_relock)
  ) {
    const candidates = frame.anonymous_candidates || [];
    candidates.forEach((candidate, index) => {
      if (!Array.isArray(candidate.bbox) || candidate.bbox.length !== 4) return;
      const [x1, y1] = point(candidate.bbox);
      const [x2, y2] = point([candidate.bbox[2], candidate.bbox[3]]);
      const selected = (
        candidate.candidate_token
        && candidate.candidate_token === state.selectedCameraCandidate?.candidate_token
      );
      context.strokeStyle = selected ? "#35d49a" : ["#35b6d4", "#9c8cff", "#e7b45a"][index % 3];
      context.fillStyle = selected ? "rgba(53,212,154,.16)" : "rgba(53,182,212,.08)";
      context.lineWidth = selected ? 4 : 2;
      context.setLineDash(selected ? [] : [7, 4]);
      context.fillRect(x1, y1, x2 - x1, y2 - y1);
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      context.setLineDash([]);
      const label = `${candidate.candidate_id || `C${index + 1}`} · ${Number(candidate.confidence || 0).toFixed(2)}`;
      context.font = "11px ui-monospace, monospace";
      context.fillStyle = selected ? "#9ff4d1" : "#dff8ff";
      context.fillText(label, x1 + 5, Math.max(13, y1 - 5));
    });
    const selected = state.selectedCameraCandidate;
    if (
      selected
      && Array.isArray(selected.bbox)
      && !candidates.some(
        (candidate) => candidate.candidate_token === selected.candidate_token
      )
    ) {
      const [x1, y1] = point(selected.bbox);
      const [x2, y2] = point([selected.bbox[2], selected.bbox[3]]);
      context.strokeStyle = "#35d49a";
      context.lineWidth = 4;
      context.setLineDash([]);
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
  }
  if (frame.track_state !== "tracked") return renderMetrics;
  if (state.showHandPose) {
    drawHand(context, hands.left, "left", point);
    drawHand(context, hands.right, "right", point);
  }
  return renderMetrics;
}

function cameraHands(records) {
  const result = {left: null, right: null};
  (records || []).forEach((record) => {
    const side = String(record?.anatomical_side || "").toLowerCase();
    if (side === "left" || side === "right") result[side] = record;
  });
  return result;
}

function renderCameraEvidence(packet) {
  const evidence = packet?.evidence;
  const frame = evidence?.frame;
  if (!frame) return;
  state.cameraEvidence = packet;
  $("pose-canvas").dataset.cameraSequence = String(packet.sequence ?? "");
  $("pose-canvas").dataset.cameraCandidateCount = String(
    frame.anonymous_candidates?.length || 0
  );
  const hands = cameraHands(evidence.hand_pose_frames);
  updateCurrentHandEvidenceSummary(hands);
  const stable = evidence.stable_action || {};
  renderProviderStates(evidence.body_model, evidence.hand_model);
  const evidenceState = normalizeEvidenceState(frame);
  $("clock").textContent = formatTime(frame.timestamp);
  $("current-action").textContent = stable.display_eligible
    ? stable.action
    : "transition";
  $("current-side").textContent = stable.anatomical_side || "—";
  $("current-action-duration").textContent = (
    `${Number(stable.duration_seconds || 0).toFixed(2)} s`
  );
  $("action-duration").textContent = (
    `${Number(stable.duration_seconds || 0).toFixed(2)} s`
  );
  $("person-ref").textContent = frame.person_ref || "unlocked";
  $("lock-epoch").textContent = frame.lock_epoch ?? 0;
  $("candidate-count").textContent = frame.candidate_person_count ?? 0;
  $("evidence-time").textContent = `${Number(frame.timestamp).toFixed(2)} s`;
  $("frame-evidence").textContent = [
    `${evidenceState} evidence`,
    `${frame.observation_state} pose`,
    `live frame ${frame.source_frame_index}`,
  ].join(" · ");
  setChip($("current-evidence-state"), evidenceState);
  setChip($("lock-state"), frame.lock_state || "uncertain");
  updateManualRelockUI(frame);
  $("lost-banner").classList.toggle("hidden", frame.track_state === "tracked");
  ["detected", "predicted", "interpolated", "missing"].forEach((name) => {
    const value = clamp(Number(frame[`${name}_ratio`]) || 0, 0, 1);
    $(`${name}-value`).textContent = value.toFixed(2);
    $(`${name}-bar`).style.width = `${value * 100}%`;
  });
  updateHandSideReadout("left", hands.left, frame);
  updateHandSideReadout("right", hands.right, frame);
  $("source-segment-ids").textContent = (
    stable.source_frame_indices?.length
      ? `live frames ${stable.source_frame_indices[0]}–${stable.source_frame_indices.at(-1)}`
      : "no stable source span"
  );
  $("stabilization-reason").textContent = (
    stable.temporal_reason || "live causal observation"
  );
  $("fragment-reason").textContent = [
    stable.temporal_reason || "live causal observation",
    "No object or true-grasp evidence is inferred.",
  ].join(" · ");
  const renderMetrics = drawFrame(frame, hands);
  if (
    state.showBodyPose
    && frame.track_state === "tracked"
    && renderMetrics.body_segment_count === 0
  ) {
    $("frame-evidence").textContent += " Â· insufficient_visible_body_geometry";
  }
}

function updateManualRelockUI(frame = null) {
  const controls = $("manual-relock-controls");
  const cameraActive = state.mode === "camera";
  controls.classList.toggle("hidden", !cameraActive);
  if (!cameraActive) return;
  const awaiting = Boolean(frame?.awaiting_manual_relock);
  const selected = state.selectedCameraCandidate;
  $("camera-select-person-button").textContent = (
    awaiting ? "重新选择人物" : "选择人物"
  );
  $("camera-confirm-relock-button").disabled = !selected;
  $("camera-cancel-relock-button").disabled = !(
    state.cameraSelectionMode || selected || awaiting
  );
  if (selected) {
    $("manual-relock-message").textContent = (
      `已选择 ${selected.candidate_id}（置信度 ${Number(selected.confidence || 0).toFixed(2)}），请明确确认。`
    );
  } else if (state.cameraSelectionMode) {
    $("manual-relock-message").textContent = "请点击当前画面中的一个匿名候选框。";
  } else if (awaiting) {
    $("manual-relock-message").textContent = "当前锁定已丢失；必须人工选择候选，系统不会自动换人。";
  } else {
    $("manual-relock-message").textContent = "可显式选择或重新选择匿名人物。";
  }
}

function beginCameraPersonSelection() {
  if (state.mode !== "camera" || state.cameraState !== "live") {
    $("manual-relock-message").textContent = "Camera live 后才能选择人物。";
    return;
  }
  state.cameraSelectionMode = true;
  state.selectedCameraCandidate = null;
  $("pose-canvas").classList.add("selecting-person");
  updateManualRelockUI(state.cameraEvidence?.evidence?.frame);
  if (state.cameraEvidence) renderCameraEvidence(state.cameraEvidence);
}

function selectCameraCandidateFromPoint(event) {
  if (!state.cameraSelectionMode || state.mode !== "camera") return;
  const frame = state.cameraEvidence?.evidence?.frame;
  const candidates = frame?.anonymous_candidates || [];
  const canvas = $("pose-canvas");
  const rect = canvas.getBoundingClientRect();
  const sourceWidth = Number(state.cameraEvidence?.width) || 0;
  const sourceHeight = Number(state.cameraEvidence?.height) || 0;
  if (!rect.width || !rect.height || !sourceWidth || !sourceHeight) return;
  const sourceRatio = sourceWidth / sourceHeight;
  const displayRatio = rect.width / rect.height;
  let shownWidth = rect.width;
  let shownHeight = rect.height;
  let offsetX = 0;
  let offsetY = 0;
  if (sourceRatio > displayRatio) {
    shownHeight = rect.width / sourceRatio;
    offsetY = (rect.height - shownHeight) / 2;
  } else {
    shownWidth = rect.height * sourceRatio;
    offsetX = (rect.width - shownWidth) / 2;
  }
  const localX = event.clientX - rect.left;
  const localY = event.clientY - rect.top;
  if (
    localX < offsetX || localX > offsetX + shownWidth
    || localY < offsetY || localY > offsetY + shownHeight
  ) return;
  const x = (localX - offsetX) * sourceWidth / shownWidth;
  const y = (localY - offsetY) * sourceHeight / shownHeight;
  const hits = candidates.filter((candidate) => (
    Array.isArray(candidate.bbox)
    && candidate.bbox.length === 4
    && candidate.bbox[0] <= x && x <= candidate.bbox[2]
    && candidate.bbox[1] <= y && y <= candidate.bbox[3]
  ));
  if (!hits.length) {
    state.selectedCameraCandidate = null;
  } else {
    state.selectedCameraCandidate = hits.sort((left, right) => (
      (left.bbox[2] - left.bbox[0]) * (left.bbox[3] - left.bbox[1])
      - (right.bbox[2] - right.bbox[0]) * (right.bbox[3] - right.bbox[1])
    ))[0];
  }
  updateManualRelockUI(frame);
  renderCameraEvidence(state.cameraEvidence);
}

async function confirmCameraRelock() {
  const selected = state.selectedCameraCandidate;
  if (!selected) return;
  try {
    const result = await cameraRequest("/api/camera/relock", {
      session_id: selected.session_id,
      frame_sequence: selected.frame_sequence,
      candidate_token: selected.candidate_token,
    });
    state.cameraSelectionMode = false;
    state.selectedCameraCandidate = null;
    $("pose-canvas").classList.remove("selecting-person");
    $("manual-relock-message").textContent = (
      `${result.selected_candidate_id || "候选"} 已提交；正在下一真实帧重新验证并建立新匿名锁定。`
    );
  } catch (error) {
    $("manual-relock-message").textContent = error.message;
    state.selectedCameraCandidate = null;
  }
  updateManualRelockUI(state.cameraEvidence?.evidence?.frame);
}

async function cancelCameraRelock() {
  const selected = state.selectedCameraCandidate;
  try {
    if (state.cameraSessionId) {
      await cameraRequest("/api/camera/relock/cancel", {
        session_id: state.cameraSessionId,
        candidate_token: selected?.candidate_token || null,
      });
    }
  } catch (error) {
    $("manual-relock-message").textContent = error.message;
  }
  state.cameraSelectionMode = false;
  state.selectedCameraCandidate = null;
  $("pose-canvas").classList.remove("selecting-person");
  updateManualRelockUI(state.cameraEvidence?.evidence?.frame);
}

async function cameraRequest(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  const result = await response.json();
  if (!response.ok && response.status !== 409) {
    throw new Error(result.message || `Camera request failed (${response.status})`);
  }
  return result;
}

function showVideoMode() {
  state.mode = "video";
  state.cameraTransportEpoch += 1;
  state.cameraEvidence = null;
  state.cameraSequence = null;
  state.cameraPendingSequence = null;
  state.cameraSelectionMode = false;
  state.selectedCameraCandidate = null;
  $("pose-canvas").classList.remove("selecting-person");
  $("manual-relock-controls").classList.add("hidden");
  const cameraView = $("camera-view");
  cameraView.onload = null;
  cameraView.removeAttribute("src");
  $("video").classList.remove("hidden");
  $("camera-view").classList.add("hidden");
  $("video-mode-button").classList.add("active");
  $("camera-mode-button").classList.remove("active");
  $("video-title").textContent = state.analysis.source_video.path;
  updateEvidence(nearestFrame($("video").currentTime), $("video").currentTime);
}

function showCameraMode() {
  state.mode = "camera";
  state.cameraTransportEpoch += 1;
  state.cameraEvidence = null;
  state.cameraSequence = null;
  state.cameraPendingSequence = null;
  state.cameraSelectionMode = false;
  state.selectedCameraCandidate = null;
  $("video").pause();
  $("video").classList.add("hidden");
  $("camera-view").classList.remove("hidden");
  $("video-mode-button").classList.remove("active");
  $("camera-mode-button").classList.add("active");
  $("video-title").textContent = "本机 USB Camera · 技术验证（不保存录像）";
  $("manual-relock-controls").classList.remove("hidden");
  updateManualRelockUI();
  drawFrame(null);
}

async function startCamera() {
  showCameraMode();
  $("camera-mode-button").disabled = true;
  try {
    const status = await cameraRequest("/api/camera/start");
    state.cameraSessionId = status.session_id || null;
    setChip($("camera-state"), status.state);
    $("camera-stop-button").disabled = !status.worker_alive;
  } catch (error) {
    setChip($("camera-state"), "error");
    $("frame-evidence").textContent = error.message;
  } finally {
    $("camera-mode-button").disabled = false;
  }
}

async function stopCamera({returnToVideo = false} = {}) {
  $("camera-stop-button").disabled = true;
  try {
    const status = await cameraRequest("/api/camera/stop");
    setChip($("camera-state"), status.state);
  } catch (error) {
    setChip($("camera-state"), "error");
    $("frame-evidence").textContent = error.message;
  }
  state.cameraTransportEpoch += 1;
  state.cameraEvidence = null;
  state.cameraSequence = null;
  state.cameraPendingSequence = null;
  state.cameraSessionId = null;
  state.cameraSelectionMode = false;
  state.selectedCameraCandidate = null;
  $("pose-canvas").classList.remove("selecting-person");
  $("camera-view").onload = null;
  $("camera-view").removeAttribute("src");
  drawFrame(null);
  if (returnToVideo) showVideoMode();
}

function scheduleCameraPoll(delayMs) {
  if (state.cameraPollTimer !== null) {
    window.clearTimeout(state.cameraPollTimer);
  }
  state.cameraPollTimer = window.setTimeout(pollCamera, delayMs);
}

async function acknowledgeDisplayedCameraSequence(sequence) {
  try {
    await cameraRequest("/api/camera/display-ack", {sequence});
  } catch (_error) {
    // A bounded cache may evict an already-rendered frame before the local
    // acknowledgement arrives. This must not clear visible real evidence.
  }
}

function loadAtomicCameraPacket(envelope) {
  const sequence = Number(envelope?.sequence);
  const frameSequence = Number(envelope?.transport?.frame_sequence);
  const evidenceSequence = Number(envelope?.transport?.evidence_sequence);
  if (
    !Number.isInteger(sequence)
    || sequence !== frameSequence
    || sequence !== evidenceSequence
    || envelope?.transport?.atomic !== true
    || typeof envelope?.jpeg_base64 !== "string"
  ) {
    $("frame-evidence").textContent = "Camera frame/evidence sequence mismatch.";
    setChip($("camera-state"), "error");
    return;
  }
  const epoch = state.cameraTransportEpoch;
  state.cameraPendingSequence = sequence;
  const cameraView = $("camera-view");
  cameraView.onload = () => {
    if (
      state.mode !== "camera"
      || epoch !== state.cameraTransportEpoch
      || state.cameraPendingSequence !== sequence
    ) return;
    state.cameraSequence = sequence;
    state.cameraPendingSequence = null;
    state.cameraEvidence = envelope.evidence;
    renderCameraEvidence(envelope.evidence);
    acknowledgeDisplayedCameraSequence(sequence);
  };
  cameraView.src = `data:image/jpeg;base64,${envelope.jpeg_base64}`;
}

async function pollCamera() {
  if (state.cameraPollInFlight) return;
  state.cameraPollInFlight = true;
  try {
    const now = performance.now();
    let status = null;
    if (
      now - state.cameraLastStatusCheck >= CAMERA_STATUS_INTERVAL_MS
      || state.cameraState !== "live"
    ) {
      const response = await fetch("/api/camera/status", {cache: "no-store"});
      status = await response.json();
      state.cameraLastStatusCheck = now;
      state.cameraState = status.state;
      state.cameraSessionId = status.session_id || state.cameraSessionId;
      setChip($("camera-state"), status.state);
      $("camera-stop-button").disabled = !status.worker_alive;
      if (
        state.mode === "camera"
        && ["no_device", "permission_denied", "error"].includes(status.state)
      ) {
        const message = status.last_error?.message || "USB Camera unavailable.";
        $("frame-evidence").textContent = message;
        $("lost-banner").classList.remove("hidden");
        drawFrame(null);
      }
    }
    if (
      state.mode === "camera"
      && state.cameraState === "live"
      && !state.cameraSelectionMode
    ) {
      if (
        state.cameraPendingSequence !== null
        && state.cameraPendingSequence !== state.cameraSequence
      ) return;
      const highWater = Math.max(
        Number(state.cameraSequence) || 0,
        Number(state.cameraPendingSequence) || 0,
      );
      const packetResponse = await fetch(
        `/api/camera/packet?after_sequence=${encodeURIComponent(highWater)}`,
        {cache: "no-store"},
      );
      if (packetResponse.status === 200) {
        const envelope = await packetResponse.json();
        if (Number(envelope.sequence) > highWater) {
          loadAtomicCameraPacket(envelope);
        }
      }
    }
  } catch (_error) {
    setChip($("camera-state"), "error");
  } finally {
    state.cameraPollInFlight = false;
    scheduleCameraPoll(
      state.mode === "camera"
        ? CAMERA_ACTIVE_POLL_INTERVAL_MS
        : CAMERA_IDLE_POLL_INTERVAL_MS,
    );
  }
}

function startVideoOverlayClock(video, update) {
  if (typeof video.requestVideoFrameCallback === "function") {
    const onVideoFrame = () => {
      if (state.mode === "video") update();
      state.videoFrameCallbackId = video.requestVideoFrameCallback(onVideoFrame);
    };
    state.videoFrameCallbackId = video.requestVideoFrameCallback(onVideoFrame);
    return "requestVideoFrameCallback";
  }
  let lastRenderAt = 0;
  const onAnimationFrame = (now) => {
    if (
      state.mode === "video"
      && (
        video.seeking
        || !video.paused
        || now - lastRenderAt >= 100
      )
    ) {
      update();
      lastRenderAt = now;
    }
    state.videoAnimationFrameId = window.requestAnimationFrame(onAnimationFrame);
  };
  state.videoAnimationFrameId = window.requestAnimationFrame(onAnimationFrame);
  return "requestAnimationFrame";
}

function asList(value) {
  if (Array.isArray(value)) return value;
  if (value === null || value === undefined || value === "") return [];
  return String(value).split(";").map((item) => item.trim()).filter(Boolean);
}

function sourceSegmentsFor(event) {
  const sourceIds = new Set(asList(event?.source_segment_ids));
  return state.poseSegments.filter((segment) => sourceIds.has(segment.segment_id));
}

function handSummary(record, side) {
  const observation = normalizeHandState(record);
  const quality = record?.quality_state || "not_observed";
  const eligibility = record?.action_feature_eligible === true
    ? "feature eligible"
    : "feature not eligible";
  const count = Array.isArray(record?.landmarks)
    ? record.landmarks.length
    : Number(record?.landmark_count) || 0;
  return `${side} ${observation} · ${quality} · ${eligibility} · ${count} points`;
}

function updateHandSideReadout(side, record, frame) {
  const observation = handObservationState(record, frame);
  const quality = String(
    record?.quality_state
    || (observation === "lost" ? "lost" : "not_observed"),
  ).toLowerCase();
  const validation = String(
    record?.validation_state || "not_evaluable",
  ).toLowerCase();
  const eligible = record?.action_feature_eligible === true;
  setChip($(`${side}-hand-observation`), observation);
  setChip($(`${side}-hand-quality`), quality);
  setChip($(`${side}-hand-validation`), validation);
  setChip(
    $(`${side}-hand-eligible`),
    eligible ? "eligible" : "not_eligible",
  );
  const landmarkCount = Array.isArray(record?.landmarks)
    ? record.landmarks.length
    : Number(record?.landmark_count) || 0;
  $(`${side}-hand-binding`).textContent = record
    ? [
      `person_ref=${record.person_ref ?? "unknown"}`,
      `lock_epoch=${record.lock_epoch ?? "unknown"}`,
      `frame_index=${record.frame_index ?? "unknown"}`,
      `anatomical_side=${record.anatomical_side ?? "unknown"}`,
      `landmark_count=${landmarkCount}`,
    ].join(" · ")
    : "no bound record";
  const reasons = [
    ...asList(record?.quality_reasons),
    ...asList(record?.feature_eligibility_reasons),
  ];
  const uniqueReasons = [...new Set(reasons)];
  $(`${side}-hand-quality-reasons`).textContent = uniqueReasons.length
    ? uniqueReasons.join(" · ")
    : record?.reason
    || (observation === "unavailable"
      ? "hand backend unavailable"
      : observation === "lost"
      ? "person tracking evidence lost"
      : "no quality-qualified hand evidence");
}

function updateEventEvidence(event, evidence, hands) {
  const lineage = event || evidence;
  const sourceIds = asList(lineage?.source_segment_ids);
  $("source-segment-ids").textContent = sourceIds.length ? sourceIds.join(", ") : "—";
  const segments = sourceSegmentsFor(lineage);
  const explicitBodyPoints = asList(
    lineage?.body_points_used
    || lineage?.body_keypoints_used
    || lineage?.evidence_body_points
  );
  const segmentBodyPoints = segments.flatMap((segment) => asList(
    segment.body_points_used || segment.body_keypoints_used || segment.required_joints
  ));
  const bodyPoints = [...new Set([...explicitBodyPoints, ...segmentBodyPoints])];
  $("body-points-used").textContent = bodyPoints.length
    ? bodyPoints.join(", ")
    : "未记录；查看原始Pose证据";
  $("hand-evidence-detail").textContent = [
    handSummary(hands.left, "left"),
    handSummary(hands.right, "right"),
  ].join(" / ");

  if (lineage) {
    const visibility = [
      `detected ${Number(lineage.detected_ratio || 0).toFixed(2)}`,
      `predicted ${Number(lineage.predicted_ratio || 0).toFixed(2)}`,
      `missing ${Number(lineage.missing_ratio || 0).toFixed(2)}`,
    ];
    $("event-visibility").textContent = visibility.join(" · ");
    const spanSeconds = eventDuration(lineage);
    const observedSupportSeconds = Number(lineage.observed_support_seconds);
    const observedSupportRatio = Number(lineage.observed_support_ratio);
    const supportFragmentCount = Number(lineage.support_fragment_count);
    const boundedGapSeconds = Number(lineage.bounded_gap_seconds);
    const maximumBoundedGapSeconds = Number(lineage.maximum_bounded_gap_seconds);
    const boundedGaps = Array.isArray(lineage.bounded_uncertain_gaps)
      ? lineage.bounded_uncertain_gaps
      : [];
    const boundedGapIds = asList(lineage.bounded_gap_source_segment_ids);
    $("event-span").textContent = `${spanSeconds.toFixed(3)} s`;
    $("event-observed-support").textContent = Number.isFinite(observedSupportSeconds)
      ? [
        `${observedSupportSeconds.toFixed(3)} s`,
        Number.isFinite(observedSupportRatio)
          ? `${(observedSupportRatio * 100).toFixed(1)}% of event span`
          : "",
        Number.isFinite(supportFragmentCount)
          ? `${supportFragmentCount} support fragment(s)`
          : "",
      ].filter(Boolean).join(" · ")
      : "not recorded";
    $("event-bounded-gap").textContent = Number.isFinite(boundedGapSeconds)
      ? [
        `${boundedGapSeconds.toFixed(3)} s total`,
        Number.isFinite(maximumBoundedGapSeconds)
          ? `${maximumBoundedGapSeconds.toFixed(3)} s max`
          : "",
        `${boundedGaps.length} explicit gap(s)`,
      ].filter(Boolean).join(" · ")
      : "not recorded";
    $("bounded-gap-source-segment-ids").textContent = boundedGapIds.length
      ? boundedGapIds.join(", ")
      : "none recorded";
    const explicitMergeValues = [
      lineage.pre_gate_merge_count,
      lineage.merged_fragment_count,
    ].map(Number).filter((value) => Number.isFinite(value) && value >= 0);
    const explicitMergeCount = explicitMergeValues.length
      ? explicitMergeValues[0]
      : null;
    const explicitAggregation = lineage.pre_gate_aggregation_applied === true;
    const supportGroupId = lineage.support_group_id || "support group not recorded";
    $("event-merge-provenance").textContent = explicitMergeCount !== null
      ? `${explicitAggregation ? "pre-gate aggregation" : "explicit merge record"} · ${explicitMergeCount} merged fragment(s) · ${supportGroupId}`
      : explicitAggregation
      ? `pre-gate aggregation recorded; count not recorded · ${supportGroupId}`
      : "not recorded; source lineage is not merge proof";
    $("stabilization-reason").textContent = [
      lineage.stabilization_reason || "未记录",
      lineage.gap_kind && lineage.gap_kind !== "none"
        ? `gap ${lineage.gap_kind}`
        : "",
    ].filter(Boolean).join(" · ");
    const intervalStart = Number(lineage.start_time);
    const intervalEnd = Number(lineage.end_time);
    const lineageSide = String(
      lineage.anatomical_side || lineage.side || "",
    ).toLowerCase();
    const gapIdSet = new Set(boundedGapIds);
    const suppressedInContext = state.suppressedEvidence.filter((item) => (
      String(item.person_ref) === String(lineage.person_ref)
      && String(item.lock_epoch) === String(lineage.lock_epoch)
    ));
    const explicitGapSuppressed = suppressedInContext.filter((item) => {
      const itemIds = new Set([
        item.segment_id,
        ...asList(item.source_segment_ids),
      ].filter(Boolean));
      return [...itemIds].some((itemId) => gapIdSet.has(itemId));
    });
    const explicitGapSegmentIds = new Set(
      explicitGapSuppressed.flatMap((item) => [
        item.segment_id,
        ...asList(item.source_segment_ids),
      ]).filter(Boolean),
    );
    const overlappingSuppressed = suppressedInContext.filter((item) => {
      const itemIds = [
        item.segment_id,
        ...asList(item.source_segment_ids),
      ].filter(Boolean);
      if (itemIds.some((itemId) => explicitGapSegmentIds.has(itemId))) return false;
      const itemSide = String(
        item.anatomical_side || item.side || "",
      ).toLowerCase();
      if (!lineageSide || itemSide !== lineageSide) return false;
      return Number(item.end_time) >= intervalStart
        && Number(item.start_time) <= intervalEnd;
    });
    const reasons = [];
    if (explicitMergeCount !== null && explicitMergeCount > 0) {
      reasons.push(`explicit pre-gate merges ${explicitMergeCount}`);
    } else if (explicitAggregation) {
      reasons.push("explicit pre-gate aggregation");
    }
    if (lineage.merge_reason) reasons.push(lineage.merge_reason);
    if (lineage.suppression_reason) reasons.push(lineage.suppression_reason);
    if (explicitGapSuppressed.length) {
      const labels = [...new Set(explicitGapSuppressed.map(
        (item) => item.stabilization_reason || "explicit bounded gap"
      ))];
      reasons.push(`absorbed bounded gaps ${explicitGapSuppressed.length}: ${labels.join(", ")}`);
    }
    if (overlappingSuppressed.length) {
      const labels = [...new Set(overlappingSuppressed.map(
        (item) => item.stabilization_reason || "suppressed short evidence"
      ))];
      reasons.push(`same-lane suppressed ${overlappingSuppressed.length}: ${labels.join(", ")}`);
    }
    $("fragment-reason").textContent = reasons.length
      ? reasons.join(" · ")
      : "no explicit merge or same-lane suppression record";
  } else {
    $("event-visibility").textContent = "—";
    $("event-span").textContent = "—";
    $("event-observed-support").textContent = "—";
    $("event-bounded-gap").textContent = "—";
    $("bounded-gap-source-segment-ids").textContent = "—";
    $("event-merge-provenance").textContent = "—";
    $("stabilization-reason").textContent = "—";
    $("fragment-reason").textContent = "—";
  }

  const detail = $("source-segment-detail");
  detail.replaceChildren();
  if (!segments.length) {
    detail.textContent = lineage ? "未找到引用的 pose_segments" : "无当前证据";
    return;
  }
  segments.forEach((segment) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "source-segment-row";
    row.title = `跳转到 ${formatTime(segment.start_time)}`;
    row.textContent = [
      segment.segment_id,
      segment.action || segment.action_name || "unknown",
      `${eventDuration(segment).toFixed(2)} s`,
      segment.observation_state || "unknown",
      segment.anatomical_side || segment.side || "unknown",
    ].join(" · ");
    row.addEventListener("click", () => jumpToTime(segment.start_time));
    detail.appendChild(row);
  });
}

function updateEvidence(frame, time) {
  $("clock").textContent = formatTime(time);
  const event = currentEvent(time);
  const evidence = currentEvidence(time);
  const evidenceState = evidence ? normalizeEvidenceState(evidence) : "unavailable";
  const duration = eventDuration(event);
  const elapsed = event
    ? clamp(time - Number(event.start_time), 0, duration)
    : 0;
  $("current-action").textContent = event?.action || "no stable event";
  $("current-side").textContent = event?.anatomical_side || event?.side || "—";
  $("current-action-duration").textContent = event
    ? `${elapsed.toFixed(2)} / ${duration.toFixed(2)} s`
    : "—";
  $("action-duration").textContent = event ? `${duration.toFixed(2)} s` : "—";
  document.querySelectorAll(".action-event").forEach((node) => {
    node.classList.toggle("active", node.dataset.eventId === event?.action_event_id);
  });
  document.querySelectorAll(".evidence-interval").forEach((node) => {
    const evidenceId = (
      evidence?.evidence_interval_id
      || evidence?.timeline_event_id
    );
    node.classList.toggle("active", node.dataset.evidenceId === evidenceId);
  });
  document.querySelectorAll(".process-step").forEach((node) => {
    const active = time >= Number(node.dataset.start) && time < Number(node.dataset.end);
    node.classList.toggle("active", active);
  });
  const hands = nearestHands(frame, time);
  updateCurrentHandEvidenceSummary(hands);
  updateHandSideReadout("left", hands.left, frame);
  updateHandSideReadout("right", hands.right, frame);
  setChip($("current-evidence-state"), evidenceState);
  updateEventEvidence(event, evidence, hands);
  if (!frame) {
    $("frame-evidence").textContent = evidence
      ? `${evidenceState} · continuous interval`
      : "outside analyzed evidence window";
    $("person-ref").textContent = event?.person_ref || "unlocked";
    $("lock-epoch").textContent = event?.lock_epoch ?? 0;
    $("candidate-count").textContent = 0;
    $("evidence-time").textContent = "—";
    setChip($("lock-state"), "uncertain");
    $("lost-banner").classList.add("hidden");
    ["detected", "predicted", "interpolated", "missing"].forEach((name) => {
      const value = name === "missing" ? 1 : 0;
      $(`${name}-value`).textContent = value.toFixed(2);
      $(`${name}-bar`).style.width = `${value * 100}%`;
    });
    drawFrame(null, hands);
    return;
  }
  $("person-ref").textContent = frame.person_ref;
  $("lock-epoch").textContent = frame.lock_epoch;
  $("candidate-count").textContent = frame.candidate_person_count;
  $("evidence-time").textContent = `${Number(frame.timestamp).toFixed(2)} s`;
  $("frame-evidence").textContent = [
    `${evidenceState} evidence`,
    `${frame.observation_state} pose`,
    `source frame ${frame.source_frame_index}`,
  ].join(" · ");
  setChip($("lock-state"), frame.lock_state);
  $("lost-banner").classList.toggle("hidden", frame.track_state === "tracked");
  ["detected", "predicted", "interpolated", "missing"].forEach((name) => {
    const value = clamp(Number(frame[`${name}_ratio`]) || 0, 0, 1);
    $(`${name}-value`).textContent = value.toFixed(2);
    $(`${name}-bar`).style.width = `${value * 100}%`;
  });
  drawFrame(frame, hands);
}

function unionDuration(items, predicate = () => true) {
  const intervals = items
    .filter(predicate)
    .map((item) => [
      clamp(Number(item.start_time), state.timelineStart, state.timelineEnd),
      clamp(Number(item.end_time), state.timelineStart, state.timelineEnd),
    ])
    .filter(([start, end]) => Number.isFinite(start) && Number.isFinite(end) && end > start)
    .sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  let total = 0;
  let activeStart = null;
  let activeEnd = null;
  intervals.forEach(([start, end]) => {
    if (activeStart === null) {
      activeStart = start;
      activeEnd = end;
    } else if (start <= activeEnd + 1e-9) {
      activeEnd = Math.max(activeEnd, end);
    } else {
      total += activeEnd - activeStart;
      activeStart = start;
      activeEnd = end;
    }
  });
  if (activeStart !== null) total += activeEnd - activeStart;
  return total;
}

function rawActionSwitchMetrics() {
  const ordered = [...state.frames].sort(
    (left, right) => Number(left.timestamp) - Number(right.timestamp),
  );
  let previous = null;
  let switches = 0;
  let denominator = 0;
  ordered.forEach((frame) => {
    const tracked = frame.track_state === "tracked";
    const action = String(frame.action || "unknown");
    const key = `${frame.person_ref}|${frame.lock_epoch}`;
    if (!tracked) {
      previous = null;
      return;
    }
    if (previous && previous.key === key) {
      denominator += 1;
      if (previous.action !== action) switches += 1;
    }
    previous = {key, action};
  });
  return {
    switches,
    denominator,
    rate: denominator ? switches / denominator : 0,
  };
}

function renderTimelineStatistics() {
  const stabilization = state.analysis.stabilization_metrics || {};
  const runtime = state.analysis.runtime || {};
  const evidenceSeconds = unionDuration(state.evidenceTimeline);
  const stableNormalEvents = state.allActionEvents.filter(
    (event) => (
      event.event_kind === "stable_action"
      && NORMAL_ACTIONS.has(String(event.action || "").toLowerCase())
    ),
  );
  const normalSpanSeconds = unionDuration(stableNormalEvents);
  const normalSupportIds = new Set(
    stableNormalEvents.flatMap((event) => asList(event.source_segment_ids)),
  );
  const normalSupportSegments = state.poseSegments.filter(
    (segment) => normalSupportIds.has(segment.segment_id),
  );
  const normalObservedSupportSeconds = unionDuration(normalSupportSegments);
  const hasDirectSupportIntervals = (
    normalSupportIds.size > 0 && normalSupportSegments.length > 0
  );
  const explicitNormalObservedSupportSeconds = Number(
    stabilization.stable_normal_observed_support_seconds
    ?? runtime.stable_normal_observed_support_seconds,
  );
  const unknownUncertainSeconds = unionDuration(
    state.evidenceTimeline,
    (interval) => ["unknown", "uncertain"].includes(
      normalizeEvidenceState(interval),
    ),
  );
  const fallbackSwitches = rawActionSwitchMetrics();
  const explicitSwitchRate = Number(
    stabilization.raw_action_switch_rate
    ?? runtime.raw_action_switch_rate,
  );
  const explicitSwitchCount = Number(
    stabilization.raw_action_switch_count
    ?? runtime.raw_action_switch_count,
  );
  const explicitSwitchDenominator = Number(
    stabilization.raw_action_switch_denominator
    ?? runtime.raw_action_switch_denominator,
  );
  const rawSwitch = {
    rate: Number.isFinite(explicitSwitchRate)
      ? explicitSwitchRate
      : fallbackSwitches.rate,
    switches: Number.isFinite(explicitSwitchCount)
      ? explicitSwitchCount
      : fallbackSwitches.switches,
    denominator: Number.isFinite(explicitSwitchDenominator)
      ? explicitSwitchDenominator
      : fallbackSwitches.denominator,
  };
  const asPercent = (seconds) => (
    `${(clamp(seconds / state.timelineDuration, 0, 1) * 100).toFixed(1)}%`
  );
  $("timeline-coverage").textContent = asPercent(evidenceSeconds);
  $("timeline-coverage").title = `${evidenceSeconds.toFixed(3)} / ${state.timelineDuration.toFixed(3)} s`;
  $("normal-action-coverage").textContent = asPercent(normalSpanSeconds);
  $("normal-action-coverage").title = `event span ${normalSpanSeconds.toFixed(3)} / ${state.timelineDuration.toFixed(3)} s`;
  if (hasDirectSupportIntervals) {
    $("normal-observed-support-coverage").textContent = asPercent(
      normalObservedSupportSeconds,
    );
    $("normal-observed-support-coverage").title = [
      `direct-support interval union ${normalObservedSupportSeconds.toFixed(3)} / ${state.timelineDuration.toFixed(3)} s`,
      Number.isFinite(explicitNormalObservedSupportSeconds)
        ? `reported cumulative support ${explicitNormalObservedSupportSeconds.toFixed(3)} s`
        : "",
    ].filter(Boolean).join(" · ");
  } else if (
    Number.isFinite(explicitNormalObservedSupportSeconds)
    && explicitNormalObservedSupportSeconds >= 0
  ) {
    $("normal-observed-support-coverage").textContent = `reported ${explicitNormalObservedSupportSeconds.toFixed(3)} s`;
    $("normal-observed-support-coverage").title = "No direct support intervals were available for a non-overlapping coverage calculation";
  } else {
    $("normal-observed-support-coverage").textContent = "not recorded";
    $("normal-observed-support-coverage").title = "No explicit observed-support metric in this payload";
  }
  $("unknown-uncertain-coverage").textContent = asPercent(unknownUncertainSeconds);
  $("unknown-uncertain-coverage").title = `${unknownUncertainSeconds.toFixed(3)} / ${state.timelineDuration.toFixed(3)} s`;
  $("raw-switch-rate").textContent = `${(clamp(rawSwitch.rate, 0, 1) * 100).toFixed(1)}%`;
  $("raw-switch-rate").title = `${rawSwitch.switches} / ${rawSwitch.denominator} comparable tracked frame pairs`;
}

function apiPost(path, payload = {}) {
  return fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  }).then(async (response) => {
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.message || result.reason || `Request failed (${response.status})`);
    }
    return result;
  });
}

function setIntakeState(value, message = null) {
  setChip($("intake-state"), value);
  if (message) $("upload-progress-text").textContent = message;
}

function formatBytes(value) {
  const bytes = Math.max(0, Number(value) || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

function renderUploadMetadata(upload) {
  const probe = upload?.probe || {};
  $("upload-metadata").textContent = [
    `文件：${upload.original_filename}`,
    `大小：${formatBytes(upload.size_bytes)}`,
    `SHA256：${String(upload.sha256 || "").slice(0, 16)}…`,
    `时长：${Number(probe.duration_seconds || 0).toFixed(3)} s`,
    `FPS：${Number(probe.fps || 0).toFixed(3)}`,
    `分辨率：${probe.width || 0} × ${probe.height || 0}`,
    `编码：${probe.codec || "unknown"}`,
    `可解码：${probe.decodable === true}`,
  ].join("\n");
  const maximumStart = Math.max(0, Number(probe.duration_seconds || 0) - 0.001);
  $("analysis-start-time").max = String(maximumStart);
}

function resetVideoPreview(message = "请生成真实人物候选预览。") {
  state.preview = null;
  state.selectedPreviewCandidate = null;
  $("video-preview-image").removeAttribute("src");
  $("video-preview-message").textContent = message;
  $("video-analysis-start-button").disabled = true;
  const canvas = $("video-preview-canvas");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function previewGeometry() {
  const image = $("video-preview-image");
  const canvas = $("video-preview-canvas");
  const canvasRect = canvas.getBoundingClientRect();
  const imageRect = image.getBoundingClientRect();
  const sourceWidth = Number(state.preview?.width || 0);
  const sourceHeight = Number(state.preview?.height || 0);
  if (!sourceWidth || !sourceHeight || !imageRect.width || !imageRect.height) return null;
  const sourceRatio = sourceWidth / sourceHeight;
  const boxRatio = imageRect.width / imageRect.height;
  let shownWidth = imageRect.width;
  let shownHeight = imageRect.height;
  let offsetX = imageRect.left - canvasRect.left;
  let offsetY = imageRect.top - canvasRect.top;
  if (sourceRatio > boxRatio) {
    shownHeight = imageRect.width / sourceRatio;
    offsetY += (imageRect.height - shownHeight) / 2;
  } else {
    shownWidth = imageRect.height * sourceRatio;
    offsetX += (imageRect.width - shownWidth) / 2;
  }
  return {
    offsetX,
    offsetY,
    shownWidth,
    shownHeight,
    sourceWidth,
    sourceHeight,
  };
}

function renderVideoPreviewCandidates() {
  const canvas = $("video-preview-canvas");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, rect.width, rect.height);
  const geometry = previewGeometry();
  if (!geometry) return;
  const sx = geometry.shownWidth / geometry.sourceWidth;
  const sy = geometry.shownHeight / geometry.sourceHeight;
  (state.preview?.candidates || []).forEach((candidate) => {
    const bbox = candidate.bbox || [];
    if (bbox.length !== 4) return;
    const selected = (
      candidate.candidate_token === state.selectedPreviewCandidate?.candidate_token
    );
    const x = geometry.offsetX + Number(bbox[0]) * sx;
    const y = geometry.offsetY + Number(bbox[1]) * sy;
    const width = (Number(bbox[2]) - Number(bbox[0])) * sx;
    const height = (Number(bbox[3]) - Number(bbox[1])) * sy;
    context.strokeStyle = selected ? "#35d49a" : "#e7b45a";
    context.fillStyle = selected ? "rgba(53,212,154,.14)" : "rgba(231,180,90,.08)";
    context.lineWidth = selected ? 3 : 2;
    context.fillRect(x, y, width, height);
    context.strokeRect(x, y, width, height);
    context.fillStyle = selected ? "#35d49a" : "#e7b45a";
    context.font = "600 12px Segoe UI";
    context.fillText(
      `${candidate.candidate_id} ${(Number(candidate.person_confidence) * 100).toFixed(0)}%`,
      x + 5,
      Math.max(14, y + 15),
    );
  });
}

function selectPreviewCandidateFromPoint(event) {
  if (!state.preview) return;
  const canvasRect = $("video-preview-canvas").getBoundingClientRect();
  const geometry = previewGeometry();
  if (!geometry) return;
  const displayX = event.clientX - canvasRect.left;
  const displayY = event.clientY - canvasRect.top;
  const sourceX = (displayX - geometry.offsetX) / geometry.shownWidth * geometry.sourceWidth;
  const sourceY = (displayY - geometry.offsetY) / geometry.shownHeight * geometry.sourceHeight;
  const hits = (state.preview.candidates || []).filter((candidate) => {
    const [x1, y1, x2, y2] = candidate.bbox || [];
    return sourceX >= x1 && sourceX <= x2 && sourceY >= y1 && sourceY <= y2;
  }).sort((left, right) => {
    const leftArea = (left.bbox[2] - left.bbox[0]) * (left.bbox[3] - left.bbox[1]);
    const rightArea = (right.bbox[2] - right.bbox[0]) * (right.bbox[3] - right.bbox[1]);
    return leftArea - rightArea;
  });
  state.selectedPreviewCandidate = hits[0] || null;
  $("video-analysis-start-button").disabled = !state.selectedPreviewCandidate;
  $("video-preview-message").textContent = state.selectedPreviewCandidate
    ? `已选择匿名候选 ${state.selectedPreviewCandidate.candidate_id}；点击“开始分析”后由 worker 重新验证。`
    : "未命中候选框；重叠框按较小面积优先。";
  renderVideoPreviewCandidates();
}

function uploadSelectedVideo() {
  const file = $("video-file-input").files?.[0];
  if (!file) return;
  resetVideoPreview("正在上传，完成后可预览人物候选。");
  $("video-upload-button").disabled = true;
  setIntakeState("uploading", "上传中 0.0%");
  const request = new XMLHttpRequest();
  request.open(
    "POST",
    `/api/video/upload?filename=${encodeURIComponent(file.name)}`,
  );
  request.setRequestHeader("Content-Type", "video/mp4");
  request.upload.addEventListener("progress", (event) => {
    if (!event.lengthComputable) return;
    const progress = clamp(event.loaded / event.total, 0, 1);
    $("upload-progress-bar").style.width = `${progress * 100}%`;
    $("upload-progress-text").textContent = `上传中 ${(progress * 100).toFixed(1)}%`;
  });
  request.addEventListener("load", () => {
    $("video-upload-button").disabled = false;
    let payload = {};
    try {
      payload = JSON.parse(request.responseText || "{}");
    } catch (_error) {
      payload = {};
    }
    if (request.status < 200 || request.status >= 300) {
      setIntakeState("error", payload.message || "上传失败。");
      return;
    }
    state.upload = payload;
    $("upload-progress-bar").style.width = `${Number(100)}%`;
    setIntakeState("ready", "上传、SHA256和真实视频探测完成。");
    renderUploadMetadata(payload);
    $("video-preview-button").disabled = false;
  });
  request.addEventListener("error", () => {
    $("video-upload-button").disabled = false;
    setIntakeState("error", "本地上传连接中断；未完成文件不会被当作完整视频。");
  });
  request.send(file);
}

async function previewUploadedVideo() {
  if (!state.upload) return;
  $("video-preview-button").disabled = true;
  $("video-preview-message").textContent = "正在使用真实 Body Pose 生成候选…";
  try {
    const preview = await apiPost("/api/video/preview", {
      upload_id: state.upload.upload_id,
      start_time: Number($("analysis-start-time").value || 0),
    });
    state.preview = preview;
    state.selectedPreviewCandidate = null;
    const image = $("video-preview-image");
    image.onload = () => renderVideoPreviewCandidates();
    image.src = preview.preview_image;
    $("video-preview-message").textContent = preview.candidate_count
      ? `检测到 ${preview.candidate_count} 个匿名候选；请点击目标人物框。`
      : "当前帧没有真实人物候选，请调整起始时间后重试。";
  } catch (error) {
    $("video-preview-message").textContent = error.message;
  } finally {
    $("video-preview-button").disabled = false;
  }
}

async function startUploadedVideoJob() {
  if (!state.upload || !state.preview || !state.selectedPreviewCandidate) return;
  $("video-analysis-start-button").disabled = true;
  try {
    const result = await apiPost("/api/video/jobs/start", {
      upload_id: state.upload.upload_id,
      preview_id: state.preview.preview_id,
      candidate_token: state.selectedPreviewCandidate.candidate_token,
      start_time: Number($("analysis-start-time").value || 0),
      duration_seconds: Number($("analysis-duration").value || 12),
      full_video: $("full-video-toggle").checked,
      hand_enabled: $("upload-hand-toggle").checked,
      body_provider_policy: $("upload-body-provider").value,
      recording_group_id: $("recording-group-id").value,
    });
    setIntakeState(result.state, "分析作业已进入队列。");
    $("video-analysis-cancel-button").disabled = false;
    renderVideoJob(result);
    scheduleVideoJobPoll(0);
  } catch (error) {
    setIntakeState("error", error.message);
    $("video-analysis-start-button").disabled = false;
  }
}

function renderVideoJob(job) {
  $("video-job-stage").textContent = job.stage || job.state || "pending";
  $("video-job-progress").textContent = `${(Number(job.progress || 0) * 100).toFixed(1)}%`;
  $("video-job-elapsed").textContent = `${Number(job.elapsed_seconds || 0).toFixed(1)} s`;
  $("video-job-message").textContent = job.public_error || job.message || "—";
  setIntakeState(job.state || "pending");
  const terminal = ["completed", "cancelled", "failed"].includes(job.state);
  $("video-analysis-cancel-button").disabled = terminal;
  if (terminal && state.selectedPreviewCandidate) {
    $("video-analysis-start-button").disabled = false;
  }
}

function scheduleVideoJobPoll(delayMs = 300) {
  if (state.jobPollTimer !== null) window.clearTimeout(state.jobPollTimer);
  state.jobPollTimer = window.setTimeout(pollVideoJob, delayMs);
}

async function pollVideoJob() {
  try {
    const response = await fetch("/api/video/job/status", {cache: "no-store"});
    const job = await response.json();
    renderVideoJob(job);
    if (job.state === "completed" && job.job_id !== state.loadedJobId) {
      state.loadedJobId = job.job_id;
      await loadCurrentAnalysis({reloadVideo: true});
      $("video-job-message").textContent = "分析完成；新的真实视频和证据已直接加载。";
      showVideoMode();
    }
    if (job.worker_alive && ["completed", "cancelled", "failed"].includes(job.state)) {
      scheduleVideoJobPoll(400);
      return;
    }
    if (!["completed", "cancelled", "failed", "pending"].includes(job.state)) {
      scheduleVideoJobPoll();
    }
  } catch (error) {
    $("video-job-message").textContent = error.message;
  }
}

async function cancelUploadedVideoJob() {
  $("video-analysis-cancel-button").disabled = true;
  try {
    const job = await apiPost("/api/video/jobs/cancel");
    renderVideoJob(job);
    scheduleVideoJobPoll();
  } catch (error) {
    $("video-job-message").textContent = error.message;
  }
}

function applyAnalysis(analysis, {reloadVideo = false} = {}) {
  state.analysis = analysis;
  state.frames = state.analysis.pose_frames || [];
  state.handFrames = state.analysis.hand_pose_frames || [];
  state.poseSegments = state.analysis.pose_segments || [];
  state.suppressedEvidence = state.analysis.suppressed_action_evidence || [];
  const explicitEvidence = state.analysis.evidence_timeline || [];
  state.evidenceTimeline = explicitEvidence.length
    ? explicitEvidence
    : deriveEvidenceTimeline(state.poseSegments);
  state.evidenceTimelineSource = explicitEvidence.length
    ? "evidence_timeline"
    : state.poseSegments.length
    ? "pose_segments fallback"
    : "unavailable";
  state.allActionEvents = state.analysis.action_events || [];
  state.actionEvents = state.allActionEvents.filter(
    (event) => (
      event.display_eligible !== false
      || event.event_kind === "hard_boundary"
      || event.action === "lost"
    )
  );
  state.processSteps = state.analysis.process_steps || [];
  const analysisWindow = state.analysis.source_video?.analysis_window || {};
  const requestedStart = Number(analysisWindow.start_time);
  const requestedEnd = Number(analysisWindow.end_time);
  state.timelineStart = Number.isFinite(requestedStart) ? requestedStart : 0;
  state.timelineEnd = (
    Number.isFinite(requestedEnd) && requestedEnd > state.timelineStart
      ? requestedEnd
      : Number(state.analysis.source_video?.duration_seconds) || state.timelineStart + 1
  );
  state.timelineDuration = Math.max(1e-9, state.timelineEnd - state.timelineStart);
  $("run-state").textContent = "analysis available";
  $("run-state").classList.add("available");
  $("video-title").textContent = state.analysis.source_video.path;
  const stabilization = state.analysis.stabilization_metrics || {};
  const runtime = state.analysis.runtime || {};
  const handQualityCounts = runtime.hand_quality_state_counts || {};
  const fallbackHandQualityCount = (qualityState) => state.handFrames.filter(
    (record) => String(record.quality_state || "").toLowerCase() === qualityState,
  ).length;
  const handQualityCount = (qualityState) => {
    const runtimeValue = handQualityCounts[qualityState];
    return Number.isFinite(Number(runtimeValue))
      ? Number(runtimeValue)
      : fallbackHandQualityCount(qualityState);
  };
  const runtimeEligibleCount = Number(
    runtime.hand_action_feature_eligible_observation_count,
  );
  const subsecondCount = stabilization.sub_1s_stable_event_count
    ?? state.actionEvents.filter(
      (event) => event.action !== "lost" && eventDuration(event) < 1
    ).length;
  const counts = {
    "pose-count": state.frames.length,
    "hand-detected-count": Number(runtime.hand_detected_frame_count) || 0,
    "hand-uncertain-count": Number(runtime.hand_uncertain_frame_count) || 0,
    "hand-missing-count": Number(runtime.hand_missing_frame_count) || 0,
    "hand-qualified-count": handQualityCount("qualified"),
    "hand-association-uncertain-count": handQualityCount(
      "association_uncertain",
    ),
    "hand-insufficient-geometry-count": handQualityCount(
      "insufficient_geometry",
    ),
    "hand-not-observed-count": handQualityCount("not_observed"),
    "hand-eligible-observation-count": Number.isFinite(runtimeEligibleCount)
      ? runtimeEligibleCount
      : state.handFrames.filter(
        (record) => record.action_feature_eligible === true,
      ).length,
    "action-count": Number(
      runtime.stable_normal_action_count
      ?? stabilization.stable_normal_action_count
      ?? state.actionEvents.filter((event) => event.status === "proposed").length
    ) || 0,
    "subsecond-action-count": Number(subsecondCount) || 0,
    "suppressed-count": Number(
      stabilization.suppressed_fragment_count
      ?? stabilization.suppressed_count
      ?? state.suppressedEvidence.length
    ) || 0,
    "merged-count": Number(
      stabilization.merged_fragment_count
      ?? stabilization.merge_count
      ?? 0
    ) || 0,
    "object-count": state.analysis.object_tracks?.length || 0,
    "interaction-count": state.analysis.interaction_events?.length || 0,
    "step-count": state.processSteps.length,
  };
  Object.entries(counts).forEach(([id, value]) => $(id).textContent = value);
  renderProcessTimeline();
  renderEvidenceTimeline();
  renderActionTimeline();
  renderTimelineStatistics();
  renderLayerRows();
  if (reloadVideo) {
    const video = $("video");
    video.src = `/media/video?revision=${Date.now()}`;
    video.load();
  }
}

async function loadCurrentAnalysis(options = {}) {
  const response = await fetch("/api/analysis", {cache: "no-store"});
  if (!response.ok) throw new Error("The completed analysis could not be loaded.");
  applyAnalysis(await response.json(), options);
}

async function boot() {
  await loadCurrentAnalysis();
  const video = $("video");
  const update = () => {
    if (state.mode === "camera") {
      if (state.cameraEvidence) renderCameraEvidence(state.cameraEvidence);
      return;
    }
    updateEvidence(nearestFrame(video.currentTime), video.currentTime);
  };
  $("body-pose-toggle").addEventListener("change", (event) => {
    state.showBodyPose = event.target.checked;
    update();
  });
  $("hand-pose-toggle").addEventListener("change", (event) => {
    state.showHandPose = event.target.checked;
    update();
  });
  $("body-renderer-mode").addEventListener("change", (event) => {
    state.bodyRendererMode = event.target.value === "evidence"
      ? "evidence"
      : "classic";
    update();
  });
  $("video-file-input").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    state.upload = null;
    resetVideoPreview(
      file
        ? `${file.name} · ${formatBytes(file.size)} · 等待上传。`
        : "尚未选择文件。",
    );
    $("video-upload-button").disabled = !file;
    $("video-preview-button").disabled = true;
    $("upload-progress-bar").style.width = "0";
    setIntakeState(
      file ? "pending" : "unavailable",
      file ? `已选择 ${file.name}` : "尚未选择文件",
    );
  });
  $("video-upload-button").addEventListener("click", uploadSelectedVideo);
  $("video-preview-button").addEventListener("click", previewUploadedVideo);
  $("video-preview-canvas").addEventListener(
    "click",
    selectPreviewCandidateFromPoint,
  );
  $("video-analysis-start-button").addEventListener(
    "click",
    startUploadedVideoJob,
  );
  $("video-analysis-cancel-button").addEventListener(
    "click",
    cancelUploadedVideoJob,
  );
  $("full-video-toggle").addEventListener("change", (event) => {
    $("analysis-duration").disabled = event.target.checked;
    $("video-job-message").textContent = event.target.checked
      ? "完整视频将显式处理；预计耗时取决于视频长度和Hand CPU推理。"
      : "默认使用有界分析窗口。";
  });
  ["analysis-start-time", "analysis-duration"].forEach((id) => {
    $(id).addEventListener("change", () => {
      if (state.preview) resetVideoPreview("范围已变化，请重新预览并选择人物。");
      $("video-preview-button").disabled = !state.upload;
    });
  });
  $("camera-mode-button").addEventListener("click", startCamera);
  $("camera-stop-button").addEventListener("click", () => stopCamera());
  $("camera-select-person-button").addEventListener(
    "click",
    beginCameraPersonSelection,
  );
  $("camera-confirm-relock-button").addEventListener(
    "click",
    confirmCameraRelock,
  );
  $("camera-cancel-relock-button").addEventListener(
    "click",
    cancelCameraRelock,
  );
  $("pose-canvas").addEventListener(
    "click",
    selectCameraCandidateFromPoint,
  );
  $("video-mode-button").addEventListener("click", () => (
    stopCamera({returnToVideo: true})
  ));
  video.addEventListener("timeupdate", update);
  video.addEventListener("seeked", update);
  video.addEventListener("loadedmetadata", () => {
    if (
      video.currentTime < state.timelineStart
      || video.currentTime >= state.timelineEnd
    ) {
      video.currentTime = clamp(
        state.timelineStart,
        0,
        Number.isFinite(video.duration) ? video.duration : state.timelineStart,
      );
    }
    update();
  });
  window.addEventListener("resize", () => {
    update();
    renderVideoPreviewCandidates();
  });
  startVideoOverlayClock(video, update);
  scheduleCameraPoll(0);
  scheduleVideoJobPoll(0);
  update();
}

boot().catch((error) => {
  console.error(error);
  $("run-state").textContent = "analysis unavailable";
  $("process-reason").textContent = "Analysis data is unavailable; inspect the local validation log.";
});
