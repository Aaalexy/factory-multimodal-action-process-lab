/*
 * Original Web Canvas adaptation of the verified classic stickman geometry.
 *
 * Evidence rules:
 * - raw COCO-17 points retain detected/predicted/interpolated states;
 * - missing, uncertain and rejected points never create geometry;
 * - head, neck, hip centre, palm-tip and foot-tip geometry is explicitly
 *   derived_visual_only and is never written back to analysis evidence;
 * - no point or segment represents a real palm, finger, grasp or object fact.
 */
(function attachClassicBodyPoseRenderer(global) {
  "use strict";

  const KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
  ];

  const SEGMENTS = [
    ["head_center", "neck"],
    ["left_shoulder", "right_shoulder"],
    ["neck", "hip_center"],
    ["left_hip", "right_hip"],
    ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
    ["left_wrist", "left_palm"],
    ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
    ["right_wrist", "right_palm"],
    ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
    ["left_ankle", "left_foot"],
    ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
    ["right_ankle", "right_foot"],
  ];

  const FACE_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"];
  const REJECTED_STATES = new Set(["missing", "uncertain", "rejected", "lost"]);
  const STYLES = Object.freeze({
    detected: {
      color: "#22d3ee",
      alpha: 0.96,
      width: 3.4,
      dash: [],
      pointRadius: 3.8,
    },
    interpolated: {
      color: "#f3b44f",
      alpha: 0.82,
      width: 2.7,
      dash: [7, 4],
      pointRadius: 3.4,
    },
    predicted: {
      color: "#8aa8ff",
      alpha: 0.56,
      width: 2.3,
      dash: [3, 5],
      pointRadius: 3.1,
    },
    derived_visual_only: {
      color: "#e05bd8",
      alpha: 0.82,
      width: 2.2,
      dash: [5, 4],
      pointRadius: 2.8,
    },
  });

  function finitePoint(raw) {
    return (
      Array.isArray(raw)
      && Number.isFinite(Number(raw[0]))
      && Number.isFinite(Number(raw[1]))
    );
  }

  function usable(point) {
    return Boolean(
      point
      && finitePoint([point.x, point.y])
      && !REJECTED_STATES.has(point.status),
    );
  }

  function rawPoint(raw, status, name) {
    const normalized = String(status || "missing").toLowerCase();
    if (!finitePoint(raw) || REJECTED_STATES.has(normalized)) {
      return null;
    }
    return {
      name,
      x: Number(raw[0]),
      y: Number(raw[1]),
      confidence: Number.isFinite(Number(raw[2])) ? Number(raw[2]) : 0,
      status: ["detected", "predicted", "interpolated"].includes(normalized)
        ? normalized
        : "predicted",
      evidence_type: normalized,
      sources: [name],
    };
  }

  function derivedPoint(name, x, y, confidence, sources) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    return {
      name,
      x,
      y,
      confidence: Math.max(0, Math.min(1, Number(confidence) || 0)),
      status: "derived_visual_only",
      evidence_type: "derived_visual_only",
      sources: [...sources],
    };
  }

  function distance(left, right) {
    return Math.hypot(left.x - right.x, left.y - right.y);
  }

  function midpoint(name, left, right) {
    if (!usable(left) || !usable(right)) return null;
    return derivedPoint(
      name,
      (left.x + right.x) / 2,
      (left.y + right.y) / 2,
      Math.min(left.confidence, right.confidence),
      [left.name, right.name],
    );
  }

  function buildGeometry(frame) {
    const keypoints = Array.isArray(frame?.keypoints) ? frame.keypoints : [];
    const statuses = Array.isArray(frame?.keypoint_statuses)
      ? frame.keypoint_statuses
      : [];
    const points = {};
    KEYPOINT_NAMES.forEach((name, index) => {
      const point = rawPoint(keypoints[index], statuses[index], name);
      if (point) points[name] = point;
    });

    const leftShoulder = points.left_shoulder;
    const rightShoulder = points.right_shoulder;
    const leftHip = points.left_hip;
    const rightHip = points.right_hip;
    const shoulderWidth = (
      usable(leftShoulder) && usable(rightShoulder)
        ? distance(leftShoulder, rightShoulder)
        : 0
    );
    const neck = midpoint("neck", leftShoulder, rightShoulder);
    const hipCenter = midpoint("hip_center", leftHip, rightHip);
    if (neck) points.neck = neck;
    if (hipCenter) points.hip_center = hipCenter;

    const facePoints = FACE_NAMES.map((name) => points[name]).filter(usable);
    let headRadius = shoulderWidth > 0 ? Math.max(4, shoulderWidth * 0.18) : 8;
    if (facePoints.length) {
      const x = facePoints.reduce((sum, point) => sum + point.x, 0) / facePoints.length;
      const y = facePoints.reduce((sum, point) => sum + point.y, 0) / facePoints.length;
      const spread = Math.max(
        0,
        ...facePoints.map((point) => Math.hypot(point.x - x, point.y - y)),
      );
      headRadius = Math.max(headRadius, spread * 1.35);
      points.head_center = derivedPoint(
        "head_center",
        x,
        y,
        facePoints.reduce((sum, point) => sum + point.confidence, 0)
          / facePoints.length,
        facePoints.map((point) => point.name),
      );
    } else if (neck && shoulderWidth > 0) {
      points.head_center = derivedPoint(
        "head_center",
        neck.x,
        neck.y - Math.max(8, shoulderWidth * 0.38),
        neck.confidence * 0.6,
        ["neck", "shoulder_width"],
      );
    }

    const scale = shoulderWidth > 0 ? shoulderWidth : 50;
    ["left", "right"].forEach((side) => {
      const elbow = points[`${side}_elbow`];
      const wrist = points[`${side}_wrist`];
      if (usable(elbow) && usable(wrist)) {
        const vx = wrist.x - elbow.x;
        const vy = wrist.y - elbow.y;
        const length = Math.hypot(vx, vy);
        if (length > 1e-6) {
          const extension = Math.min(scale * 0.16, length * 0.28);
          points[`${side}_palm`] = derivedPoint(
            `${side}_palm`,
            wrist.x + (vx / length) * extension,
            wrist.y + (vy / length) * extension,
            Math.min(elbow.confidence, wrist.confidence) * 0.8,
            [`${side}_elbow`, `${side}_wrist`],
          );
        }
      }

      const knee = points[`${side}_knee`];
      const ankle = points[`${side}_ankle`];
      if (usable(knee) && usable(ankle)) {
        const vx = ankle.x - knee.x;
        const vy = ankle.y - knee.y;
        const length = Math.hypot(vx, vy);
        if (length > 1e-6) {
          const lateral = side === "left" ? -0.45 : 0.45;
          let dx = vx / length + lateral;
          let dy = vy / length + 0.05;
          const directionLength = Math.max(Math.hypot(dx, dy), 1e-6);
          dx /= directionLength;
          dy /= directionLength;
          const extension = Math.min(scale * 0.24, length * 0.30);
          points[`${side}_foot`] = derivedPoint(
            `${side}_foot`,
            ankle.x + dx * extension,
            ankle.y + dy * extension,
            Math.min(knee.confidence, ankle.confidence) * 0.8,
            [`${side}_knee`, `${side}_ankle`],
          );
        }
      }
    });

    const segments = SEGMENTS.filter(([start, end]) => (
      usable(points[start]) && usable(points[end])
    )).map(([start, end]) => ({start, end}));
    return {points, segments, headRadius};
  }

  function segmentStatus(left, right) {
    const states = new Set([left.status, right.status]);
    if (states.has("derived_visual_only")) return "derived_visual_only";
    if (states.has("predicted")) return "predicted";
    if (states.has("interpolated")) return "interpolated";
    return "detected";
  }

  function applyStyle(context, status, scale = 1) {
    const style = STYLES[status] || STYLES.predicted;
    context.strokeStyle = style.color;
    context.fillStyle = style.color;
    context.globalAlpha = style.alpha;
    context.lineWidth = Math.max(1, style.width * scale);
    context.setLineDash(style.dash.map((value) => value * scale));
    context.lineCap = "round";
    context.lineJoin = "round";
    return style;
  }

  function render(context, frame, point, options = {}) {
    if (!context || typeof point !== "function") {
      throw new TypeError("Classic renderer requires a Canvas context and point transform");
    }
    const metrics = {
      raw_point_count: 0,
      derived_visual_only_point_count: 0,
      segment_count: 0,
      drew_fixed_geometry: false,
    };
    const scale = Number.isFinite(options.lineScale)
      ? Math.max(0.6, Math.min(1.8, options.lineScale))
      : 1;

    (frame?.anonymous_candidates || []).forEach((candidate) => {
      if (!Array.isArray(candidate?.bbox) || candidate.bbox.length < 4) return;
      const [x1, y1] = point([candidate.bbox[0], candidate.bbox[1]]);
      const [x2, y2] = point([candidate.bbox[2], candidate.bbox[3]]);
      context.globalAlpha = 0.55;
      context.strokeStyle = "#f3b44f";
      context.lineWidth = 1;
      context.setLineDash([5, 5]);
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
    context.setLineDash([]);
    context.globalAlpha = 1;

    if (frame?.track_state !== "tracked") return metrics;
    const geometry = buildGeometry(frame);
    const values = Object.values(geometry.points);
    metrics.raw_point_count = values.filter(
      (item) => item.status !== "derived_visual_only",
    ).length;
    metrics.derived_visual_only_point_count = values.filter(
      (item) => item.status === "derived_visual_only",
    ).length;

    if (Array.isArray(frame?.bbox) && frame.bbox.length >= 4) {
      const [x1, y1] = point([frame.bbox[0], frame.bbox[1]]);
      const [x2, y2] = point([frame.bbox[2], frame.bbox[3]]);
      context.globalAlpha = 0.82;
      context.strokeStyle = STYLES.detected.color;
      context.lineWidth = 1.5 * scale;
      context.setLineDash([]);
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = `${String(frame.person_ref || "anonymous")} · epoch ${String(frame.lock_epoch ?? "—")}`;
      context.font = `${Math.round(10 * scale)}px ui-monospace, SFMono-Regular, Consolas, monospace`;
      context.fillStyle = "rgba(4, 15, 22, .86)";
      const labelWidth = Math.max(110, context.measureText(label).width + 10);
      context.fillRect(x1, Math.max(0, y1 - 18 * scale), labelWidth, 17 * scale);
      context.fillStyle = STYLES.detected.color;
      context.globalAlpha = 0.96;
      context.fillText(label, x1 + 5, Math.max(11, y1 - 6 * scale));
    }

    geometry.segments.forEach(({start, end}) => {
      const left = geometry.points[start];
      const right = geometry.points[end];
      const status = segmentStatus(left, right);
      applyStyle(context, status, scale);
      const [x1, y1] = point([left.x, left.y]);
      const [x2, y2] = point([right.x, right.y]);
      context.beginPath();
      context.moveTo(x1, y1);
      context.lineTo(x2, y2);
      context.stroke();
      metrics.segment_count += 1;
    });

    const head = geometry.points.head_center;
    if (usable(head) && geometry.headRadius > 0) {
      applyStyle(context, "derived_visual_only", scale);
      const [x, y] = point([head.x, head.y]);
      const [edgeX] = point([head.x + geometry.headRadius, head.y]);
      context.beginPath();
      context.arc(x, y, Math.max(3, Math.abs(edgeX - x)), 0, Math.PI * 2);
      context.stroke();
    }

    values.forEach((item) => {
      if (!usable(item)) return;
      const style = applyStyle(context, item.status, scale);
      const [x, y] = point([item.x, item.y]);
      const radius = Math.max(2, style.pointRadius * scale);
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      if (item.status === "detected") {
        context.fill();
      } else {
        context.stroke();
      }
    });
    context.setLineDash([]);
    context.globalAlpha = 1;
    return metrics;
  }

  global.ClassicBodyPoseRenderer = Object.freeze({
    KEYPOINT_NAMES,
    SEGMENTS,
    STYLES,
    buildGeometry,
    render,
  });
})(window);
