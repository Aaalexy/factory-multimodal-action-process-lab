const fs = require("fs");
const path = require("path");
const {chromium} = require("playwright");

async function main() {
  const [baseUrl, outputDirectory] = process.argv.slice(2);
  if (!baseUrl || !outputDirectory) {
    throw new Error("Usage: validate_relock_ui_real_evidence.js BASE_URL OUTPUT_DIR");
  }
  fs.mkdirSync(outputDirectory, {recursive: true});
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  });
  const page = await (await browser.newContext({
    viewport: {width: 1280, height: 720},
  })).newPage();
  const runtimeErrors = [];
  page.on("pageerror", (error) => runtimeErrors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") runtimeErrors.push(message.text());
  });
  const result = {
    browser: "Google Chrome",
    evidence_source: "real uploaded-video Body Pose frame",
    hardware_claim: "not_applicable_ui_only_fixture",
  };
  try {
    await page.goto(baseUrl, {waitUntil: "domcontentloaded", timeout: 30000});
    await page.waitForFunction(
      () => state.frames.some((item) => Array.isArray(item.bbox) && item.bbox.length === 4),
      null,
      {timeout: 30000},
    );
    const fixture = await page.evaluate(() => {
      const frame = state.frames.find((item) => (
        Array.isArray(item.bbox) && item.bbox.length === 4
      ));
      if (!frame) throw new Error("No real Body Pose frame with bbox");
      const handRecords = state.handFrames.filter(
        (item) => Number(item.frame_index) === Number(frame.source_frame_index),
      );
      const candidate = {
        candidate_id: "C-real-frame-1",
        candidate_token: "ui-only-real-evidence-token",
        session_id: "ui-only-real-evidence-session",
        frame_sequence: 1,
        bbox: frame.bbox,
        confidence: Number(frame.person_confidence || 0),
        expiry: Date.now() + 5000,
        source_width: state.analysis.source_video.width,
        source_height: state.analysis.source_video.height,
        mirror_horizontal: false,
      };
      showCameraMode();
      state.cameraState = "live";
      state.cameraSessionId = null;
      const packet = {
        sequence: 1,
        width: state.analysis.source_video.width,
        height: state.analysis.source_video.height,
        evidence: {
          frame: {...frame, anonymous_candidates: [candidate]},
          hand_pose_frames: handRecords,
          stable_action: {
            action: "transition",
            raw_action: frame.action || "transition",
            anatomical_side: frame.anatomical_side || "bilateral",
            duration_seconds: 0,
            display_eligible: false,
            status: "uncertain",
            training_eligible: false,
            source_frame_indices: [frame.source_frame_index],
            temporal_reason: "ui_only_real_evidence_fixture",
          },
          body_model: state.analysis.body_model,
          hand_model: state.analysis.hand_model,
        },
      };
      renderCameraEvidence(packet);
      beginCameraPersonSelection();
      return {
        candidate,
        sourceWidth: packet.width,
        sourceHeight: packet.height,
        personRef: frame.person_ref,
        lockEpoch: frame.lock_epoch,
      };
    });
    await page.evaluate(({candidate, sourceWidth, sourceHeight}) => {
      const canvas = document.querySelector("#pose-canvas");
      const rect = canvas.getBoundingClientRect();
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
      const x = (candidate.bbox[0] + candidate.bbox[2]) / 2;
      const y = (candidate.bbox[1] + candidate.bbox[3]) / 2;
      canvas.dispatchEvent(new MouseEvent("click", {
        bubbles: true,
        clientX: rect.left + offsetX + x / sourceWidth * shownWidth,
        clientY: rect.top + offsetY + y / sourceHeight * shownHeight,
      }));
    }, fixture);
    result.confirm_enabled_after_click = !await page.isDisabled(
      "#camera-confirm-relock-button",
    );
    result.selected_message = await page.textContent("#manual-relock-message");
    await page.screenshot({
      path: path.join(outputDirectory, "stage_c_relock_ui_real_evidence_1280x720.png"),
      fullPage: true,
    });
    await page.click("#camera-cancel-relock-button");
    await page.waitForFunction(
      () => document.querySelector("#camera-confirm-relock-button")?.disabled === true,
    );
    result.confirm_disabled_after_cancel = await page.isDisabled(
      "#camera-confirm-relock-button",
    );
    result.person_after_cancel = await page.textContent("#person-ref");
    result.epoch_after_cancel = Number(await page.textContent("#lock-epoch"));
    result.cancel_preserved_person = (
      result.person_after_cancel === fixture.personRef
      && result.epoch_after_cancel === Number(fixture.lockEpoch)
    );
    await page.setViewportSize({width: 1920, height: 1080});
    await page.screenshot({
      path: path.join(outputDirectory, "stage_c_relock_ui_real_evidence_1920x1080.png"),
      fullPage: true,
    });
    result.javascript_runtime_errors = runtimeErrors;
    result.status = (
      result.confirm_enabled_after_click
      && result.confirm_disabled_after_cancel
      && result.cancel_preserved_person
      && runtimeErrors.length === 0
    ) ? "passed" : "failed";
  } finally {
    await browser.close();
  }
  fs.writeFileSync(
    path.join(outputDirectory, "stage_c_relock_ui_real_evidence.json"),
    JSON.stringify(result, null, 2),
  );
  if (result.status !== "passed") {
    throw new Error(JSON.stringify(result));
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
