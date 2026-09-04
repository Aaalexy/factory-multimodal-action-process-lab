const fs = require("fs");
const path = require("path");
const {chromium} = require("playwright");

async function main() {
  const [baseUrl, outputDirectory] = process.argv.slice(2);
  if (!baseUrl || !outputDirectory) {
    throw new Error("Usage: validate_manual_relock_browser.js BASE_URL OUTPUT_DIR");
  }
  fs.mkdirSync(outputDirectory, {recursive: true});
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  });
  const context = await browser.newContext({viewport: {width: 1280, height: 720}});
  const page = await context.newPage();
  const runtimeErrors = [];
  let latestPacket = null;
  page.on("pageerror", (error) => runtimeErrors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") runtimeErrors.push(message.text());
  });
  page.on("response", async (response) => {
    if (
      response.status() === 200
      && response.url().includes("/api/camera/packet")
    ) {
      try {
        latestPacket = await response.json();
      } catch (_error) {
        // A later valid packet remains authoritative.
      }
    }
  });
  const result = {browser: "Google Chrome", camera_index: 0};
  try {
    await page.goto(baseUrl, {waitUntil: "domcontentloaded", timeout: 30000});
    await page.click("#camera-mode-button");
    await page.waitForFunction(
      () => [
        "live", "no_device", "permission_denied", "error",
      ].includes(document.querySelector("#camera-state")?.textContent || ""),
      null,
      {timeout: 120000},
    );
    const reachedState = await page.textContent("#camera-state");
    if (reachedState !== "live") {
      const status = await page.evaluate(async () => (
        await (await fetch("/api/camera/status", {cache: "no-store"})).json()
      ));
      throw new Error(
        `Camera did not reach live: ${reachedState} ${status.last_error?.message || ""}`,
      );
    }
    await page.waitForFunction(
      () => Number(document.querySelector("#pose-canvas")?.dataset.cameraCandidateCount || 0) > 0,
      null,
      {timeout: 30000},
    );
    const before = {
      person_ref: await page.textContent("#person-ref"),
      lock_epoch: Number(await page.textContent("#lock-epoch")),
    };
    await page.click("#camera-select-person-button");
    await page.waitForFunction(
      () => document.querySelector("#pose-canvas")?.classList.contains("selecting-person"),
    );
    const displayedSequence = Number(
      await page.getAttribute("#pose-canvas", "data-camera-sequence"),
    );
    const deadline = Date.now() + 5000;
    while (
      (!latestPacket || Number(latestPacket.sequence) !== displayedSequence)
      && Date.now() < deadline
    ) {
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    if (!latestPacket || Number(latestPacket.sequence) !== displayedSequence) {
      throw new Error("Could not bind browser candidate to displayed Camera sequence");
    }
    const candidate = latestPacket.evidence?.evidence?.frame?.anonymous_candidates?.[0];
    if (!candidate) throw new Error("Displayed Camera frame has no candidate");
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
      const centerX = (candidate.bbox[0] + candidate.bbox[2]) / 2;
      const centerY = (candidate.bbox[1] + candidate.bbox[3]) / 2;
      canvas.dispatchEvent(new MouseEvent("click", {
        bubbles: true,
        clientX: rect.left + offsetX + centerX / sourceWidth * shownWidth,
        clientY: rect.top + offsetY + centerY / sourceHeight * shownHeight,
      }));
    }, {
      candidate,
      sourceWidth: latestPacket.evidence.width,
      sourceHeight: latestPacket.evidence.height,
    });
    await page.waitForFunction(
      () => !document.querySelector("#camera-confirm-relock-button")?.disabled,
    );
    const selectedMessage = await page.textContent("#manual-relock-message");
    await page.click("#camera-confirm-relock-button");
    await page.waitForFunction(
      ({personRef, lockEpoch}) => (
        document.querySelector("#person-ref")?.textContent !== personRef
        && Number(document.querySelector("#lock-epoch")?.textContent) > lockEpoch
      ),
      before,
      {timeout: 15000},
    );
    const after = {
      person_ref: await page.textContent("#person-ref"),
      lock_epoch: Number(await page.textContent("#lock-epoch")),
    };
    await page.click("#camera-select-person-button");
    await page.click("#camera-cancel-relock-button");
    const cancelled = await page.textContent("#manual-relock-message");
    const afterCancel = {
      person_ref: await page.textContent("#person-ref"),
      lock_epoch: Number(await page.textContent("#lock-epoch")),
    };
    const status = await page.evaluate(async () => (
      await (await fetch("/api/camera/status", {cache: "no-store"})).json()
    ));
    result.before = before;
    result.selected_message = selectedMessage;
    result.after_confirm = after;
    result.after_cancel = afterCancel;
    result.cancel_preserved_person = (
      after.person_ref === afterCancel.person_ref
      && after.lock_epoch === afterCancel.lock_epoch
    );
    result.camera_session_id = status.session_id;
    result.body_provider = status.metrics?.body_pose_provider_status?.active_provider;
    result.body_fallback_active = status.metrics?.body_pose_provider_status?.fallback_active;
    result.hand_provider = status.metrics?.hand_pose_provider;
    result.frame_evidence_sequence_mismatch_count = (
      status.metrics?.frame_evidence_sequence_mismatch_count
    );
    result.javascript_runtime_errors = runtimeErrors;
    await page.screenshot({
      path: path.join(outputDirectory, "stage_c_manual_relock_1280x720.png"),
      fullPage: true,
    });
    await page.setViewportSize({width: 1920, height: 1080});
    await page.screenshot({
      path: path.join(outputDirectory, "stage_c_manual_relock_1920x1080.png"),
      fullPage: true,
    });
    await page.click("#camera-stop-button");
    await page.waitForFunction(
      () => document.querySelector("#camera-state")?.textContent === "stopped",
      null,
      {timeout: 15000},
    );
    result.camera_stopped = true;
    result.status = "passed";
  } finally {
    await browser.close();
  }
  fs.writeFileSync(
    path.join(outputDirectory, "stage_c_manual_relock_browser.json"),
    JSON.stringify(result, null, 2),
  );
  if (runtimeErrors.length) {
    throw new Error(`JavaScript runtime errors: ${runtimeErrors.join(" | ")}`);
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
