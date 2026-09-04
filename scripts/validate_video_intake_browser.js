const fs = require("fs");
const path = require("path");
const {chromium} = require("playwright");

async function main() {
  const [
    baseUrl,
    videoPath,
    outputDirectory,
    startText = "0",
    durationText = "2",
    prefix = "video_intake",
  ] = process.argv.slice(2);
  if (!baseUrl || !videoPath || !outputDirectory) {
    throw new Error("Usage: validate_video_intake_browser.js BASE_URL VIDEO OUTPUT_DIR");
  }
  fs.mkdirSync(outputDirectory, {recursive: true});
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  });
  const context = await browser.newContext({viewport: {width: 1280, height: 720}});
  const page = await context.newPage();
  const runtimeErrors = [];
  page.on("pageerror", (error) => runtimeErrors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") runtimeErrors.push(message.text());
  });
  const started = Date.now();
  try {
    await page.goto(baseUrl, {waitUntil: "domcontentloaded", timeout: 30000});
    await page.waitForSelector("#video-upload-button", {timeout: 30000});
    await page.setInputFiles("#video-file-input", videoPath);
    await page.click("#video-upload-button");
    await page.waitForFunction(
      () => document.querySelector("#intake-state")?.textContent === "ready",
      null,
      {timeout: 120000},
    );
    const uploadMetadataText = await page.textContent("#upload-metadata");
    await page.fill("#analysis-start-time", startText);
    await page.fill("#analysis-duration", durationText);
    const previewResponsePromise = page.waitForResponse(
      (response) => response.url().includes("/api/video/preview"),
      {timeout: 120000},
    );
    await page.click("#video-preview-button");
    const previewResponse = await previewResponsePromise;
    const preview = await previewResponse.json();
    if (!preview.candidates?.length) throw new Error("No real preview candidates");
    await page.waitForFunction(
      () => document.querySelector("#video-preview-image")?.complete,
      null,
      {timeout: 30000},
    );
    await page.evaluate((selection) => {
      const candidate = selection.candidate;
      const canvas = document.querySelector("#video-preview-canvas");
      const image = document.querySelector("#video-preview-image");
      const canvasRect = canvas.getBoundingClientRect();
      const imageRect = image.getBoundingClientRect();
      const sourceRatio = selection.sourceWidth / selection.sourceHeight;
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
      const centerX = (candidate.bbox[0] + candidate.bbox[2]) / 2;
      const centerY = (candidate.bbox[1] + candidate.bbox[3]) / 2;
      canvas.dispatchEvent(new MouseEvent("click", {
        bubbles: true,
        clientX: canvasRect.left + offsetX + centerX / selection.sourceWidth * shownWidth,
        clientY: canvasRect.top + offsetY + centerY / selection.sourceHeight * shownHeight,
      }));
    }, {
      candidate: preview.candidates[0],
      sourceWidth: preview.width,
      sourceHeight: preview.height,
    });
    await page.waitForFunction(
      () => !document.querySelector("#video-analysis-start-button")?.disabled,
    );
    await page.click("#video-analysis-start-button");
    await page.waitForFunction(
      () => ["completed", "failed", "cancelled"].includes(
        document.querySelector("#video-job-stage")?.textContent,
      ),
      null,
      {timeout: 240000},
    );
    const jobStage = await page.textContent("#video-job-stage");
    const jobMessage = await page.textContent("#video-job-message");
    if (jobStage !== "completed") {
      throw new Error(`Video analysis did not complete: ${jobStage} ${jobMessage}`);
    }
    await page.waitForFunction(
      () => document.querySelector("#video-job-message")?.textContent.includes("直接加载"),
      null,
      {timeout: 30000},
    );
    const analysis = await page.evaluate(async () => (
      await (await fetch("/api/analysis", {cache: "no-store"})).json()
    ));
    const range = await page.evaluate(async () => {
      const response = await fetch("/media/video", {headers: {Range: "bytes=0-127"}});
      return {
        status: response.status,
        acceptRanges: response.headers.get("accept-ranges"),
        byteLength: (await response.arrayBuffer()).byteLength,
      };
    });
    await page.screenshot({
      path: path.join(outputDirectory, `${prefix}_1280x720.png`),
      fullPage: true,
    });
    await page.setViewportSize({width: 1920, height: 1080});
    await page.waitForTimeout(300);
    await page.screenshot({
      path: path.join(outputDirectory, `${prefix}_1920x1080.png`),
      fullPage: true,
    });
    const runtime = analysis.runtime || {};
    const result = {
      schema_version: "factory_video_intake_browser_validation_v1",
      browser: "Google Chrome",
      base_url: baseUrl,
      source_path: path.resolve(videoPath),
      source_name: path.basename(videoPath),
      elapsed_seconds: Number(((Date.now() - started) / 1000).toFixed(3)),
      upload_metadata_text: uploadMetadataText,
      preview_candidate_count: preview.candidate_count,
      selected_candidate_id: preview.candidates[0].candidate_id,
      job_state: jobStage,
      job_message: jobMessage,
      loaded_source_video: analysis.source_video,
      body_provider_status: runtime.pose_provider_status,
      hand_provider: analysis.hand_model?.provider || analysis.hand_model?.runtime_provider,
      hand_detected_frame_count: runtime.hand_detected_frame_count,
      hand_uncertain_frame_count: runtime.hand_uncertain_frame_count,
      hand_missing_frame_count: runtime.hand_missing_frame_count,
      range,
      javascript_runtime_error_count: runtimeErrors.length,
      javascript_runtime_errors: runtimeErrors,
      validation_flags: analysis.validation_flags,
    };
    fs.writeFileSync(
      path.join(outputDirectory, `${prefix}_browser_validation.json`),
      JSON.stringify(result, null, 2),
      "utf8",
    );
    process.stdout.write(JSON.stringify(result));
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
