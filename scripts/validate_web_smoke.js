const fs = require("fs");
const path = require("path");
const {chromium} = require("playwright");

async function main() {
  const [baseUrl, outputDirectory, prefix = "web_smoke"] = process.argv.slice(2);
  if (!baseUrl || !outputDirectory) {
    throw new Error("Usage: validate_web_smoke.js BASE_URL OUTPUT_DIR [PREFIX]");
  }
  fs.mkdirSync(outputDirectory, {recursive: true});
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  });
  const context = await browser.newContext({viewport: {width: 1280, height: 720}});
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  try {
    await page.goto(baseUrl, {waitUntil: "domcontentloaded", timeout: 30000});
    await page.waitForFunction(
      () => document.querySelector("#run-state")?.textContent.includes("available"),
      null,
      {timeout: 30000},
    );
    await page.waitForTimeout(1000);
    const details = await page.evaluate(() => ({
      intakeVisible: Boolean(document.querySelector("#video-file-input")),
      timelineVisible: Boolean(document.querySelector("#action-timeline")),
      handToggleChecked: document.querySelector("#hand-pose-toggle")?.checked,
      bodyRendererMode: document.querySelector("#body-renderer-mode")?.value,
      bodyProviderText: document.querySelector("#provider-states")?.textContent,
      visibleTraceback: document.body.innerText.includes("Traceback"),
      viewport: {width: window.innerWidth, height: window.innerHeight},
    }));
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
    const result = {
      schema_version: "factory_web_smoke_validation_v1",
      browser: "Google Chrome",
      base_url: baseUrl,
      details,
      javascript_runtime_error_count: errors.length,
      javascript_runtime_errors: errors,
    };
    fs.writeFileSync(
      path.join(outputDirectory, `${prefix}.json`),
      JSON.stringify(result, null, 2),
      "utf8",
    );
    process.stdout.write(JSON.stringify(result));
    if (errors.length || details.visibleTraceback) process.exitCode = 2;
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
