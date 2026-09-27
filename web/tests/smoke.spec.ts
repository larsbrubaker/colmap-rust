// Browser smoke test (testing net 4): boots the built wasm (web/dist) in headless Chrome and
// checks that the app reaches its ready state (window.__colmapReady / canvas data-ready, set by
// colmap-web after the first presented frame), that nothing threw or logged an error, and that
// the frame the GPU rendered actually shows the app: not one flat colour, with the viewport's red,
// green and blue axis-gizmo lines somewhere in it (colmap-app/src/scene.rs AXIS_COLORS).
// The pixels come from `window.__colmap_frame_stats()`, which `?test=1` installs (colmap-web
// src/probe.rs): a GPU readback, so the assertion doesn't depend on how headless Chrome composites
// the page. Screenshots of the canvas and the page are saved for humans, never asserted on.
// The Pages deploy (.github/workflows/deploy.yml) is gated on this. Run: cd web && bun run test
import { test, expect } from "@playwright/test";

/** Pixel classes of the rendered frame, computed in Rust (colmap-web/src/frame_stats.rs). */
interface FrameStats {
  width: number;
  height: number;
  distinct: number;
  dominantFraction: number;
  red: number;
  green: number;
  blue: number;
}

test("web app boots, reaches ready and paints the viewport", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
  });

  await page.goto("/?test=1");
  await expect(page).toHaveTitle("COLMAP Rust");

  // Ready = a frame has been presented. Fail fast with the collected errors if it never comes
  // (no WebGPU adapter, a panic during start-up, ...).
  await page
    .waitForFunction(() => (window as unknown as { __colmapReady?: boolean }).__colmapReady === true, null, {
      timeout: 45_000,
    })
    .catch((e) => {
      throw new Error(`app never became ready: ${e}\n${errors.join("\n")}`);
    });
  const canvas = page.locator("#canvas");
  await expect(canvas).toHaveAttribute("data-ready", "1");
  await expect(page.locator("#loading")).toBeHidden();

  // Pixel statistics of the next rendered frame, read back from the GPU.
  const stats = await page.evaluate(
    () => (window as unknown as { __colmap_frame_stats: () => Promise<FrameStats> }).__colmap_frame_stats(),
  );
  console.log(`frame stats: ${JSON.stringify(stats)}`);

  // What the compositor shows, for humans (CI uploads test-results/ on every run).
  const canvasPng = await canvas.screenshot({ path: testInfo.outputPath("canvas.png") });
  await testInfo.attach("canvas", { body: canvasPng, contentType: "image/png" });
  await page.screenshot({ path: testInfo.outputPath("page.png") });

  expect(stats.width).toBeGreaterThan(100);
  expect(stats.height).toBeGreaterThan(100);
  expect(stats.distinct, "frame has too few distinct colours (blank?)").toBeGreaterThan(8);
  expect(stats.dominantFraction, "one colour fills the frame (blank?)").toBeLessThan(0.98);
  // The axis gizmo: each axis line covers at least a few pixels in its colour.
  expect(stats.red, "no red (X axis) pixels").toBeGreaterThan(3);
  expect(stats.green, "no green (Y axis) pixels").toBeGreaterThan(3);
  expect(stats.blue, "no blue (Z axis) pixels").toBeGreaterThan(3);

  expect(errors, errors.join("\n")).toEqual([]);
});
