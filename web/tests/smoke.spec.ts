// Browser smoke test (testing net 4): boots the built wasm (web/dist) in headless Chrome and
// checks that the app reaches its ready state (window.__colmapReady / canvas data-ready, set by
// colmap-web after the first presented frame), that nothing threw or logged an error, and that
// the canvas actually shows the app: not one flat colour, with the viewport's red, green and
// blue axis-gizmo lines somewhere in it (colmap-app/src/scene.rs AXIS_COLORS).
// The Pages deploy (.github/workflows/deploy.yml) is gated on this. Run: cd web && bun run test
import { test, expect, type Page } from "@playwright/test";

/** Pixel classes counted in the canvas screenshot. */
interface PixelStats {
  width: number;
  height: number;
  distinct: number;
  dominantFraction: number;
  red: number;
  green: number;
  blue: number;
}

/**
 * Decode a PNG in the page (no Node image dependency) and classify its pixels. "Red" means the
 * red channel beats both others by a wide margin, and so on; the dark theme's greys and the
 * text never qualify.
 */
async function pixelStats(page: Page, png: Buffer): Promise<PixelStats> {
  const dataUrl = `data:image/png;base64,${png.toString("base64")}`;
  return page.evaluate(async (url) => {
    const img = new Image();
    img.src = url;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.width;
    c.height = img.height;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const { data } = ctx.getImageData(0, 0, c.width, c.height);
    const buckets = new Map<number, number>();
    let red = 0, green = 0, blue = 0;
    const margin = 60;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
      buckets.set(key, (buckets.get(key) ?? 0) + 1);
      if (r > g + margin && r > b + margin) red++;
      else if (g > r + margin && g > b + margin) green++;
      else if (b > r + margin && b > g + margin) blue++;
    }
    const total = data.length / 4;
    const dominant = Math.max(...buckets.values());
    return {
      width: c.width,
      height: c.height,
      distinct: buckets.size,
      dominantFraction: dominant / total,
      red,
      green,
      blue,
    };
  }, dataUrl);
}

test("web app boots, reaches ready and paints the viewport", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(`console.error: ${msg.text()}`);
  });

  await page.goto("/");
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

  // One more frame, then capture what the compositor shows (works for WebGPU canvases, where
  // toDataURL would read an already-presented, cleared texture).
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => r(null))));
  const png = await canvas.screenshot();
  await testInfo.attach("canvas", { body: png, contentType: "image/png" });

  const stats = await pixelStats(page, png);
  console.log(`canvas stats: ${JSON.stringify(stats)}`);
  expect(stats.width).toBeGreaterThan(100);
  expect(stats.height).toBeGreaterThan(100);
  expect(stats.distinct, "canvas has too few distinct colours (blank?)").toBeGreaterThan(8);
  expect(stats.dominantFraction, "one colour fills the canvas (blank?)").toBeLessThan(0.98);
  // The axis gizmo: each axis line covers at least a few pixels in its colour.
  expect(stats.red, "no red (X axis) pixels").toBeGreaterThan(3);
  expect(stats.green, "no green (Y axis) pixels").toBeGreaterThan(3);
  expect(stats.blue, "no blue (Z axis) pixels").toBeGreaterThan(3);

  expect(errors, errors.join("\n")).toEqual([]);
});
