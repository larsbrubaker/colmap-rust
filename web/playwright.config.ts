// Playwright config for the COLMAP Rust browser smoke test (testing net 4, see CLAUDE.md).
// Serves the built site (web/dist, from build.sh) with serve.ts and drives real Google Chrome
// (channel "chrome"), as agg-gui's demo does: the bundled chromium-headless-shell has no usable
// GPU path for wgpu. colmap-web *requires* WebGPU, so on Linux (CI's ubuntu-latest has no GPU)
// Chrome is told to expose its SwiftShader (CPU Vulkan) adapter to WebGPU; macOS/Windows use the
// real GPU and need no flags.
import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.PORT ?? 3002);

// WebGPU on a GPU-less Linux runner: unsafe-webgpu turns WebGPU on where Chrome doesn't ship it
// by default (Linux), Vulkan + SwiftShader give it a software adapter, and the surface flag
// avoids needing a display-backed Vulkan swapchain in headless mode.
const LINUX_WEBGPU_ARGS = [
  "--enable-unsafe-webgpu",
  "--enable-unsafe-swiftshader",
  "--enable-features=Vulkan",
  "--use-vulkan=swiftshader",
  "--use-webgpu-adapter=swiftshader",
  "--use-angle=swiftshader",
  "--disable-vulkan-surface",
  "--ignore-gpu-blocklist",
];

const CHROME_USE = {
  ...devices["Desktop Chrome"],
  channel: "chrome",
  headless: true,
  viewport: { width: 1280, height: 800 },
  launchOptions: {
    args: process.platform === "linux" ? LINUX_WEBGPU_ARGS : ["--enable-unsafe-webgpu"],
  },
};

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  retries: 0,
  timeout: 90_000,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    ...CHROME_USE,
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chrome", use: CHROME_USE }],
  webServer: {
    command: "bun run serve",
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
