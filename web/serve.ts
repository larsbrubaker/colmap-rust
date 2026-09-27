// Static file server for web/dist (the built site): `bun run serve` → http://localhost:3002.
// Used by hand and as Playwright's webServer. PORT overrides the port. Serves `.wasm` with
// application/wasm (needed for streaming instantiation) and never lists directories.
import { join, normalize } from "path";

const DIST = join(import.meta.dir, "dist");
const port = Number(process.env.PORT ?? 3002);

Bun.serve({
  port,
  hostname: "127.0.0.1",
  async fetch(req) {
    let path = decodeURIComponent(new URL(req.url).pathname);
    if (path.endsWith("/")) path += "index.html";
    const full = normalize(join(DIST, path));
    if (!full.startsWith(DIST)) return new Response("Forbidden", { status: 403 });
    const file = Bun.file(full);
    if (!(await file.exists())) return new Response("Not found", { status: 404 });
    const headers: Record<string, string> = { "Cache-Control": "no-store" };
    if (full.endsWith(".wasm")) headers["Content-Type"] = "application/wasm";
    return new Response(file, { headers });
  },
});
console.log(`Serving ${DIST} at http://127.0.0.1:${port}/`);
