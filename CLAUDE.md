# colmap-rust — agent guidelines

A **pure Rust** port of [COLMAP](https://github.com/colmap/colmap) (Structure-from-Motion and
Multi-View Stereo): photos → camera poses → sparse point cloud → dense depth → mesh. It is built
to sit directly on [agg-gui](https://github.com/larsbrubaker/agg-gui) and its wgpu renderer, so
the same app runs natively on Windows, macOS and Linux and in the browser (WebGPU) on GitHub
Pages. Reference version: `REFERENCE` (COLMAP 4.2.0). Remaining work and phase order are in
`PORTING_PLAN.md`.

colmap-sharp (`../../MatterCAD/Submodules/colmap-sharp`, github.com/larsbrubaker/colmap-sharp) is a
finished-through-Phase-13 C# port of the same COLMAP version by the same team. It is our map:
see "Two references" below.

## Philosophy

**Quality through iterations.** Start with correct implementations, then improve. In a porting
project, every function matters.

**Be a collaborator, not a stenographer.** Apply judgment, push back when something looks wrong,
and propose what you believe is the best approach, even if it differs from what was asked.
Explain the trade-offs, then defer once a decision is made.

## The contract (non-negotiable)

1. **Pure Rust.** No C/C++, no `-sys` crates, no native build scripts, no linking to COLMAP or
   anything else. The core crate builds for `wasm32-unknown-unknown`. Every COLMAP dependency
   is ported, replaced by Rust written here, or dropped (`docs/LICENSE_AUDIT.md`).
2. **MIT, commercial-safe.** Port only permissively licensed code. Never port, transcribe or
   "follow along with" GPL/LGPL/AGPL/MPL or non-commercial code — Eigen, CHOLMOD/CSparse, CGAL,
   LSD, SiftGPU, Qt are all out. Check `docs/LICENSE_AUDIT.md` before porting anything that is
   not COLMAP's own source or colmap-sharp's own code, and add the upstream's notice to
   `THIRD_PARTY_NOTICES.md` in the same change.
3. **No stubs.** No `todo!()`, `unimplemented!()`, `panic!("not implemented")`, placeholders or
   partial implementations. If a dependency isn't ported yet, port it first.
4. **COLMAP's tests are the specification, 1:1.** Every `*_test.cc` becomes a Rust test file
   with the same test names (`TEST(Rigid3d, Inverse)` → `fn rigid3d_inverse()`) and the same
   expected values and tolerances. A test that exercises an excluded feature is listed as
   skipped with the reason in `PORTING_PLAN.md`, never silently dropped. Rust-only tests are
   named `rust_only_*` and never stand in for a ported one.
5. **Never weaken a test to make it pass.** Every failure is a real bug, found by
   instrumentation and root-cause analysis. For a ported test, COLMAP's expected value is the
   spec and the Rust output is the bug.
6. **Deliberate divergence is documented.** Anything that behaves differently from COLMAP on
   purpose gets a numbered entry in `docs/CPP_DIVERGENCES.md`: what differs, why, and the
   evidence. Convenience is not a reason.

## Two references

- **COLMAP's C++ (`cpp-reference/`) is the specification** for COLMAP's own code. Always read
  the C++ function *and everything it calls* before porting it. Tests are ported from the
  `*_test.cc` files, not from colmap-sharp's C# tests.
- **colmap-sharp is the worked map.** Before porting a module, read colmap-sharp's port of it,
  its tests and the `CPP_DIVERGENCES.md` entries that cite it: they record every trap already
  found (tie orders, libc++ distribution details, hash iteration order, FMA contraction). For
  the code colmap-sharp had to *write* because COLMAP's dependency is excluded or native
  (linear algebra, sparse Cholesky + AMD, the Ceres subset, Delaunay, the libc++ emulations,
  PatchMatch RNG, the WGSL PatchMatch kernels), port colmap-sharp's code to Rust — it is our own
  MIT code and already validated.
- When the two disagree, COLMAP wins unless colmap-sharp documented a divergence; then take the
  same divergence (full entry here, "Same as colmap-sharp entry N") or fix it and say why.
- A result that differs between colmap-rust and colmap-sharp on the same input is a bug in one
  of them. Find out which.

## How closely must results match?

Bit-exactness with COLMAP is not achievable everywhere, because COLMAP's numbers run through
Eigen and Ceres, which we replace rather than port.

| Tier | Code | Bar |
|---|---|---|
| **A — exact** | Scalar, deterministic code: camera models, pose/transform algebra, PRNG and samplers, track/graph bookkeeping, reconstruction file I/O, SIFT's scale space | Bit-identical to COLMAP on the same input. |
| **B — tolerance** | Code through decompositions (SVD, eigen, QR) or solvers: minimal solvers, triangulation, homography/essential estimation | COLMAP's own test tolerances; oracle fixtures agree within a stated tolerance. Solution *sets* compare order-insensitively only where COLMAP's contract is a set. |
| **C — outcome** | Iterative / randomized pipelines: RANSAC, bundle adjustment, incremental mapping, PatchMatch | Same convergence on oracle fixtures: final cost, registered image count, reprojection error, and poses after Sim3 alignment within stated bounds. |

Tier is decided per function when it is ported, and stated in the test that pins it.

## Testing framework

Four nets. A phase is not done until all that apply are green.

1. **Ported COLMAP tests** — `colmap-rust/tests/<module>/<file>.rs`, one file per `*_test.cc`,
   same test names. Tests MUST exercise production code, never copies of it. Unit tests that
   need private access live in `src/**/tests.rs`. Each module is one test binary,
   `colmap-rust/tests/<module>.rs`, which pulls its files in with
   `#[path = "<module>/<file>.rs"] mod <file>;` (run one with
   `cargo test -p colmap-rust --test <module>`).
2. **Oracle fixtures** — `oracle/` holds Python scripts that run the pinned `pycolmap` wheel
   (`oracle/requirements.txt`) and write inputs and outputs to `colmap-rust/tests/data/oracle/`.
   Fixtures are checked in, so `cargo test` never needs Python. colmap-sharp's checked-in
   fixtures (`ColmapSharp.Tests/TestData/oracle/`) were produced by the same wheel and may be
   copied over as each phase needs them; copy their generator script with them. PRNG-dependent
   fixtures must be generated on macOS (libc++).
3. **Headless UI tests** — `colmap-app-test` builds the real production widget tree
   (`colmap_app::build_app`) with no window and no GPU, drives it with synthetic agg-gui events,
   and asserts on app state and the widget tree via agg-gui reflection. Every UI behavior and
   every UI bug fix gets one. (Same design as AtomArtist's `atomartist-ui-test`.)
4. **Browser smoke test** — Playwright (`web/tests/`) boots the built wasm in headless Chrome,
   checks the canvas paints something non-blank and the app reaches its ready state. The GitHub
   Pages deploy is gated on it, so a broken web build never ships.

Plus the gates that run with net 1:
- `file_compliance` (`colmap-rust/tests/file_compliance.rs`): every `.rs`, `.wgsl`, `.py`,
  `.sh`, `.ts` and `.js` file in the workspace has at most **800 non-empty lines**, starts
  with a header comment, and contains no merge-conflict markers. No exemptions, ever.
- `cargo build -p colmap-rust --target wasm32-unknown-unknown` must stay green.
- `cargo clippy --workspace -- -D warnings`.

### Test-first bug fixing

1. Write a failing test that reproduces the bug.
2. Fix the root cause with the minimal change.
3. Watch the test pass. Never commit a fix without its test.

## Rust translation rules

- **No FMA.** Never use `f64::mul_add` / `f32::mul_add` in math paths; Rust never contracts on
  its own, so results are identical on every platform. Explicit SIMD is allowed only when each
  lane repeats the scalar IEEE operations in the same order, with a test pinning bit-identity to
  the scalar path. Apple clang contracts `a*b + c` by default, so when a Tier A oracle diff
  lands exactly on a multiply-add, suspect the C++ side first and record it.
- **Transcendentals.** `sin`/`cos`/`exp`/`log`/`atan2`/`pow` (f64, and the `sinf`/`expf`/...
  f32 forms) always go through `colmap_rust::math::fns`, never the `f64`/`f32` methods. It is
  backed by the pure-Rust `libm` crate, which gives the same bits native and wasm (std's does
  not); the cost is 1-2 ulp differences from Apple libm, so Tier A oracle tests of code that
  calls transcendentals may need a tolerance and cite `docs/CPP_DIVERGENCES.md` entry 1.
  `tests/math/fns_probe.rs` pins the bits on every CI platform, natively and as a
  `wasm32-wasip1` build under wasmtime (`CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test
  -p colmap-rust --target wasm32-wasip1 --test math`), so native == wasm is enforced.
- **Hash iteration order.** `std::collections::HashMap`/`HashSet` are randomly seeded; never
  iterate one where the order can reach an output. Use `BTreeMap`, a sorted `Vec`, an
  insertion-ordered map, or — where COLMAP's own order matters — the libc++ `unordered_map`
  emulation (port of colmap-sharp's `LibcxxUnorderedMap`). Document which in a comment.
- **Stable sorts.** `std::sort` is unstable; Rust's `sort_unstable*` differs from it among
  ties. Where tie order can reach an output, sort with an explicit tie-break (usually the
  index) and note it. `std::stable_sort` → `sort_by` (stable).
- **PRNG.** COLMAP uses `std::mt19937` + libc++ distributions. Port colmap-sharp's exact
  mt19937 and libc++ distribution ports; don't use the `rand` crate in the core.
- **Numeric constants.** `std::numeric_limits<double>::epsilon()` is `f64::EPSILON`.
  `float` stays `f32` — COLMAP stores descriptors, bitmaps and depth maps in single precision.
- **Casts.** `static_cast<int>(x)` truncation is `x as i32`, but `as` *saturates* on
  overflow/NaN where C++ is UB; where it can matter, comment it. Unsigned wrap uses
  `wrapping_*`; don't rely on debug-build overflow panics being absent.
- **Eigen semantics.** Quaternions are stored `(x, y, z, w)` in Eigen memory but constructed
  `(w, x, y, z)`; COLMAP's file formats write `qw qx qy qz`. Matrices are column-major.
  `normalized()` of a zero vector returns zero. Match documented behavior, don't port Eigen.
- **Value types.** `Vector3d`, `Matrix3d`, `Rigid3d`, `Quaterniond` are `Copy` structs with
  operator impls. Hot loops don't allocate.
- **Threading.** Parallelism uses rayon behind the `parallel` feature, only where each worker
  writes its own slot. Sequential and parallel runs give identical results; wasm runs
  sequential.
- **Errors.** `THROW_CHECK*` → `check!`-style macros that return `Err` with COLMAP's
  "Check failed: …" message in library code (panic only where COLMAP's contract is an
  internal invariant). `LOG(FATAL)` → error. Never swallow.
- **Cancellation and progress** go through a `CancelToken` and a progress callback, because
  the app shows progress and lets the user cancel, and the browser must stay responsive.
- **C++ → Rust patterns:** pointer graphs → arenas (`Vec<T>` + index ids);
  `std::vector` → `Vec`; `std::array` → `[T; N]`; templates → generics/traits;
  `std::optional` → `Option`; `std::shared_ptr` graph ownership → ids into owning maps.

## Architecture

```
colmap-rust/        core library (lib `colmap_rust`): pure Rust, std only, no GUI, no GPU.
                    Modules mirror COLMAP: math/ geometry/ sensor/ scene/ optim/ estimators/
                    feature/ sfm/ mvs/ controllers/ util/, plus linalg/ and solver/
                    (the Eigen and Ceres replacements).
colmap-gpu/         wgpu compute (WGSL PatchMatch, later SIFT/matching) behind a device seam.
                    Takes a wgpu Device/Queue from the host — the app passes agg-gui-wgpu's.
colmap-app/         the agg-gui application: widgets, views (image browser, feature/match
                    viewer, 3D sparse/dense viewer via agg-gui-wgpu custom rendering),
                    pipeline runner with progress/cancel. `build_app()` is shared by both shells.
colmap-app-test/    headless UI test harness over colmap-app (testing net 3).
colmap-native/      desktop binary: agg-gui-shell (winit + wgpu).
colmap-web/         wasm cdylib: canvas + requestAnimationFrame shell, WebGPU.
web/                index.html, bun + Playwright smoke tests, Pages staging.
oracle/             pycolmap fixture generators (test-time only).
```

The app is agg-gui from the first commit — there is no JavaScript UI. The core library never
depends on agg-gui or wgpu, so it stays usable from other hosts (and testable without a GPU).

agg-gui is a **path dependency** (`../agg-gui/<crate>`, a sibling checkout in rust-apps) so
colmap-rust can track agg-gui `main` or a development branch without waiting for a crates.io release.
Only the app crates (`colmap-app`, `colmap-app-test`, `colmap-native`, `colmap-web`) use it; the core
`colmap-rust` crate never depends on agg-gui, so it stays publishable on its own. CI checks agg-gui
out next to this repo (`AGG_GUI_REF` in the workflows, default `main`).

## agg-gui is ours: fix it upstream

colmap-rust is allowed, and expected, to change agg-gui (`../agg-gui`, github.com/larsbrubaker/agg-gui)
from here. When the app hits an agg-gui bug, a missing feature, or a wrong doc, fix it in agg-gui.
Don't work around it in colmap-app.

- Do the agg-gui change as its own step, following agg-gui's own CLAUDE.md: a test-first bug fix,
  the 800-line limit, and its CI green (including its Pages demo and Playwright tests when the
  change touches rendering or the web shell). Commit it in agg-gui (on `main`, or on a branch that
  colmap-rust's CI points `AGG_GUI_REF` at) *before* the colmap-rust commit that relies on it, and
  push agg-gui first so colmap-rust's CI can see it.
- There is no publish round trip. Publishing agg-gui crates to crates.io is a separate decision
  (Lars confirms each publish), never something colmap-rust waits on.
- A short-lived workaround is allowed only while the upstream fix is in flight. It carries a
  `// agg-gui workaround:` comment naming the agg-gui change that removes it, and is deleted when
  that change lands.
- Implementers who find an agg-gui gap report it and don't patch agg-gui themselves unless their
  brief says so. The orchestrator schedules the upstream step (queue: `PORTING_PLAN.md`).

## Coding standards

- **800-line limit per file**, enforced by `file_compliance`. No exemptions. If a port would
  exceed it, split by responsibility (see the `file-size-refactoring` skill). Never trim
  comments or blank lines to fit.
- **Every file starts with a header**: what it is, the C++ file(s) it ports, and how it relates
  to its neighbors.
- Comments explain *why*. Keep COLMAP's non-obvious comments, since they carry the reasoning.
  Doc-comment the C++ name when it differs enough to be hard to find (`/// Port of
  colmap::EstimateRigid3d`).
- `Result`/`Option` over `unwrap` in library code; `expect` is fine in `main` for startup.
- No `unsafe` unless there is no alternative; document every `unsafe` block.
- **Performance: measure, never guess.** Profile the real workload before optimizing and keep a
  change only if the numbers move. UI frames stay under 10 ms; long work never runs on the UI
  frame (native: worker thread; web: chunked cooperative steps or a worker).
- **Icons:** Font Awesome via Unicode code points, as in agg-gui.

## Commands

```bash
cargo build --workspace
cargo test --workspace
cargo test -p colmap-rust --test <module>            # one module's ported tests
cargo test -p colmap-rust <name> -- --exact --nocapture
cargo test -p colmap-rust --test file_compliance
cargo build -p colmap-rust --target wasm32-unknown-unknown
cargo clippy --workspace -- -D warnings
cargo run -p colmap-native                            # desktop app
web/build.sh && (cd web && bun run serve)             # web app at localhost
(cd web && bunx playwright test)                      # browser smoke test
scripts/fetch-reference.sh                            # C++ reference into cpp-reference/
oracle/setup.sh && oracle/.venv/bin/python oracle/<script>.py   # regenerate fixtures
```

## Git

Commit on `main`; push to `origin` (github.com/larsbrubaker/colmap-rust). Pushing `main`
deploys the web app to GitHub Pages once the smoke test passes.

## Orchestration pattern

The main session acts as **planner and orchestrator only**; it does not write or edit code.
All implementation is delegated to the `implementer` subagent (`.claude/agents/implementer.md`),
one scoped step at a time. All post-change review is delegated to the `reviewer` subagent
(`.claude/agents/reviewer.md`). When tests fail, use the `fix-test-failures` agent, which treats
every failure as a real bug. The main session handles planning, architecture decisions, and
synthesizing subagent results.

Brief each implementer with one deliverable and a 25-minute budget; a run over 30 minutes is an
error. A deliverable is one coherent slice of a `PORTING_PLAN.md` phase with its ported tests.
Implementers that may run concurrently get `isolation: "worktree"`.

Working conventions that every brief repeats:
- A new worktree starts with `git reset --hard main`. `cpp-reference/` and `oracle/.venv` are
  git-ignored, so worktrees read them from the main checkout. On a fresh machine run
  `scripts/fetch-reference.sh` and `oracle/setup.sh` first.
- Implementers commit on their own branch and never push or edit `PORTING_PLAN.md`; they list
  what their change makes stale, and the orchestrator prunes the plan when it merges.
- Divergence entry numbers are stable and never reused. Hand each concurrent implementer its own
  range. Merge `docs/CPP_DIVERGENCES.md` entry by entry, then check no conflict marker is left
  (`file_compliance` also rejects them).
- Keep headers, comments and divergence entries true to the final code in the same commit.
- Every phase that lands library capability also lands a way to see it in the app (a view,
  an overlay, a pipeline stage), with a headless UI test — that is how the web build tests the
  port "robustly online".
