# COLMAP → Rust porting plan

Open work only; history lives in git. Remove a phase when it is done and green, and prune stale
notes as you go. The contract, testing framework and translation rules are in `CLAUDE.md`;
license decisions are in `docs/LICENSE_AUDIT.md`.

**Reference:** COLMAP 4.2.0 (`REFERENCE`); oracle `pycolmap==4.2.0`; map: colmap-sharp.
**Size:** ~100k lines of C++ in scope (tests, UI and CLI excluded), 142 `*_test.cc` files,
~1,640 gtest cases. colmap-sharp is ~137k lines of C# for the same scope.
**Goal:** photos → camera poses + sparse cloud → dense depth → textured mesh, in an agg-gui app
that runs natively and on GitHub Pages, cancellable with progress. Each phase makes its result
visible in the app so the web build is the integration test.

## Out of scope (do not re-litigate)

Same as colmap-sharp: COLMAP's Qt GUI (`ui/`; our UI is new and agg-gui based), CLI executables
(`exe/`), CUDA/HIP kernels as such (PatchMatch runs on CPU and on WebGPU via WGSL), SiftGPU,
ONNX learned features (ALIKED, LightGlue, LoMa, AnyCalib), Caspar GPU BA, LSD line detection and
`estimators/coordinate_frame` (AGPL), CGAL code paths, SQLite database files, `download`
support, `retrieval/` vocabulary-tree matching.

## Phases (dependency order)

Each phase ends with its ported tests green, `file_compliance` green, the core building for
wasm32, and its app view covered by a headless UI test.

### Phase 0 — Scaffold, testing framework, app shell, deploy
- Cargo workspace (`colmap-rust`, `colmap-gpu`, `colmap-app`, `colmap-app-test`,
  `colmap-native`, `colmap-web`), dev-profile opt-levels as in agg-gui/AtomArtist, agg-gui
  from crates.io with a commented `[patch.crates-io]` to `../agg-gui`.
- `file_compliance` test (800 non-empty lines, header comment, no conflict markers).
- `colmap-app-test` headless harness with a first test.
- An app that opens natively and in the browser: title bar, a placeholder-free "About /
  pipeline stages" panel listing the phases and their status, and a 3D viewport widget
  (agg-gui-wgpu custom render) drawing an axis gizmo — the canvas later phases fill.
- `web/`: index.html, build script (wasm-pack, WebGPU backend, release), bun + Playwright
  smoke test (non-blank canvas, ready flag).
- GitHub Actions: `ci.yml` (fmt check, clippy, test, wasm32 core build on Linux/macOS/Windows)
  and `deploy.yml` (build → Playwright → Pages).

### Phase 1 — Math and linear algebra foundation
Port colmap-sharp's `LinearAlgebra/` (fixed-size vectors/matrices, quaternion, angle-axis,
dynamic `VectorXd`/`MatrixXd`, QR, SVD, symmetric eigen, LU, LLᵀ/LDLᵀ) and COLMAP's `math/`
(`math.h`, `random` = mt19937 + libc++ distributions, `polynomial`, `union_find`,
`connected_components`, `spanning_tree`, `graph_cut`, `matrix.h`). Tests: `math/*_test.cc`
plus the decomposition oracle fixtures. App: none required (a "Diagnostics" page showing the
determinism probe results native vs. web is enough).

### Phase 2 — Geometry
`geometry/`: `rigid3`, `sim3`, `pose`, `essential_matrix`, `homography_matrix`,
`triangulation`, `normalization`, `bbox`, `gps`, `pose_prior`.
App: camera-frustum rendering in the 3D viewport.

### Phase 3 — Sensor
`sensor/models` (all camera models, Tier A incl. iterative undistortion), `rig`, `specs`,
`bitmap` (pixel buffer; the app decodes images), EXIF focal-length reader.
App: image loading (native file dialog / browser file picker + drag-drop), an image browser,
and a camera-model undistortion preview.

### Phase 4 — Scene
`point2d`, `point3d`, `track`, `camera`, `frame`, `image`, `rig`, `correspondence_graph`,
`two_view_geometry`, `pose_graph`, `projection`, `visibility_pyramid`, `reconstruction`,
`reconstruction_io` (binary + text), `reconstruction_manager`, `reconstruction_pruning`,
`synthetic`, in-memory `database`, `database_cache`, `scene_clustering`,
`reconstruction_clustering`.
App: open a COLMAP model folder / zip (native + browser) and show cameras + sparse points;
bundled sample model on the web page.

### Phase 5 — Optimization primitives
`optim/`: samplers, `ransac`, `loransac`, `support_measurement`, `sprt`,
`least_absolute_deviations`, `tiny_solver`, `sparse_cholesky` (port colmap-sharp's simplicial
Cholesky + AMD).

### Phase 6 — Minimal solvers and estimators
`estimators/solvers/*` (PoseLib parts, BSD-3), `two_view_geometry`, `pose`, `generalized_pose`,
`triangulation`, `alignment`, `fundamental_matrix_degensac`, `rotation_averaging`,
`global_positioning`, `gravity_refinement`, `view_graph_calibration`.

### Phase 7 — Nonlinear least squares (Ceres replacement)
Port colmap-sharp's `Solver/` (Jets, residual blocks, manifolds, losses, LM trust region,
Schur dense/sparse/iterative, summary). Riskiest phase; Tier C against pycolmap BA.

### Phase 8 — Bundle adjustment
`estimators/bundle_adjustment*` (CPU), `cost_functions/*`, `ceres_loss_function`, `covariance`.
App: run BA on a loaded model with live cost plot and progress/cancel.

### Phase 9 — Features
`feature/sift` (VLFeat CPU SIFT, BSD-2), `types`, `utils`, `matcher` (brute force + exact k-NN,
cross-check, ratio, guided), `extractor`, `index`.
App: keypoint overlay on images; match viewer between two images.

### Phase 10 — Incremental SfM
`sfm/observation_manager`, `incremental_triangulator`, `incremental_mapper(_impl)`,
`controllers/incremental_pipeline`, `controllers/bundle_adjustment`.
App: photos → sparse model end to end, with the model growing live in the viewport.

### Phase 11 — Pipeline controllers
`controllers/pairing` (exhaustive, sequential, spatial), `feature_extraction`,
`feature_matching(_utils)`, `matcher_cache`, `image_reader`, `undistorters`,
`automatic_reconstruction` (minus CGAL/GPU-only branches).

### Phase 12 — Dense reconstruction (MVS)
`image/undistortion`, `image/warp`, `mvs/*`, CPU PatchMatch, `fusion`, `poisson_meshing`
(PoissonRecon, MIT), `delaunay_meshing` (port colmap-sharp's tetrahedralization + COLMAP's
graph cut), `mesh_simplification`, `texture_mapping`.
App: depth/normal map viewer, fused cloud, mesh display, export (PLY/OBJ/STL).

### Phase 13 — GPU PatchMatch (colmap-gpu)
Port colmap-sharp Phase 13: the compute seam, its WGSL kernels (reuse as-is where possible),
planner, orchestrator, CPU twin, conformance checker. Device comes from agg-gui-wgpu, so it
runs on Metal/DX12/Vulkan natively and WebGPU in the browser.

### Phase 14 — Global and hierarchical mapping
`sfm/global_mapper`, `controllers/global_pipeline`, `hierarchical_pipeline`,
`rotation_averaging` controller.

## Skipped tests

Mirror colmap-sharp's "Skipped tests" list as each module is ported (bitmap file I/O,
SQLite files, Ceres internals not ported, CUDA/GPU, SiftGPU, ONNX, vocabulary tree), adapting
reasons to Rust. Every skipped COLMAP test is listed here by name when its file is ported.

## Decisions
- **Web runs single-threaded.** GitHub Pages can't send the COOP/COEP headers that
  SharedArrayBuffer (and so wasm threads / rayon) needs, and we accept that cost: on the web,
  long work is chunked across frames (or moved to a plain Web Worker without shared memory).
  Native uses rayon. A `coi-serviceworker` shim could enable isolation on Pages later; not now.
- **Web shell comes from agg-gui.** A published `agg-gui-web-shell` crate (canvas,
  requestAnimationFrame loop, DOM input, WebGPU surface) lives in the agg-gui repo; colmap-web
  uses it rather than carrying its own copy.
- **colmap-rust will be published to crates.io** once the core is usable (crate `colmap-rust`,
  lib `colmap_rust`); keep its package metadata publish-ready.
