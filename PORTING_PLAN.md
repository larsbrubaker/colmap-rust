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

## Resume here

**On `main`, by phase:**
- Phase 0: scaffold, app shells (native, web, headless harness), CI.
- Phase 1: util, math, linalg, including the sparse CSC matrix, AMD ordering and simplicial
  Cholesky.
- Phase 2: geometry (the `Rigid3dNear` / `Sim3dNear` matchers exist as test support).
- Phase 3: all of sensor/.
- Phase 4: point2d, point3d, track, camera, frame, image, projection, visibility_pyramid,
  two_view_geometry, correspondence_graph, the reconstruction core (bounds, Normalize,
  Transform, Crop), binary + text reconstruction I/O, synthetic datasets (with and without a
  database, SynthesizeNoise, SynthesizeImages), `Reconstruction::Load`,
  `TranscribeImageIdsToDatabase`, rig configs (`ReadRigConfig`, `ApplyRigConfig`), PLY I/O (`util/ply`), ConvertToPLY / ImportPLY /
  ExtractColors / CreateImageDirs, every `reconstruction_io.cc` exporter, `pose_graph`, `reconstruction_manager`,
  `reconstruction_pruning`, `scene_clustering`, `reconstruction_clustering`, the in-memory `Database` (with Merge and LoadRandomDatabaseDescriptors) and `DatabaseCache`.
- Phase 5: all of optim/ (RANSAC family, sparse_cholesky, least_absolute_deviations,
  tiny_solver).
- Phase 6: `SimilarityTransformEstimator`, `poselib_utils`, the PoseLib CameraPose,
  PoseLibCamera, `ImagePair`, essential (motion from E, cheirality), sturm, re3q3, p3p, gp3p,
  p4pf, relpose_5pt, relpose_6pt_shared_focal, relpose_6pt_onesided_focal and gen_relpose_6pt
  solvers, and COLMAP's
  `absolute_pose` (P3P, P4Pf, EPnP, `ComputeSquaredReprojectionError`),
  `generalized_absolute_pose` (GP3P), `affine_transform`, `translation_transform`,
  `homography_matrix` (points and rays), `fundamental_matrix` (7- and 8-point, Sampson
  refiner), `generalized_relative_pose` (GR6P, GR8P), estimators `triangulation`,
  `fundamental_matrix_degensac`, `pose` (EstimateAbsolutePose, EstimateRelativePose) and
  `generalized_pose` (absolute, relative, structure-less), `two_view_geometry` (all of it), `alignment` (all of it), `rotation_averaging` (all of it), `relpose_shared_focal`, `relpose_one_sided_focal` and
  `essential_matrix` (five-point, eight-point, tangent
  Sampson); `cost_functions` tiny_manifold, the closed-form tiny_sampson_error functors and
  quaternion_utils (double forms).
- Phase 9: feature types and utils, VLFeat's CPU SIFT, `SiftCPUFeatureExtractor`, the
  `FeatureExtractor` / `FeatureMatcher` traits, the exact k-NN descriptor index,
  `SiftCPUFeatureMatcher` with guided matching, and the SIFT text loader.
- Phase 13: the compute seam in colmap-gpu (`ComputeDevice`, `WebGpuComputeDevice`).

**Setting up on another machine:** clone `rust-apps` with submodules (colmap-rust needs `../agg-gui`
as a sibling); in colmap-rust run `scripts/fetch-reference.sh` (and `oracle/setup.sh` only if you
regenerate fixtures). Agent worktrees live in `.claude/worktrees/`, so create the git-ignored
symlinks `.claude/worktrees/agg-gui`, `.claude/worktrees/agg-rust` and
`.claude/worktrees/tess2-rust`, each pointing at the rust-apps checkout of the same name, or the
app crates won't resolve.
Divergence-log merges: keep both sides' entries in numeric order (entries are independent sections).
After merging `THIRD_PARTY_NOTICES.md`, check the ```` ``` ```` fences are balanced.

**Next slices** (start each from `main`):
- Phase 6 — the estimators layer (see Phase 6 below).
- Phase 4 — the exporter oracle fixture and the app view (see Phase 4 below).
- Phase 9 — `CovariantSiftCPUFeatureExtractor` (covdet's detector and shape are on main).
- Phase 7 — the Ceres replacement.

**Other open items:**
- colmap-rust is a rust-apps submodule (pushed); after cloning rust-apps,
  `git -C colmap-rust checkout main && git -C colmap-rust pull`. Only `main` exists on origin;
  agents work on local worktree branches that are merged to `main` and then deleted.

## Out of scope (do not re-litigate)

Same as colmap-sharp: COLMAP's Qt GUI (`ui/`; our UI is new and agg-gui based), CLI executables
(`exe/`), CUDA/HIP kernels as such (PatchMatch runs on CPU and on WebGPU via WGSL), SiftGPU,
ONNX learned features (ALIKED, LightGlue, LoMa, AnyCalib), Caspar GPU BA, LSD line detection and
`estimators/coordinate_frame` (AGPL), CGAL code paths, SQLite database files, `download`
support, `retrieval/` vocabulary-tree matching.

## Phases (dependency order)

Each phase ends with its ported tests green, `file_compliance` green, the core building for
wasm32, and its app view covered by a headless UI test.

### Phase 0 — Scaffold (done except CI confirmation)
- First CI and Pages runs on GitHub: confirm the Linux SwiftShader WebGPU flags in
  `web/playwright.config.ts` get an adapter, and that `colmap-native` builds on Linux and Windows.

### Phase 1 — Math, linear algebra, util
- The rest of `util/` that later phases need: `IdMap`, the libc++ `unordered_map` emulation,
  `enum_utils`, `timestamp` (port each when its first user lands).
- `colmap-web` installs `util::timer::set_clock_source` from `performance.now()`.

### Phase 2 — Geometry
Remaining: `rigid3_matchers_test.cc` and `sim3_matchers_test.cc` (the matchers themselves are
in `tests/support/transform_matchers.rs`).
App: camera-frustum rendering in the 3D viewport.

### Phase 3 — Sensor
App: image loading (native file dialog / browser file picker + drag-drop), an image browser,
and a camera-model undistortion preview.

### Phase 4 — Scene
Remaining:
- A pycolmap oracle fixture for the NVM, Bundler, Cam and Recon3D exporters' text (COLMAP's tests
  only check return values; the layout is pinned by Rust-only tests).

App: open a COLMAP model folder / zip (native + browser) and show cameras + sparse points;
bundled sample model on the web page. The library side is ready (`Reconstruction::read`).

### Phase 6 — Minimal solvers and estimators
Remaining:
- A test that pins `MaybeDecomposeRelativePoses`' unit-length translation on a PLANAR pair
  (nothing catches it today: essential decompositions are unit length already).

Estimators with a C++ `Refine` implement `optim::LocalEstimator`; those without opt in via
`EstimateAsLocal`.

### Phase 7 — Nonlinear least squares (Ceres replacement)
Port colmap-sharp's `Solver/` (Jets, residual blocks, manifolds, losses, LM trust region,
Schur dense/sparse/iterative, summary) into `solver/`. Riskiest phase; Tier C against pycolmap BA.
On main: the general `Jet<N>`, `Scalar` (f64 and Jets; the camera models use it), Ceres'
rotation conversions, `CostFunction`, `AutoDiffCostFunction`, `TinySolverAutoDiffFunction`, the
`sampson_error.h` functors, the small BLAS kernels (`Operation` enum, const-generic fixed
kernels), `LinearOperator`, `SparseMatrix`/`DenseSparseMatrix`, the compressed-row block
structure, `BlockSparseMatrix`, and the dense, sparse and diagonal `BlockRandomAccessMatrix`
(cells resolved once to a `CellLocation`). Slices left, in dependency order:
2. Losses and manifolds: LossFunctions, Manifolds, SphereProductManifolds, CeresPolynomial
   (tests LossFunction, Corrector, Manifold, CeresPolynomial).
3. The problem model: ParameterBlock, ResidualBlock, Problem, Problem.Evaluate,
   ParameterBlockOrdering, Program, `BlockJacobianLayout` (tests Problem, OrderedGroups, ParameterBlockOrdering;
   colmap-sharp entries 36, 37, 53).
4. Evaluator and basic linear solvers: ProgramEvaluator, SolverTypes, LinearSolvers (tests
   ConjugateGradientsSolver, the LinearLeastSquaresProblems fixtures; colmap-sharp 18, 22).
5. Schur: SchurOrdering, SchurEliminator (+ parallel), SchurComplementSolvers,
   ImplicitSchurComplement, IterativeSchurSolver (tests SchurEliminator, SchurEliminatorParallel,
   SchurComplementSolver, SchurDeterminism, ImplicitSchurComplement; colmap-sharp 35).
6. The minimizer: trust region strategy, line search, minimizer, LeastSquaresSolver (tests
   TrustRegionMinimizer, CeresExample, CeresProblem, BundleAdjustmentProblem).
7. Refiners: `estimators/pose`'s `RefineAbsolutePose`, `RefineRelativePose`,
   `RefineEssentialMatrix` (pose_test.cc's six refinement cases) and
   `RefineGeneralizedAbsolutePose` (generalized_pose_test.cc's three), GradientChecker; and the
   shared-focal refiner back on `TinySolverAutoDiffFunction` as COLMAP has it, retiring
   divergence 252.

And the Ceres estimators of Phase 6 (each also takes `PoseGraph`):
- `global_positioning`: `GlobalPositioner` (Solve, SetupProblem, InitializeRandomPositions,
  AddPointToCameraConstraints, AddPoint3DToProblem, AddCamerasAndPointsToParameterGroups,
  ParameterizeVariables, ConvertBackResults), `RunGlobalPositioning`, the
  `cost_functions/motion_averaging.h` costs; `GlobalPositioning.{Nominal, MultiCameraRig,
  RefineSensorFromRigFalsePreservesRig}`.
- `gravity_refinement`: `GravityRefiner::RefineGravity` (with its helpers) and
  `RunGravityRefinement`, over `cost_functions/manifold.h`; `GravityRefinement.{RefineGravity,
  RefineGravityWithNonTrivialRigs}`.
- `view_graph_calibration`: `CalibrateFocalLengths` (`cost_functions/calibration.h`),
  `CalibrateViewGraph` and its helpers, the options' Ceres loss; `CalibrateViewGraph.{Nominal,
  PriorFocalLength, ConfigTagging, RelativePoseReestimation, SphericalCamerasAreIgnored,
  FisheyeCamerasAreIgnored}`.

### Phase 8 — Bundle adjustment
`estimators/bundle_adjustment*` (CPU), `cost_functions/*`, `ceres_loss_function`, `covariance`.
App: run BA on a loaded model with live cost plot and progress/cancel.

### Phase 9 — Features
Remaining:
- The covariant extractor. On main: VLFeat's scale space, smoothing, gradients, the 2x2 SVD and
  Gaussian elimination, the raw descriptor, and `VlCovDet` (detector and shape), bit-exact against
  colmap-sharp's VLFeat oracle (`covariant_sift_vlfeat.json`; one frame value 1 ulp off, entry 1).
  Left: `CovariantSiftCPUFeatureExtractor` (`CovariantSift.cs`: `ComputeRawDescriptors`,
  `PooledMean` with `VlCovDetTests.CSharpOnly_PooledMeanAddsRowsInBlocksOfFour`,
  `CreateDescriptorFilter`, sort and truncation, the minimum-image-side guard; an interleaved
  `vl_imgradient_polar_f` form), its oracle test against `covariant_sift.json`
  (`CovariantSiftOracleTests`), `CreateSiftFeatureExtractor`, and the five `Covariant*` rows of
  `sift_test.cc`'s `SiftCpuExtractionTest.Nominal`.
- `vl_floor_d(-inf)` panics (`as i64` saturates, then subtracts); clamp rather than panic. Only
  `put_image` guards it today.
- The orientation sort's tie rule is not pinned (the oracle has no ties).

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
Remaining from colmap-sharp Phase 13, all waiting on Phase 12's MVS: the WGSL kernels (reuse
as-is where possible), planner, orchestrator, CPU twin, conformance checker, and the three
ColmapGpu PatchMatch classes. They run on the compute seam's dedicated wgpu device
(`WebGpuComputeDevice`, not agg-gui-wgpu's), on Metal/DX12/Vulkan natively and WebGPU in the
browser.

### Phase 14 — Global and hierarchical mapping
`sfm/global_mapper`, `controllers/global_pipeline`, `hierarchical_pipeline`,
`rotation_averaging` controller.

## Skipped tests

Mirror colmap-sharp's "Skipped tests" list as each module is ported (bitmap file I/O,
SQLite files, Ceres internals not ported, CUDA/GPU, SiftGPU, ONNX, vocabulary tree), adapting
reasons to Rust. Every skipped COLMAP test is listed here by name when its file is ported.

### scene/
- `database_test.cc` ParameterizedDatabaseTests.{OpenFile,OpenCloseFile,OpenFileWithNonASCIIPath}:
  they need SQLite database files, which are out of scope (the database is in memory, divergence
  265).

### sensor/
- `bitmap_test.cc` ReadWriteAsRGB, ReadWriteUnicodePath, ReadWriteAsGrey,
  ReadWriteAsGreyNonLinear, ReadWriteLinearColorspace, WriteJpegWithQuality, WriteInvalidFormat,
  ReadNonImageFile, ReadNonExistentFile, ReadUnsupportedChannels, all
  ParameterizedBitmapFormatTests, and the PNG round-trip tails of CloneAsRGB/CloneAsGrey: the host
  decodes and encodes images; the core has no file I/O.

### feature/
- `extractor_test.cc` FeatureExtractionOptions.{Move,MoveAssignment}: they check the LoMa
  options' `shared_ptr` use counts, and LoMa is ONNX (out of scope). The ALIKED_N16ROT,
  ALIKED_N32, LOMA_B and LOMA_B128 rows of CheckAndRequiresOpenGLWithNoGpu: their ONNX
  sub-options are not ported (`rust_only_onnx_extractor_types_fail_check` pins that they fail
  `Check`).
- `matcher_test.cc` the rows of FeatureMatchingOptions.Check for the nine ONNX matcher types:
  ONNX matchers are out of scope (`rust_only_onnx_matcher_types_fail_check_and_create` pins
  that they fail).
- `sift_test.cc` ExtractSiftFeaturesGPU.Nominal, CreateSiftGPUMatcherOpenGL.Nominal,
  CreateSiftGPUMatcherCUDA.Nominal, MatchSiftFeaturesGPU.{TypeMismatch,Nominal},
  MatchGuidedSiftFeaturesGPU.{TypeMismatch,Nominal,EssentialMatrix,Spherical,
  SphericalMixedHemispheres,UnprojectableKeypoints,SharedFocal,SharedFocalPerPairFocal},
  MatchSiftFeaturesCPUvsGPU.Nominal, MatchGuidedSiftFeaturesCPUvsGPUGuided.EssentialMatrix:
  SiftGPU, excluded by its licence.

### solver/ (colmap-sharp's C#-only tests)
- `AutoDiffCostFunctionTests.EveryGradientWidth_GivesIdenticalBits`: it tests colmap-sharp's
  chunked Jet evaluation, which exists only because C# has no const generics; Rust runs one
  `Jet<N>` pass as Ceres does.

### util/
- `types_test.cc` Span.SizeAndEmpty, FilterView.{Empty,All,None,Nominal,RangeExpression}:
  replaced by Rust slices and `Iterator::filter`.
- `string_test.cc` StringPrintf.Nominal (`format!` replaces it);
  ConversionBetweenPlatformAndUTF8.NonASCIIStringRoundtrip (Rust strings are UTF-8); the
  comma-decimal global-locale half of StringToDouble.LocaleIndependence (Rust parsing has no
  locale).
- `signals_test.cc` ScopedSignalHandler.* (5): CLI process signal handling, no wasm equivalent.
- `threading_test.cc` Thread/ThreadPool/JobQueue/ThreadSafeQueue/Barrier cases: replaced by rayon.
- `file_test.cc` filesystem, download and URI cases: the core crate has no filesystem API.
- `eigen_matchers_test.cc`, `opengl_utils_test.cc`: excluded features.

## agg-gui upstream queue
Fix each one in agg-gui, then remove its workaround here:
- `WgpuCustomRenderCtx::parent_clip`'s doc says `[x,y,w,h]` top-down, but it is stored Y-up
  (`[x, y_bottom, w, h]`, `ctx_core::compute_scissor`). Fix the doc or the representation, and add a
  readback test. Workaround: `colmap-app/src/viewport_render.rs::clip_to_top_down`.
- Stock widgets have no id setter, so reflection tests can't find them. Workaround: `Tagged`.
- A `Spacer` in a `FlexRow` reports the row's full height. Workaround: `row_filler()`.
- A wrapper's child doesn't get the parent's final `set_bounds`; wrappers must fill stretch axes
  themselves.
- agg-gui-shell needs a ready-made file-backed `WindowBoundsStore`. Workaround:
  `colmap-native/src/bounds_store.rs`.
- `wgpu::Device` equality compares only the wgpu-core id, and ids restart per `Instance`, so a
  custom renderer can't reliably tell that the device was replaced (device loss / backend switch)
  and would reuse old pipelines. Expose a device generation/identity on `WgpuCustomRenderCtx`.
  Caveat noted on `colmap-app/src/viewport_render.rs::ensure_gpu`.
- agg-gui's CI has no `cargo fmt --check` step; add one.
- `agg-gui-web-shell`: publish it to crates.io when Lars says so. Move agg-gui's own `demo-wasm` (JS-driven loop in `demo/src/app.ts`)
  onto it; migrate AtomArtist, AstroRock, KeyInSight, instant-astronomer (deprecation warnings from
  the old `demo_wgpu::web_shell` wrapper) and Solitaire (its own hand-rolled shell).
- agg-gui-shell has no `after_present` hook (the web shell does); colmap-native marks readiness in
  `after_paint`, just before present.
- Two agg-gui tests fail on clean `main`: `widgets::menu::widget::tests_2::unconsumed_shortcut_fires_top_menu_action`
  and demo-ui `app_builder_tests::each_demo_window_quiesces_after_close`.

## Decisions
- **Web runs single-threaded.** GitHub Pages can't send the COOP/COEP headers that
  SharedArrayBuffer (and so wasm threads / rayon) needs, and we accept that cost: on the web,
  long work is chunked across frames (or moved to a plain Web Worker without shared memory).
  Native will use rayon once the core adds it (see below). A `coi-serviceworker` shim could enable isolation on Pages later; not now.
- **agg-gui is a path dependency** (`../agg-gui`), so colmap-rust can follow agg-gui `main` or a
  development branch; CI checks it out beside this repo. Only the app crates use it.
- **Web shell comes from agg-gui.** A (to-be-published) `agg-gui-web-shell` crate (canvas,
  requestAnimationFrame loop, DOM input, WebGPU surface) lives in the agg-gui repo; colmap-web
  uses it rather than carrying its own copy.
- **Scene ownership: ids, not pointers.** Frames and images store the ids of their rig, camera
  and frame; methods that need the owners take them as arguments (`scene/mod.rs` header,
  divergence 170).
- **Reconstruction mutators return plain `&mut`.** Consistency after an id change is checked by
  `is_valid` and the owner-taking methods, not by the setters (divergence 214).
- **SIFT is CPU-only.** SiftGPU is excluded by its licence; a WGSL SIFT could be a later
  colmap-gpu step.
- **`FeatureExtractor` / `FeatureMatcher` traits follow colmap-sharp**, returning `Result` and
  taking a `CancelToken`.
- **faiss is replaced by an exact nearest-neighbour search** (divergence 231). The descriptor
  index cache is `Option<Arc<dyn FeatureDescriptorIndexCache>>`, which Phase 11's matcher cache
  implements; `ThreadSafeLRUCache<ImageId, dyn FeatureDescriptorIndex>` (`util/cache.rs`) implements it.
- **`ImgFromCamFunc` is `Arc<dyn Fn(Vector3d) -> Option<Vector2d> + Send + Sync>`** (the absolute
  pose estimators, `estimators/solvers/absolute_pose.rs`).
- **`fast_expn` is a constant table** of correctly rounded `exp` values (divergence 202), an
  exception to the `math::fns` rule.
- **The database is one in-memory type** (`scene::database::Database`, divergence 265): no
  path-keyed factory; `Database::new()` stands for COLMAP's in-memory path. Methods take `&self`
  with the tables behind a `Mutex`, so it is shared as `Arc<Database>`; every method returns
  `Result` and fails after `close()`. SQLite's ids (AUTOINCREMENT), UNIQUE indices, foreign keys
  and cascades are reproduced; a write that breaks a constraint stores nothing (266).
  `DatabaseCache` keeps its objects in id order (267) and is returned by value.
- **Optional `Database*` arguments become two functions** (`synthesize_dataset` and
  `synthesize_dataset_with_database`; the same for `synthesize_noise`). `synthesize_images` hands
  each bitmap to a sink (divergence 268): the core has no image encoder.
- **JSON in the core is a small reader written here** (`util/json.rs`; ptree's text-to-value rules
  in `util/ptree_translate.rs`), not serde_json: the core stays
  std-only (plus `libm`), and COLMAP's Boost property_tree is replaced by Rust written here.
  It takes a `&str`; the host reads files.
- **`ReconstructionManager` owns its models by value** (`Vec<Reconstruction>`; `get`/`get_mut`
  borrow, `set` swaps one in), as the rest of the scene owns by id; pipelines borrow a model
  for the steps that change it.
- **The core is single-threaded for now** (no rayon): `num_threads` is validated and otherwise
  ignored, so results never depend on it.
- **colmap-rust will be published to crates.io** once the core is usable (crate `colmap-rust`,
  lib `colmap_rust`); keep its package metadata publish-ready.
