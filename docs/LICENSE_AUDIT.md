# License audit

colmap-rust ships under MIT and has to stay usable in closed-source commercial products
(MatterCAD and anything else built on agg-gui). This file lists which upstream code may be
ported and which may not. Check it before porting any code that is not COLMAP's own, and
before adding a crate dependency.

**Pure Rust, always.** No C/C++, no `-sys` crates, no build scripts that compile native code,
no linking against COLMAP or anything else. Every COLMAP dependency is ported to Rust,
replaced by Rust written here, or dropped. The core crate must build for
`wasm32-unknown-unknown`.

**The license rule:** port or depend on only code under a permissive license (MIT, BSD,
Apache-2.0, zlib, Boost, public domain). Anything copyleft (GPL, LGPL, AGPL, or MPL file-level
copyleft) or "non-commercial" is excluded. Excluded code may not be read as a guide for a
line-by-line transcription either. Where COLMAP depends on excluded code, we write a
replacement from the published algorithm or port a permissively licensed equivalent, and cite
the source in the file header.

**colmap-sharp** (github.com/larsbrubaker/colmap-sharp, MIT, same author) already made every
decision below and wrote the replacements. Its own code (the linear algebra, the sparse
Cholesky, the Delaunay tetrahedralization, the PatchMatch random generator, the WGSL kernels)
may be ported to Rust freely. Code *it* ported from an upstream (Ceres, PoseLib, VLFeat,
PoissonRecon, libc++) carries that upstream's notice here too. The decisions table below is
colmap-sharp's; file paths in it name colmap-sharp files, and the Rust port keeps the same
module layout (`ColmapSharp/Mathematics/LibcxxRandom.cs` → `colmap-rust/src/math/libcxx_random.rs`).

## COLMAP's own source (BSD-3-Clause): port freely

All of `src/colmap/**` is BSD-3. Keep the notice in `THIRD_PARTY_NOTICES.md`.
The modules that exist only because of an excluded dependency are listed below.

## COLMAP's dependencies

| Upstream | License | Used by COLMAP for | Decision |
|---|---|---|---|
| Eigen | MPL-2.0 | All linear algebra | **Do not port.** Write our own linear algebra (`ColmapSharp.LinearAlgebra`) from textbook algorithms (Golub & Van Loan). Match Eigen's *documented* semantics (e.g. quaternion `w,x,y,z` layout, `AngleAxis` conventions), not its code. Dense decompositions cite their textbook source in each file header; where a sign or pivoting convention must match, it is taken from LAPACK's published documentation (BSD-3; no LAPACK code is ported) and pinned against numpy. |
| libc++ (LLVM) | Apache-2.0 WITH LLVM-exception | `std::uniform_*_distribution`, `normal_distribution` behind `math/random`; `std::priority_queue` in `mvs/mesh_simplification`; `std::unordered_map` iteration in PoissonRecon's `SparseMatrix` product | **Ported** (`ColmapSharp/Mathematics/LibcxxRandom.cs`: `uniform_int_distribution` with `__independent_bits_engine`, `generate_canonical`, `uniform_real_distribution`, `normal_distribution`, `std::shuffle`) so seeded RANSAC matches the macOS pycolmap oracle; and (`ColmapSharp/Mvs/CollapseHeap.cs`) the heap algorithms `make_heap`/`push_heap`/`pop_heap` (`__sift_down`, `__floyd_sift_down`, `__sift_up`), so tied collapse costs pop in the wheel's order; and (`ColmapSharp/Util/LibcxxUnorderedMap.cs`) `std::unordered_map<int, T>`'s hash table (`__constrain_hash`, the unique-key insert, `__rehash`/`__do_rehash`, and `__next_prime` reimplemented from its contract), so PoissonRecon's sparse matrix products list row entries in libc++'s iteration order. LLVM's notice is in `THIRD_PARTY_NOTICES.md`. |
| Ceres Solver | BSD-3 | Bundle adjustment, all nonlinear refinement | **Port the subset we use:** Levenberg-Marquardt trust region, Schur complement (dense and iterative), Jets for automatic differentiation, loss functions, manifolds. Add Ceres' notice when the first file lands. |
| SuiteSparse / CHOLMOD | LGPL-2.1+ / GPL | `optim/sparse_cholesky`, LAD, rotation averaging | **Do not port.** Write a supernodal- or simplicial-LDLᵀ sparse Cholesky with AMD ordering from the published algorithms (Davis, *Direct Methods for Sparse Linear Systems*, as a description only; CSparse's code is LGPL). Done as `LinearAlgebra/SimplicialCholesky.cs` (simplicial up-looking LLᵀ/LDLᵀ, Liu's elimination tree) and `LinearAlgebra/AmdOrdering.cs` (Amestoy, Davis and Duff, SIAM J. Matrix Anal. Appl. 1996); no SuiteSparse code was read. |
| METIS | Apache-2.0 | Graph partitioning for hierarchical mapping (`ComputeNormalizedMinGraphCut`) | Allowed, but not ported. Replaced by `Mathematics/MultilevelPartitioner.cs`, written here from the published multilevel/FM papers without reading METIS's code (docs/CPP_DIVERGENCES.md entry 77), so no METIS notice is needed. |
| PoseLib | BSD-3 | Minimal solvers (P3P, 5-pt, generalized pose, focal solvers, homography, essential) | **Port freely.** Ported at commit `fa7280fee27f97aff31ae7f98bab7f583fac7d08` (COLMAP 4.2.0's FetchContent pin) into `ColmapSharp/Estimators/Solvers/PoseLib/`, one C# file per PoseLib source; notice in `THIRD_PARTY_NOTICES.md`. |
| VLFeat (`thirdparty/VLFeat`) | BSD-2 | CPU SIFT extraction | **Port freely.** SIFT's patent (US 6,711,293) expired in March 2020. SIFT filter (sift.c) and the DoG covariant detector (covdet.c, scalespace.c), with the mathop/imopv pieces they use, ported into `ColmapSharp/Feature/VLFeat/`; notice in `THIRD_PARTY_NOTICES.md`. |
| SiftGPU (`thirdparty/SiftGPU`) | UNC, "educational, research and non-profit purposes" only | GPU SIFT | **Excluded.** CPU SIFT (VLFeat) is the only extractor. |
| LSD (`thirdparty/LSD`) | **AGPL-3.0** | Line segment detection (`image/line`, `estimators/coordinate_frame`) | **Excluded.** Coordinate-frame estimation from lines is skipped, or reimplemented from the published LSD paper (von Gioi et al., IPOL 2012) if it is ever needed. |
| CGAL | **GPL-3.0** / commercial | Delaunay meshing, advancing-front meshing, texture mapping | **Excluded.** No CGAL source is read. The 3D Delaunay tetrahedralization is written here (`Geometry/Delaunay/DelaunayTriangulation3*.cs`) from the published algorithms: Bowyer-Watson insertion (1981), visibility/straight walks (Devillers, Pion and Teillaud 2002), BRIO (Amenta, Choi and Rote 2003) with a Hilbert sort (Hamilton 2006, "Compact Hilbert Indices"), and Shewchuk's predicates (below). Only CGAL's *documented* conventions (infinite vertex, cell/facet/neighbor indexing, orientation signs) are matched. Graph cut, visibility scoring and surface extraction are COLMAP's own (BSD) and are ported. Texture mapping's occlusion AABB tree is replaced by `Mvs/TriangleBvh.cs`, written from the published SAH / slab / Möller–Trumbore papers (CPP_DIVERGENCES 90). |
| Shewchuk's robust predicates (predicates.c and the 1997 DCG paper) | Public domain | Exact orient/in-sphere signs for the Delaunay tetrahedralization (replacing CGAL's filtered kernel) | **Allowed.** `Geometry/Delaunay/RobustPredicates.cs` uses the paper's floating-point filter (the errboundA bounds) and an exact BigInteger fallback written here instead of the adaptive expansion stages. Public domain, so no notice is required; the source is cited in the file header and in `THIRD_PARTY_NOTICES.md`. |
| PoissonRecon (`thirdparty/PoissonRecon`) | MIT (folder `LICENSE`); files carry Johns Hopkins BSD-style headers | Poisson surface reconstruction | **Port freely.** Being ported into `ColmapSharp/Mvs/PoissonRecon/`: the polynomial and B-spline machinery generically for degrees 0-2 with all three boundary types (what COLMAP instantiates), the octree and later stages specialized to 3D and `float` as `poisson_meshing.cc` runs them; both notices are in `THIRD_PARTY_NOTICES.md`. COLMAP builds it with `-ffast-math`; the port is strict IEEE (docs/CPP_DIVERGENCES.md, entry 74). |
| faiss | MIT | Nearest-neighbor descriptor matching, retrieval | Not used (native). Replaced by an exact managed k-NN search (`Feature/FeatureDescriptorIndex.cs`, docs/CPP_DIVERGENCES.md entry 42). |
| Symforce-Caspar | Apache-2.0 | GPU bundle adjustment | Not needed (GPU only). |
| OpenImageIO | Apache-2.0 | Image decoding, EXIF | Not used (native). The library takes decoded pixel buffers; the host (MatterCAD/agg-sharp) decodes. EXIF focal length is read by a small managed parser written here. |
| SQLite | Public domain | Feature/match database | Not used (native). The database is an in-memory C# store with the same API; persistence, if needed, is a managed format of our own. |
| Boost, gflags, glog | BSL / BSD | Utilities, CLI, logging, hash containers | Not ported; BCL replacements. One exception: Boost.Unordered's `mulx` hash mixer (BSL-1.0, permissive) is ported into `Scene/SceneClustering.cs` to model the iteration order of COLMAP's Boost hash map backend (docs/CPP_DIVERGENCES.md entry 120); notice in `THIRD_PARTY_NOTICES.md`. |
| googletest | BSD-3 | COLMAP's test framework | Not a dependency of the library. The test project ports `FloatingPoint<double>::AlmostEquals` (EXPECT_DOUBLE_EQ) into `ColmapSharp.Tests/GTestDouble.cs`; notice in `THIRD_PARTY_NOTICES.md`. |
| Qt | LGPL / commercial | GUI | **Excluded.** No GUI in this library. |
| ONNX Runtime + models (ALIKED, LightGlue, LoMa, AnyCalib) | MIT runtime; model weights vary | Learned features | Out of scope. Check each model's weights license separately if this is ever revisited. |
| CUDA / HIP | Proprietary toolchains | PatchMatch stereo, GPU SIFT | Not used. PatchMatch stereo is ported to managed CPU code from `mvs/patch_match_cuda.cu` (COLMAP's own BSD code). cuRAND (per-pixel random numbers) is replaced by `Mvs/PatchMatchRandom.cs`, a counter-based generator written here from SplitMix64's published output function (Steele, Lea and Flood, OOPSLA 2014; public-domain constants); no cuRAND code was read. CUDA texture sampling is reimplemented from the CUDA Programming Guide's documented filtering rules (`Mvs/PatchMatchTextures.cs`). |

## Crate dependencies

`colmap-rust` (the core library) may depend only on crates listed here. Adding one needs a row,
a permissive license, pure Rust (it must still build for wasm32), and a pinned major version.

| Crate | License | Why |
|---|---|---|
| `libm` 0.2 | MIT (a port of musl's libm, MIT) | Every transcendental in `math::fns` (f64 and f32). Pure Rust, no dependencies; its `build.rs` only emits cfg flags (no native code). Chosen because it gives identical bits on every target, native and wasm32, where std's platform libm does not (docs/CPP_DIVERGENCES.md, entry 1). |

`rayon` (MIT/Apache-2.0) is the expected next addition, behind a `parallel` feature, for
determinism-preserving parallel loops.

`colmap-gpu` may additionally use `wgpu` (MIT/Apache-2.0). The app crates may use agg-gui and
its stack (MIT), `wasm-bindgen`/`web-sys`/`js-sys` (MIT/Apache-2.0), and permissive image
decoders (`image`, `png`, `jpeg-decoder`/`zune-jpeg`; MIT/Apache-2.0/zlib).

App crates in use today (`colmap-app`, `colmap-app-test`, `colmap-native`):

| Crate / asset | License | Why |
|---|---|---|
| `agg-gui` 0.5, `agg-gui-wgpu` 0.5.2, `agg-gui-shell` 0.5.1 | MIT | The GUI, its wgpu renderer (custom-render hook for the 3D viewport) and the native winit shell. |
| `wgpu` 29 | MIT/Apache-2.0 | Viewport line pipeline; native backends (Metal/Vulkan/DX12) enabled in `colmap-native` only. |
| `glam` 0.30 | MIT/Apache-2.0 | Orbit-camera math in `colmap-app` only. The core library never uses it; it gets its own `linalg`. |
| `bytemuck` 1 | MIT/Apache-2.0/Zlib | Casting viewport vertex / uniform data to bytes for wgpu. |
| Noto Sans Regular (font, `colmap-app/assets/fonts/`) | SIL OFL 1.1 | UI text. Bundled font file, not code; OFL allows embedding in commercial software. |
| Font Awesome 4.7 (font, `colmap-app/assets/fonts/`) | SIL OFL 1.1 (font) | Icons, drawn as code points through the UI font's fallback chain. |
