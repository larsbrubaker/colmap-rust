# Deliberate divergences from COLMAP

Each entry: what differs, why, and the evidence. Numbered so code comments can cite them
(`docs/CPP_DIVERGENCES.md`, entry N). Remove an entry when the divergence is gone.

Numbers are stable: an entry keeps its number for good, and a removed entry's number is not
reused, so gaps in the sequence are intentional. Never renumber, because code comments and open
branches cite these numbers.

colmap-sharp's `docs/CPP_DIVERGENCES.md` documents ~136 divergences it took. Most will recur here
for the same reason (a replaced dependency, a deterministic tie-break). When the Rust port takes
the same divergence, write a full entry here (don't just point there) and add
"Same as colmap-sharp entry N." so the two can be cross-checked.

## 140. PROSAC's out-of-range sample index fails a check instead of reading past the data

**What differs.** COLMAP's `ProgressiveSampler` (ported faithfully in
`optim/progressive_sampler.rs`) makes index `n` the mandatory element of a progressive sample,
and `n` can equal `total_num_samples`: on the first sample when `num_samples ==
total_num_samples`, and in general on the sample where the growth schedule reaches the last
element. COLMAP's `Sampler::SampleXY` then reads `X[total_num_samples]`, past the end of the
`std::vector` (undefined behavior: garbage or a crash). `Sampler::sample_x` / `sample_xy` in
`optim/sampler.rs` check every sampled index against the data length and return COLMAP's
"Check failed" error instead.

**Why.** Undefined behavior has no Rust equivalent to match (an out-of-bounds index panics),
and silently clamping or shifting the index would change the sampler's Tier A sequence that
`progressive_sampler_test.cc` pins. COLMAP 4.2.0 instantiates `ProgressiveSampler` nowhere
outside its own test, so no pipeline reaches this path. Same as colmap-sharp entry 16.

**Evidence.** `tests/optim/progressive_sampler.rs` (1:1) and the seeded sequence in
`tests/optim/rust_only_sampler_sequence.rs` pass unchanged;
`tests/optim/rust_only_ransac.rs`, `rust_only_progressive_sampler_index_past_end_fails_check`
runs RANSAC with PROSAC on exactly `MIN_NUM_SAMPLES` pairs and expects the check failure.

## 141. RANSAC and LO-RANSAC always run their trial loop serially

**What differs.** With `RANSACOptions::num_threads > 1` (or -1), COLMAP built with OpenMP runs
the trial loop on several threads, each with its own sampler seeded `random_seed + thread
index`, sharing an atomic trial counter and a mutex-guarded best model. `optim/ransac.rs` and
`optim/loransac.rs` validate `num_threads` exactly as COLMAP does (`Check()`, and "Parallel
RANSAC only supports RandomSampler" for any other sampler with more than one effective thread)
and then run the loop once on the calling thread with the thread-0 seed, which is what COLMAP
itself does when built without OpenMP ("the block runs once serially").

**Why.** COLMAP's parallel result depends on thread scheduling (which thread claims which trial
index, and which thread's model reaches the shared best first), so it is not reproducible even
against itself; CLAUDE.md requires sequential and parallel runs to give the same result. The
serial loop is COLMAP's own `num_threads == 1` behavior. A deterministic parallel scheme (for
example fixed per-trial seeds) would be a different algorithm from both COLMAP builds and is
left for when profiling shows RANSAC is a bottleneck. Same as colmap-sharp entry 17.

**Evidence.** `tests/optim/rust_only_ransac.rs`: `rust_only_ransac_parallel_line_fit` and
`rust_only_loransac_parallel_line_fit` (`num_threads = 4`) pass the same checks as the serial
runs, and `rust_only_parallel_requires_random_sampler` pins the kept validation. The 1:1
`ParallelSimilarityTransform` cases join them with Phase 6's `SimilarityTransformEstimator`.

## 160. Track element access out of range panics instead of throwing `std::out_of_range`

**What differs.** COLMAP's `Track::Element(idx)` and `SetElement(idx, ..)` go through
`std::vector::at`, which throws `std::out_of_range` for a bad index. `Track::element`,
`element_mut` and `set_element` (`scene/track.rs`) index the `Vec` and panic.
`DeleteElement(idx)`, a `THROW_CHECK_LT` in COLMAP, stays an `Err`.

**Why.** No COLMAP code catches that exception: callers iterate `0..Length()`, so an
out-of-range index is a bug in the caller, an internal invariant where CLAUDE.md's error rule
allows a panic. A `Result` on every element access would burden the per-observation loops of
triangulation and bundle adjustment for a condition that cannot occur in correct code.

**Evidence.** `tests/scene/rust_only_scene_types.rs`:
`rust_only_track_element_out_of_range_panics` and
`rust_only_track_delete_element_out_of_range_fails_check`.

## 161. VisibilityPyramid defines its two undefined-behavior cases

**What differs.** Two inputs are undefined behavior in COLMAP's `VisibilityPyramid` and
defined in `scene/visibility_pyramid.rs`:
- `max_score_ += dim * dim * dim * dim` is `int` arithmetic and overflows from 8 levels on
  (dim 256, 2^32); here it is `usize`, so `max_score` is the sum the formula means.
- `CellForPoint` converts `max_dim * x / width` to `size_t` before clamping; for a negative or
  NaN coordinate that conversion is undefined. Rust's `as usize` saturates both to 0, so the
  point lands in the first cell (and a coordinate beyond the range in the last, as in COLMAP).

**Why.** Undefined behavior has no result to match; these are the values a caller would
expect. COLMAP's only user, the incremental mapper's observation manager, builds pyramids
with `kNumPoint3DVisibilityPyramidLevels = 6` levels and the image's size, where both versions
agree.

**Evidence.** `tests/scene/visibility_pyramid.rs` (1:1, 1 to 7 levels) passes unchanged;
`tests/scene/rust_only_scene_types.rs`: `rust_only_visibility_pyramid_eight_levels_max_score`
and `rust_only_visibility_pyramid_out_of_range_points_clamp`.

## 170. Frames and images store ids, not pointers to their rig, camera and frame

**What differs.** COLMAP's `Frame` holds a raw `Rig* rig_ptr_`, and its `Image` holds
`Camera* camera_ptr_` and `Frame* frame_ptr_`; the reconstruction sets them (`SetRigPtr`,
`SetCameraPtr`, `SetFramePtr`) and clears them (`Reset*Ptr`), and methods such as
`Frame::SensorFromWorld`, `Image::CamFromWorld` and `Image::ProjectPoint` follow them.
`scene/frame.rs` and `scene/image.rs` store only the ids; the owner of the maps passes the
objects to the methods that need them, and each method checks it was handed the right one:
- `Frame::sensor_from_world(&rig, ..)`, `set_cam_from_world(&rig, ..)`;
  `Image::cam_from_world(&frame, &rig)`, `projection_center`, `viewing_direction`,
  `is_ref_in_frame` (same parameters), `project_point(&frame, &rig, &camera, ..)`. Each fails
  unless the frame / camera has the image's id and the rig has the frame's rig id, where
  COLMAP fails on a null pointer.
- `SetRigPtr`, `SetCameraPtr`, `SetFramePtr` become `Frame::verify_rig`,
  `Image::verify_camera`, `Image::verify_frame`: the same checks, with nothing stored.
  Rebinding to another rig / camera / frame (COLMAP's `Set*Ptr` on an already bound object,
  which adopts the new id) is setting the id, then verifying.
- `AddDataId` on a frame with a rig (which checks the rig has the sensor) is
  `Frame::add_data_id_in_rig(&rig, ..)`; `add_data_id` is the unbound form.
- `Image::HasPose` takes the image's frame as an `Option` (`frames.get(&image.frame_id())`);
  `operator<<` is `Image::display(frame)`. `Image`'s `PartialEq` compares the image's own data;
  `Image::eq_with_frames` is COLMAP's `operator==`, which also compares the frames' poses.
- `RigPtr`, `CameraPtr`, `FramePtr`, `Has*Ptr` and `Reset*Ptr` have no counterpart, and
  `Frame::set_rig_id` / `Image::set_camera_id` / `Image::set_frame_id` drop COLMAP's
  `THROW_CHECK(!Has*Ptr())`: there is no binding to protect, and a frame or image inside a
  reconstruction is only reached through it.

**Why.** CLAUDE.md's translation rule: pointer graphs become ids into owning maps. Shared
mutable pointers between the reconstruction's maps cannot be expressed in safe Rust without
`Rc<RefCell<..>>` everywhere, which would make every pose read in bundle adjustment and the
mapper a runtime borrow check. Passing the owner keeps frames and images plain data that
clone, move between threads and serialize without fix-ups (COLMAP's reconstruction re-links
every pointer after a copy; with ids there is nothing to re-link).

**Evidence.** `tests/scene/frame.rs` and `tests/scene/image.rs` port `frame_test.cc` and
`image_test.cc` 1:1, with the pointer tests (`SetResetRigPtr`, `CameraPtr`, `FramePtr`,
`ConstructCopy`, `AssignCopy`) as behaviour tests of the id-based API under the C++ names;
`tests/scene/rust_only_frame_image.rs` checks that every owner-taking method rejects the
wrong rig, frame or camera, and the two equality forms.

## 171. Image point access out of range panics instead of throwing `std::out_of_range`

**What differs.** COLMAP's `Image::Point2D(idx)`, `SetPoint3DForPoint2D` and
`ResetPoint3DForPoint2D` go through `std::vector::at`, which throws `std::out_of_range` for a
bad index. `Image::point2d`, `point2d_mut`, `set_point3d_for_point2d` and
`reset_point3d_for_point2d` (`scene/image.rs`) index the `Vec` and panic.
`SetPoint3DForPoint2D`'s `THROW_CHECK_NE(point3D_id, kInvalidPoint3DId)` stays an `Err`.

**Why.** The same reason as entry 160 for `Track`: no COLMAP code catches the exception;
callers index with feature indices taken from the image itself or from its correspondences,
so an out-of-range index is a caller bug, an internal invariant where CLAUDE.md's error rule
allows a panic, and a `Result` on every point access would burden triangulation and bundle
adjustment.

**Evidence.** `tests/scene/rust_only_frame_image.rs`:
`rust_only_image_point2d_out_of_range_panics` and
`rust_only_image_set_point3d_out_of_range_panics`.

## 180. SIFT descriptors converted from float saturate to 0..255

**What differs.** `FeatureDescriptors::FromFloat` casts each SIFT value with
`static_cast<uint8_t>`, which is undefined behavior for a value outside `[0, 256)` or NaN.
`FeatureDescriptors::from_float` (`feature/types.rs`) uses Rust's `as u8`: it truncates toward
zero like `static_cast` for in-range values, clamps out-of-range values to 0 or 255, and maps
NaN to 0.

**Why.** Undefined behavior has no result to match. Saturation is the value a caller would
expect from a byte descriptor, and SIFT's own descriptors are already quantized to 0..255
before they reach this conversion, so in-range inputs (every input COLMAP produces) give
COLMAP's bytes.

**Evidence.** `tests/feature/types.rs` `feature_descriptors_sift_conversion` (1:1) passes;
`tests/feature/rust_only_feature_types.rs`
`rust_only_feature_descriptors_sift_cast_saturates` pins `[12.9, 255, 300, -4, NaN, 0.5]` →
`[12, 255, 255, 0, 0, 0]`.

## 181. The keypoint and match matrix conversions check their column count at run time

**What differs.** COLMAP's `KeypointsFromMatrix` and `MatchesFromMatrix` take
`Eigen::Ref<const Matrix<.., Dynamic, 4 or 2, RowMajor>>`, so the column count is part of the
type: a fixed-size source with the wrong count does not compile, and a dynamic one trips an
Eigen assertion (undefined behavior in a release build). `keypoints_from_matrix` and
`matches_from_matrix` (`feature/types.rs`) take a dynamic `RowMajorMatrix` and return the
`THROW_CHECK_EQ`-style `Err` for any other column count. `KeypointsFromMatrix` also returns the
keypoint constructor's `Err` for a negative scale, where COLMAP throws.

**Why.** The Rust matrix type has no compile-time column count; a checked error is the closest
equivalent and never reads past a row. Same as colmap-sharp, which checks the column count
the same way (`FeatureKeypoints.KeypointsFromMatrix` / `MatchesFromMatrix`).

**Evidence.** `tests/feature/rust_only_feature_types.rs`
`rust_only_feature_matrix_conversions_check_columns`.

## 200. ExtractTopScaleFeatures keeps equal-scale keypoints in input order

**What differs.** COLMAP's `ExtractTopScaleFeatures` (feature/utils.cc) selects the largest
scales with `std::partial_sort`, which leaves keypoints of equal scale in an unspecified,
implementation-defined order (libc++'s heap-based selection). `extract_top_scale_features`
(`feature/utils.rs`) sorts by scale, largest first, with the input index as the tie-break
(`f32::total_cmp`, so a NaN scale also has a fixed place, where COLMAP's `>` comparator is not
a strict weak order).

**Why.** Equal scales are common: the SIFT extractor emits one keypoint per orientation, and
all of them share the scale. Reproducing libc++'s heap order would tie the result to one
standard library; the index tie-break is deterministic and stable. Only the order among equal
scales differs, and *which* keypoints are kept differs only when a tie straddles the cut,
where COLMAP's choice is arbitrary too. Same as colmap-sharp entry 40.

**Evidence.** `tests/feature/utils.rs` `extract_top_scale_features_nominal` (1:1, distinct
scales) passes unchanged; `tests/feature/rust_only_feature_utils.rs`
`rust_only_extract_top_scale_features_ties_keep_input_order`.

## 201. SIFT never reuses a previous image's gradient

**What differs.** VLFeat caches the polar gradient of the current octave and records which
octave it holds in `grad_o`, which only `vl_sift_new` resets. A filter reused for a second
image of the same size keeps the first image's `grad_o`: when the first image's last octave
with keypoints is the second image's first octave, VLFeat's orientations and descriptors of the
second image's first octave are computed from the *first* image's gradient. COLMAP's
`SiftCPUFeatureExtractor` reuses its filter across images of equal size, so its results can
depend on the order images are processed. `VlSiftFilter::process_first_octave`
(`feature/vlfeat/sift_filter.rs`) resets `grad_o`, so every image gets its own gradient.

**Why.** It is a bug in VLFeat's reuse path: a feature's descriptor should depend only on its
own image. A fresh filter (what VLFeat computes for the first image) is the correct result.
Same as colmap-sharp entry 43 (the gradient part; its cancellation part is a Rust
`CancelToken` here and changes no numbers).

**Evidence.** `tests/feature/rust_only_vl_sift_filter.rs`
`rust_only_vl_sift_filter_reused_filter_matches_fresh_one` (fails with the reset removed).

## 202. SIFT's `fast_expn` table is a constant of correctly rounded values

**What differs.** VLFeat fills its 257-entry `exp(-x)` interpolation table (`sift.c`'s
`fast_expn_init`) by calling the platform `exp`. colmap-rust's rule (entry 1) routes every
transcendental through the `libm` crate, whose `exp` is 1 ulp from the correctly rounded value
for some of these 257 arguments. `feature/vlfeat/mathop.rs` instead holds the table as a
constant: each entry the correctly rounded double of `exp(-k * 25 / 256)`, written as bits.

**Why.** The table weights every orientation-histogram vote and descriptor sample, and a 1 ulp
difference in it moved SIFT orientations by an ulp against the VLFeat fixture. Apple libm,
which the pycolmap wheel and the fixture run, returns exactly the correctly rounded values for
all 257 arguments (checked against Python's `decimal` at 80 digits), so the constant table gives
VLFeat-on-macOS's bits on every target, native and wasm, which is what entry 1 aims for.

**Evidence.** `tests/feature/rust_only_vl_sift_filter.rs`: the six
`rust_only_vl_sift_filter_matches_unfused_vlfeat_*` tests are bit-exact with the table (with
`libm`'s `exp`, four of them fail on an orientation's last bit);
`rust_only_vl_sift_expn_table_is_exp` checks every entry within 1 ulp of `fns::exp`.

## 210. CorrespondenceGraph lists image pairs in insertion order

**What differs.** `CorrespondenceGraph::image_pairs` and `num_matches_between_all_images`
(`scene/correspondence_graph.rs`) enumerate the image pairs in the order
`add_two_view_geometry` added them. COLMAP iterates its `FlatHashMap<image_pair_t, ImagePair>`,
whose order is the hash table's: `std::unordered_map` or `boost::unordered_flat_map`, whichever
`COLMAP_HASH_MAP_BACKEND` the build picked, so it is not even the same across COLMAP builds.
`NumMatchesBetweenAllImages` returns a `NodeHashMap`; here it is a `Vec<(pair id, number of
matches)>`, since every COLMAP caller (`ObservationManager`, `PoseGraph::Load`,
`SceneClustering`, `DatabaseCache`) only iterates it.

**Why.** CLAUDE.md requires deterministic iteration, and reproducing either backend's bucket
layout would mean porting it; the order is not part of COLMAP's contract
(`correspondence_graph_test.cc` compares `ImagePairs()` with `UnorderedElementsAre` and reads
`NumMatchesBetweenAllImages()` by key). Insertion order rather than ascending pair id keeps
colmap-rust and colmap-sharp walking the pairs in the same order, so pipelines that iterate
them (the pose graph and the global mapper) agree between the two ports. Every per-point
correspondence list, `extract_matches_between_images` and `extract_transitive_correspondences`
are in COLMAP's order already (they come from vectors, not hash iteration).
Same as colmap-sharp entry 14.

**Evidence.** `tests/scene/correspondence_graph.rs` (`correspondence_graph_test.cc` 1:1)
passes; `tests/scene/rust_only_correspondence_graph.rs`
`rust_only_correspondence_graph_pairs_in_insertion_order` pins the order.

## 211. CorrespondenceGraph does not log dropped matches

**What differs.** COLMAP's `AddTwoViewGeometry` emits a `LOG(WARNING)` line for a self-match
(`image_id1 == image_id2`), for each match whose point index is out of bounds and for each
duplicate match. `CorrespondenceGraph::add_two_view_geometry` drops and counts them exactly as
COLMAP does, without the log lines.

**Why.** The core crate has no logging facility yet (COLMAP's glog is not ported); the lines are
diagnostic only, and the counts, which are the function's contract, are unchanged. When a
logger lands, this entry goes away (as the same note on
`geometry::pose_prior::gravity_from_exif_orientation` does).

**Evidence.** `tests/scene/correspondence_graph.rs` `correspondence_graph_out_of_bounds` and
`correspondence_graph_duplicate` (1:1) check the counts; `tests/scene/rust_only_correspondence_graph.rs`
`rust_only_correspondence_graph_self_match_is_ignored`.

## 212. A rejected AddTwoViewGeometry leaves the correspondence graph unchanged

**What differs.** COLMAP's `AddTwoViewGeometry` adds the match count to both images'
`num_correspondences` before its `THROW_CHECK(inserted)` for a pair that was already added
(and before `ImagePairToPairId` rejects an id beyond `kMaxNumImages`), so the throw leaves
both totals inflated. `CorrespondenceGraph::add_two_view_geometry` checks the
pair (and that both images exist) first, so an `Err` leaves the graph as it was.

**Why.** A failed call that half-applies is a bug: after it, the counts the incremental mapper
uses to pick its initial pair no longer match the stored correspondences. No caller relies on
the inflated totals.

**Evidence.** `tests/scene/rust_only_correspondence_graph.rs`
`rust_only_correspondence_graph_rejected_duplicate_pair_leaves_graph_unchanged` and
`rust_only_correspondence_graph_missing_image_is_out_of_range`.

## 213. Reconstruction iterates its objects in ascending id order

**What differs.** COLMAP's `Reconstruction` keeps rigs, cameras, frames, images and 3D points
in `NodeHashMap`s, whose iteration order is unspecified (it depends on the hash-map backend and
its history). `scene/reconstruction.rs` keeps them in `BTreeMap`s, which iterate in ascending
id order. Where that order reaches an output, the result can differ from a given COLMAP build:
- `find_image_with_name` returns the smallest-id image when several share a name, and so
  `find_common_reg_image_ids` pairs with that one.
- `rigs()` / `cameras()` / `frames()` / `images()` / `points3d()` and `point3d_ids()` enumerate
  in ascending id order for every caller; `compute_mean_reprojection_error` sums in that order.

`reg_frame_ids()` and `reg_image_ids()` are COLMAP's vector, in registration order, unchanged.

**Why.** CLAUDE.md requires deterministic iteration. COLMAP relies on no particular order (its
tests only use order-free properties), so ascending id is the natural deterministic choice and
needs no extra state. Same as colmap-sharp entry 21.

**Evidence.** `tests/scene/reconstruction.rs` (`reconstruction_test.cc` 1:1 for the ported
methods) passes; `tests/scene/rust_only_reconstruction.rs`
`rust_only_reconstruction_iterates_in_ascending_id_order` pins the order.

## 214. Changing an attached image's or frame's ids is caught by IsValid, not by the setter

**What differs.** In COLMAP, `Image::SetCameraId` and `Frame::SetRigId` throw once the object is
bound to a camera / rig pointer (`THROW_CHECK(!HasCameraPtr())`), so the image or frame inside a
reconstruction cannot be moved to another camera or rig by accident; `SetFrameId` has no such
guard. `Reconstruction::image_mut` / `frame_mut` hand out `&mut Image` / `&mut Frame`, and the
setters (entry 170) do not know the object is in a reconstruction, so such a change is not
rejected when made. `Reconstruction::is_valid` reports it (the camera, rig or frame must exist
and hold the object), and every owner-taking method of `Frame` / `Image` rejects mismatched
objects, so a broken association never yields a wrong pose silently.

**Why.** With ids into owning maps there is no pointer to guard; a guarded mutable view (a
wrapper type forwarding every safe setter) would double the `Image` / `Frame` API for a
mistake no COLMAP code makes (COLMAP's own code changes these ids only on objects outside a
reconstruction, or in `SetRigsAndFrames`, which re-verifies them). Flagged for review: if a
later port needs the guard, `image_mut` becomes that view.

**Evidence.** `tests/scene/rust_only_reconstruction.rs`
`rust_only_reconstruction_changed_ids_are_invalid`; `tests/scene/reconstruction.rs`
`reconstruction_is_valid` (the "empty frame pointer" case clears the image's frame id).

## 215. DeleteAllPoints2DAndPoints3D resets each image's triangulated count

**What differs.** COLMAP's `DeleteAllPoints2DAndPoints3D` clears the 3D points and calls
`image.SetPoints2D(std::vector<Eigen::Vector2d>(0))`, which resizes the image points but does not
touch `num_points3D_`, so every image that had triangulated points keeps a stale count: its
`NumPoints3D()` is still non-zero with no points left, and `IsValid()` is false.
`Reconstruction::delete_all_points2d_and_points3d` leaves each image's count at 0.

**Why.** A count of triangulated points on an image without points is a bug; it makes the
reconstruction fail its own validity check and inflates `ComputeNumObservations`. Nothing
relies on the stale value.

**Evidence.** `tests/scene/rust_only_reconstruction.rs`
`rust_only_reconstruction_delete_all_points2d_and_points3d_resets_counts` (COLMAP would leave
`num_points3d` at 1 and `is_valid` false).

## 216. Rejected Reconstruction operations leave the reconstruction unchanged

**What differs.** Several COLMAP `Reconstruction` operations throw part-way, after they changed
something:
- `AddPoint3D(point3D_id, point3D)` raises `max_point3D_id_` and marks the track's image points
  one by one before it fails on a missing image, an image point observing another 3D point, or
  an id already in use.
- `AddObservation` marks the image point before it looks up the 3D point.
- `DeletePoint3D` resets image points one by one before it fails on a missing track image.

Here each checks everything it can fail on first, so an `Err` leaves the reconstruction as it
was (including the next id `add_point3d` hands out).

**Why.** A failed call that half-applies is a bug: afterwards image points claim a 3D point
that does not exist (or no longer claim one that does), and `IsValid()` is false. No caller
relies on the partial state.

**Evidence.** `tests/scene/rust_only_reconstruction.rs`
`rust_only_reconstruction_failed_operations_leave_it_unchanged`.

## 220. re3q3's random change of variables uses a fixed-seed mt19937, not std::rand

**What differs.** PoseLib's `re3q3` (P4Pf's three-quadratics solver,
`estimators/solvers/poselib/re3q3.rs`, with the retry in `re3q3_var_change.rs`) retries a
near-degenerate system (all three elimination determinants below 1e-10) after a random affine
change of variables. PoseLib draws that from Eigen's `Quaternion::UnitRandom()` and
`setRandom()`, which read the C library's global `std::rand()`. The port draws a uniform
rotation (Shoemake, Graphics Gems III) and a shift in [-1, 1]^3 from a fresh `Mt19937` seeded
with 0 on every call, through libc++'s `uniform_real_distribution`
(`math::random::libcxx::uniform_real`).

**Why.** `std::rand()` is a process-global stream whose sequence depends on the C library and
on every earlier call in the process, so COLMAP's result in this branch is not reproducible
even against itself; porting it would add hidden global state shared across threads, which
CLAUDE.md rules out (sequential and parallel runs must agree). Eigen's `UnitRandom` code is
MPL-2.0 and is not ported, so the rotation follows the published method it cites. A fixed
per-call seed keeps the result a pure function of the input. Only the degenerate branch is
affected; the regular path is a straight port. Same as colmap-sharp entry 26.

**Evidence.** A change of variables followed by `refine_3q3` against the original
coefficients targets the same system, so the solution set agrees up to solver tolerance
whichever random matrix is drawn. `tests/estimators/rust_only_re3q3.rs`,
`rust_only_degenerate_elimination_uses_random_var_change` (x^2 = 1, y^2 = 4, z^2 = 9: every
elimination determinant is 0) finds all eight solutions, each satisfying the original system
to 1e-8. colmap-sharp cross-checked C++ PoseLib fa7280f (clang++, Eigen 3.4) in a scratch
harness: for `std::srand` seeds 1..20 it also returns 8 solutions with max residual below
3e-9, and for x^2 = 1, y^2 = 4, z = x + y (a linear third equation) it returns 0 solutions for
every seed, as the port does (upstream behavior, pinned by
`rust_only_linear_equation_returns_no_solutions_like_poselib`).

## 221. re3q3_rotation's pre-rotation uses a fixed-seed mt19937, not std::rand

**What differs.** PoseLib's `re3q3_rotation` (used by `gp3p`, which COLMAP's `GP3PEstimator`
calls; `estimators/solvers/poselib/re3q3.rs`, `solve_rotation`) rotates every problem by a
random rotation R0 before solving in Cayley parameters, so that no solution lands on the
Cayley transform's singularity (a rotation by pi). Unlike re3q3's change of variables (entry
220), this draw happens on *every* call, not only for degenerate input. PoseLib takes R0 from
Eigen's `Quaternion::UnitRandom()`, i.e. from the process-global `std::rand()`. The port draws
it (Shoemake's uniform rotation) from a fresh `Mt19937` seeded with 1 on every call.

**Why.** The same as entry 220: `std::rand()` is hidden global state whose sequence depends on
the C library and on every earlier call, so COLMAP's gp3p is not reproducible even against
itself, and porting it would make sequential and parallel runs disagree. Eigen's `UnitRandom`
is MPL-2.0 and not ported. The seed differs from entry 220's so the two draws are not the same
rotation. Consequence: for a fixed input the port always uses the same R0, so an input whose
true rotation is (close to) R0 composed with a rotation by pi is always ill-conditioned here,
where PoseLib would be only for an unlucky draw. That set has measure zero and RANSAC draws
other samples. Same as colmap-sharp entry 29.

**Evidence.** colmap-sharp ran PoseLib fa7280f's gp3p (clang++, Eigen 3.4) in a scratch
harness: it returns the same four poses for `std::srand` seeds 1..5 on a fixed three-ray
input, agreeing to about 1e-10 across seeds. `tests/estimators/rust_only_gp3p.rs`,
`rust_only_matches_poselib_solution_set` returns those same four poses to 1e-8.

## 230. CPU SIFT matches the macOS pycolmap wheel within FMA drift, not bit for bit

**What differs.** COLMAP's `SiftCPUFeatureExtractor` runs VLFeat, which the macOS arm64
pycolmap wheel compiles with Apple clang's default `-ffp-contract=on`: multiply-adds in the
Gaussian convolution, the DoG refinement and the descriptor binning are fused into FMAs that
round once. colmap-rust never fuses (CLAUDE.md "No FMA"), so against that wheel the extractor
(`feature/sift/cpu_extractor.rs`) gives the same keypoints, in the same order and count, with
last-bit drift: keypoint values differ by up to about 1e-3 and descriptor bytes by at most one
grey level, on at most 0.1% of the bytes.

**Why.** Fusing depends on the compiler and target, so it cannot be reproduced on every
platform (wasm has no guaranteed FMA); unfused VLFeat is the portable reference, and against it
the port is bit-exact. Same as colmap-sharp entry 41.

**Evidence.** `tests/feature/rust_only_vl_sift_filter.rs`: bit-exact against VLFeat compiled
with `-ffp-contract=off` (fixture `sift_vlfeat.json`). `tests/feature/rust_only_sift.rs`
`rust_only_sift_matches_pycolmap_*`: the nine `sift.json` cases against the wheel within
2e-3 per keypoint value, 1 per descriptor byte and 0.1% mismatched bytes.

## 231. FeatureDescriptorIndex is an exact nearest-neighbour search, not faiss's IVF index

**What differs.** COLMAP's `FeatureDescriptorIndex` (`feature/index.cc`) wraps faiss: an exact
`IndexFlatL2` for fewer than 512 indexed descriptors, and for 512 or more an inverted-file
index (`IndexIVFScalarQuantizer` with `QT_8bit_direct` for SIFT, `IndexIVFFlat` otherwise;
4 sqrt(N) k-means centroids, searched with `nprobe = 8`). The IVF search is approximate: it
only visits the 8 nearest clusters, so it can miss a query's true nearest or second-nearest
neighbour. The port (`colmap-rust/src/feature/index.rs`) always returns the exact k nearest
neighbours, ordered by (squared distance, index). SIFT descriptors are compared as integers
(exact); other descriptor types sum (a - b)^2 in `f32`, left to right, where faiss's flat index
uses the ||a||^2 + ||b||^2 - 2ab expansion.

**Why.** faiss is native code (CLAUDE.md contract 1). Porting it, k-means training included,
would reproduce a randomized approximation whose exact output also depends on faiss's OpenMP
partitioning and BLAS. The exact search is the result the IVF index approximates, and it makes
the default (index) matching path agree with COLMAP's own brute-force matcher, which is what
`SiftCPUFeatureMatcherFaissVsBruteForce` asserts. On an image pair with 512+ features, the SIFT
matches can therefore differ from COLMAP's default CPU matcher wherever faiss's approximation
missed a neighbour (COLMAP returns fewer or different matches there); they equal COLMAP's
`cpu_brute_force_matcher` matches except where acos-of-dot and sqrt-of-L2 round a threshold
comparison differently, which COLMAP's two paths also do. Ties in distance are ordered by
index; faiss leaves them unspecified, and the ratio test rejects a tie for the best neighbour
either way. Same as colmap-sharp entry 42.

**Evidence.** `tests/feature/index.rs` (`index_test.cc` 1:1, including the 1000-descriptor rows,
where COLMAP uses IVF) and `tests/feature/sift_matching.rs`
`sift_cpu_feature_matcher_faiss_vs_brute_force_nominal` (`sift_test.cc` 1:1) pass;
`rust_only_index_float_queries_on_byte_index_and_large_k_order` checks the order against the
definition.

## 240. Synthetic datasets visit points, images and chained pairs in ascending id order

**What differs.** COLMAP's `SynthesizeDataset` projects the reconstruction's 3D points into
each image, and `track_length` pruning visits them, in `Reconstruction::Points3D()` order,
which is the order of COLMAP's hash map; its CHAINED match configuration writes the pairs of a
`NodeHashMap`; `SynthesizeNoise` adds 2D and 3D point noise visiting `Images()` and
`Points3D()`, and `SynthesizeImages` renders the images in `Images()` order. `scene/synthetic.rs`,
`scene/synthetic_matches.rs` and `scene/synthetic_noise.rs` visit all of them in ascending id
order (the reconstruction's `BTreeMap`s, entry 213; a `BTreeMap` of pairs for CHAINED). The
PRNG draws are the same and in the same sequence; what differs is the order of each image's 2D
points before the shuffle, which point's track receives which pruning shuffle, which pair
receives which outlier-match shuffle, which point receives which noise draws, and the
descriptor seeds `SynthesizeImages` gives points without a 3D point. So a seeded run gives the
same cameras, rigs, frame poses and 3D point positions as pycolmap's, but different 2D point
orders, pruned tracks and noise assignment.

**Why.** CLAUDE.md requires deterministic iteration; COLMAP's tests rely only on order-free
properties (counts, track lengths, error bounds, connectivity, equality of two seeded runs).
Same as colmap-sharp entry 31.

**Evidence.** `tests/scene/synthetic.rs`, `synthetic_database.rs` and `synthetic_noise.rs`
(`synthetic_test.cc` 1:1, all 18 cases) and every Phase 4 test that synthesizes its scene pass.

## 241. Model file readers reject input COLMAP's streams misread

**What differs.** COLMAP's model readers use `std::istream`, which parses a value
character by character and does not check every read. The Rust readers:
- parse each text value as a whole whitespace-separated token (`util/line_tokens.rs`), so
  "12abc" fails where libc++ reads 12 and leaves "abc" for the next value;
- reject a sensor type number in `rigs.bin` / `frames.bin` that names no `SensorType`, and a
  camera model number that names no model (a Rust enum cannot hold an unnamed value);
- reject an image name or text line that is not UTF-8 (a Rust `String` cannot hold it);
- fail when a binary stream ends early (as `util/endian.rs` does), where COLMAP reads
  unspecified values or, for an image name, loops on the last character.

Valid files, including every file COLMAP writes, read to the same model.

**Why.** Each case is malformed input where COLMAP's result is unspecified or silently wrong;
an error is the only safe answer. Same as colmap-sharp entry 25 for the token parsing.

**Evidence.** `tests/scene/rust_only_reconstruction_io.rs`
`rust_only_reconstruction_io_rejects_what_cpp_misreads`; the pycolmap fixtures in
`tests/data/oracle/reconstruction_io` read and re-write byte for byte
(`rust_only_reconstruction_io_rewrites_pycolmap_files_byte_for_byte`).

## 242. Reading a model resets the reconstruction's registration

**What differs.** COLMAP's `Reconstruction::ReadText` / `ReadBinary` clear the rig, camera,
frame, image and 3D point maps, but keep the registered-frame list, the registered-image count
and the largest 3D point id. Reading into a reconstruction that already held registered frames
leaves frames that no longer exist registered, so `NumRegFrames()` is wrong and `IsValid()`
fails. Here `read_text` / `read_binary` start from an empty reconstruction.

**Why.** Stale registration of frames that are gone is a bug; nothing relies on it (COLMAP
reads into fresh reconstructions in practice).

**Evidence.** `tests/scene/rust_only_reconstruction_io.rs`
`rust_only_reconstruction_read_resets_registration` (COLMAP would report 5 registered frames
for a file with 3).

## 250. The five-point solver takes its null space from unpivoted Householder QR

**What differs.** PoseLib's `relpose_5pt` (the minimal case of COLMAP's
`EssentialMatrixFivePointEstimator`) takes the 4D null space of its five epipolar constraints
from the last four columns of the full Q of `epipolar_constraints.fullPivHouseholderQr()` (a
9 x 5 matrix). `estimators/solvers/poselib/relpose_5pt.rs` takes the same four columns from an
unpivoted Householder QR (`linalg::householder::factor_in_place` / `apply_q`). Both are
orthonormal bases of the same space; only the basis inside it differs, and with it the
intermediate polynomial coefficients (`compute_trace_constraints`, the degree-10 determinant
polynomial and its roots, which parametrize the solutions in basis coordinates).

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no published
convention precise enough to reproduce Eigen's Q column for column. The solution set does not
depend on the basis: the solver returns every unit-norm E = x N0 + y N1 + z N2 + N3 in the
null space that meets the cubic constraints, and a change of orthonormal basis maps that set
onto itself (up to the sign of each E). What can differ is numerical conditioning near the
back substitution's 1e-6 switch to the three-row QR solve and the Sturm bracketing tolerance,
where a near-degenerate root can be kept on one side and dropped on the other. Same as
colmap-sharp entry 27.

**Evidence.** `tests/estimators/rust_only_relpose_5pt.rs`,
`rust_only_recovers_true_essential_matrix`: on 100 random minimal problems every solution has
unit norm and satisfies the five epipolar constraints to 1e-8, and the true E is among the
solutions up to sign. colmap-sharp's 1:1 `EssentialMatrixFivePointEstimatorTests_Nominal`
cases pass on the same construction. Not compared against C++ PoseLib intermediates.

## 251. TinySphereManifold's tangent basis uses Hughes & Moller, not Eigen's unitOrthogonal()

**What differs.** COLMAP's `SphereManifold<3>` (`estimators/cost_functions/tiny_manifold.h`)
builds the tangent basis at x as `b1 = x_hat.unitOrthogonal()`, `b2 = x_hat.cross(b1)`.
`estimators/cost_functions/tiny_manifold.rs` (`TinySphereManifold`) builds `b1` with the
construction of Hughes & Moller, "Building an Orthonormal Basis from a Unit Vector" (JGT 4(4),
1999): zero the component of smallest magnitude (the first on a tie), swap the other two and
negate one, normalize. `b2` is the same cross product. The two constructions can pick a
different unit vector in the tangent plane, so the tangent coordinates of a step, and with them
TinySolver's Jacobi scaling and iterates, can differ from COLMAP's while spanning the same
plane.

**Why.** Eigen documents only that `unitOrthogonal()` returns some unit vector orthogonal to
the input; its choice for 3-vectors is an implementation detail of MPL-2.0 code we may not
transcribe (CLAUDE.md contract 2). Any orthonormal tangent basis gives a valid retraction and
Plus Jacobian; TinySolver is Tier C, so what must agree is the converged result, which does not
depend on the basis. Same as colmap-sharp entry 19.

**Evidence.** `tests/estimators/tiny_manifold.rs`, the 1:1
`sphere_manifold_plus_stays_on_unit_sphere` and `sphere_manifold_plus_jacobian_matches_finite_diff`
(orthogonality to x within 1e-12, the finite-difference Jacobian within 1e-6) and the product
manifold tests pass. colmap-sharp records the downstream effect of the same basis: where
TinySolver stops before converging (the 25-iteration cap of
`EssentialMatrixTangentSampsonEstimator`'s refinement, LO-RANSAC's ten refit rounds), the
relative poses its global-SfM oracle test re-estimates agree with pycolmap to 6.5e-7 deg.

## 252. TinyFocalSampsonErrorCostFunctor's Jacobian is closed-form, not Ceres autodiff

**What differs.** COLMAP's `TinyFocalSampsonErrorCostFunctor`
(`estimators/cost_functions/tiny_sampson_error.h`), the refiner of
`RelativePoseSharedFocalEstimator::Refine`, is a templated functor that TinySolver
differentiates through `ceres::TinySolverAutoDiffFunction` (Jets).
`estimators/cost_functions/tiny_sampson_error.rs` evaluates the same residual,
`SampsonError(F, point1, point2)` with F = diag(1/f, 1/f, 1) E diag(1/f, 1/f, 1), in the same
arithmetic order, and computes its 8-parameter Jacobian in closed form: dr/dF contracted with
dF/d[q, t, log_f], where the pose columns are COLMAP's dE/d[q, t] carried through the focal
scaling and d F/d(log_f) negates F with the top-left 2x2 block doubled and F(2, 2) zeroed. It
is the exact derivative of the same function, so it differs from autodiff's only by rounding;
TinySolver's iterates can therefore differ in the last bits.

**Why.** The closed form is the pattern COLMAP itself uses for the neighboring tangent,
one-sided focal and fundamental functors in the same header, and it is cheaper than autodiff.
TinySolver is Tier C, so what must agree is the converged result.

**Evidence.** `tests/estimators/tiny_sampson_error.rs`:
`tiny_focal_sampson_error_cost_functor_matches_squared_sampson_error` (the 1:1
`TinyFocalSampsonErrorCostFunctor.MatchesSquaredSampsonError`, residuals within 1e-9) and
`rust_only_tiny_focal_sampson_error_cost_functor_jacobian_matches_finite_difference` (every
column, log-focal included, against central differences within 1e-5 at two poses and focals;
scaling the log-focal column wrongly fails it). `tests/estimators/relpose_shared_focal.rs`'s
1:1 `relative_pose_shared_focal_estimator_refine_from_initial_model` converges from a 10%
focal error to within 1% on all 50 draws, and fails with that same mutation.

## 255. The six-point focal relative pose solvers take their null space from unpivoted Householder QR

**What differs.** PoseLib's `relpose_6pt_shared_focal` and `relpose_6pt_onesided_focal` (the
minimal solvers of COLMAP's `RelativePoseSharedFocalEstimator` and
`RelativePoseOneSidedFocalEstimator`) take the 3D null space of their six epipolar
constraints from the last three columns of the full Q of
`epipolar_constraints.fullPivHouseholderQr()` (a 9 x 6 matrix).
`estimators/solvers/poselib/relpose_6pt_shared_focal.rs` (`null_space3`, used by both) takes
the same three columns from an unpivoted Householder QR (`linalg::householder::factor_in_place`
/ `apply_q`). Both are orthonormal bases of the same space; only the basis inside it differs,
and with it the 280 (shared) or 190 (one-sided) template coefficients, the action matrix and
its eigenvalues, which parametrize F = N0 + x N1 + y N2 in basis coordinates. The one-sided
solver also orders its nine solutions by the eigenvalue order of `linalg::EigenSolver` (Schur
order), which may differ from Eigen's.

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no published
convention precise enough to reproduce Eigen's Q column for column (entry 250 is the same
choice for the five-point solver). In exact arithmetic the solution set does not depend on the
basis (apart from solutions whose F has no N0 component, which the affine parametrization
cannot represent; a measure-zero configuration that moves with the basis). In floating point
it can: the shared-focal solver brackets the real roots of an ill-conditioned degree-15
characteristic polynomial with Sturm sequences, and which spurious (non-physical) real roots
survive that bracketing depends on the basis, so extra models can appear or vanish on either
side. The output order is not part of the contract: COLMAP treats the models as a set. What
can differ is numerical conditioning near the solvers' thresholds (the Sturm bracketing
tolerance, the 1e-8 tests on imaginary parts and on q = 1/f^2). Same as colmap-sharp entry 30.

**Evidence.** `tests/estimators/rust_only_relpose_6pt_focal.rs`: on 50 random minimal problems
each, both solvers return the true pose and focal (relative error below 1e-4; well-conditioned
problems agree to 1e-12, one shared-focal problem was 5e-6 off). colmap-sharp's differential
run against the real PoseLib (its entry 30) found the one-sided models identical and the
shared-focal models matching except for spurious roots. Not compared against C++ PoseLib here.

## 256. The generalized six-point solver returns its poses in Schur eigenvalue order

**What differs.** PoseLib's `gen_relpose_6pt` (the minimal solver of COLMAP's `GR6PEstimator`)
takes the real eigenvalues (|imag| < 1e-6) of its 64 x 64 action matrix from
`Eigen::EigenSolver` and returns one pose per eigenvalue that passes cheirality, in Eigen's
eigenvalue order. `estimators/solvers/poselib/gen_relpose_6pt.rs` takes them from
`linalg::EigenSolver` (colmap-sharp's Hessenberg + Francis QR port), whose eigenvalue order
(Schur order) and last-bit rounding may differ from Eigen's. It also computes
`C0^-1 C1` column by column through `linalg::PartialPivLu` rather than colmap-sharp's fused
augmented elimination, which colmap-sharp documents as bit-identical.

**Why.** Eigen (MPL-2.0) is not ported. Only eigenvalues are used, never eigenvectors, so the
eigenvector phase freedom cannot reach the result; the solution set is the same up to
rounding. The order is not part of the contract (RANSAC treats the models as a set). What can
differ is an eigenvalue whose imaginary part sits at the 1e-6 threshold, kept on one side and
dropped on the other. Same choice as colmap-sharp's `GenRelpose6pt.cs` (which records it in
its file header, not as a numbered entry).

**Evidence.** `tests/estimators/rust_only_gen_relpose_6pt.rs`
`rust_only_gen_relpose_6pt_recovers_true_pose`: on 50 random minimal problems between two
generalized cameras the true metric pose is among the solutions to 1e-6. Not compared against
C++ PoseLib.

## 260. The 7-point fundamental solver takes its null space from unpivoted Householder QR

**What differs.** `FundamentalMatrixSevenPointEstimator::Estimate` gets the 2D null space of
its 7 x 9 constraint system from columns 7 and 8 of the full Q of `A.fullPivHouseholderQr()`
(A is 9 x 7). `estimators/solvers/fundamental_matrix.rs` takes the same two columns from an
unpivoted Householder QR of A (`linalg::householder::factor_in_place` / `apply_q`,
allocation-free). Both pairs are orthonormal bases of the same space: in any factorization
A = Q R, the first seven columns of Q contain A's column space, so the last two are orthogonal
to every constraint, whatever the pivoting and whatever A's rank. Only the basis inside that
plane differs.

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR (which also swaps
rows) has no published convention precise enough to reproduce Eigen's Q column for column
(entries 250 and 255 are the same choice for the five- and six-point solvers). The basis does
not reach the solutions: the solver finds the rank-2 members of the pencil {a f1 + b f2},
normalized to unit norm, and that set of matrices (up to sign, which the Sampson error and
COLMAP's F / F(2, 2) comparisons ignore) does not depend on the basis. The one place it shows
is the parametrization lambda f1 + f2 with f1 = q7 - q8: when |det(f1)| < 1e-16 COLMAP returns
no model, and which member of the pencil f1 is depends on the basis. For a generic sample that
is a measure-zero event on either side, so the two can disagree only on a sample that sits on
that numeric knife edge, and then one of them emits no model for that RANSAC hypothesis. Same
as colmap-sharp entry 24.

**Evidence.** `tests/estimators/fundamental_matrix.rs`:
`fundamental_seven_point_estimator_reference` (COLMAP's Matlab reference values, 1e-6) and
`fundamental_seven_point_estimator_nominal` (100 random problems, at least one model equal to
the true F up to scale) pass 1:1.

## 261. GR8P returns its four candidate poses in Schur eigenvector order

**What differs.** `GR8PEstimator::Estimate` reads the translations of its four candidate poses
from the eigenvectors of the 4 x 4 matrix G (`Eigen::EigenSolver<Matrix4d>`,
`V.real().colwise().hnormalized()`) and returns them in Eigen's eigenvector order.
`estimators/solvers/generalized_relative_pose_gr8p.rs` takes them from `linalg::EigenSolver`
(colmap-sharp's Hessenberg + Francis QR port), whose eigenvalue order (Schur order) and
last-bit rounding may differ from Eigen's, so the four candidates can come in a different
order.

**Why.** Eigen (MPL-2.0) is not ported, and its deflation order is not a documented
convention. G is symmetric (`ComposeG` mirrors every off-diagonal entry), so every eigenvalue
is real and so is its eigenvector, and `hnormalized()` divides out the scale and sign that are
the only freedom left: the candidate set is the same up to rounding. RANSAC scores all four,
so the order only matters when two candidates have exactly the same support and score, where
the first one wins. Same choice as colmap-sharp's `GeneralizedRelativePose.GR8P.cs` (which
records it in its file header, not as a numbered entry).

**Evidence.** `tests/estimators/generalized_relative_pose.rs`: the seven
`..._gr8p_*` instantiations of COLMAP's `ParameterizedGRNPEstimatorTests.GR8P` pass 1:1
(RANSAC over 16 correspondences finds the true `rig2_from_rig1` within rtol and ttol 1e-2 in
one of ten trials). Not compared against C++ Eigen.

## 262. IsPanoramicRig compares rig camera origins to the smallest camera index

**What differs.** `IsPanoramicRig` (estimators/generalized_pose.cc) decides whether all
cameras used by a set of correspondences share one optical center: it puts the camera indices
in a `FlatHashSet`, takes the set's first element as the reference, and checks every other
camera's origin in the rig with `isApprox(reference, 1e-6)`. The set's iteration order is
unspecified. `estimators/generalized_pose.rs` takes the smallest camera index as the
reference. Same as colmap-sharp entry 46.

**Why.** Hash iteration order is not reproducible across containers (CLAUDE.md, hash
iteration order). The reference only matters when the origins agree to about 1e-6 relative
with one camera but not with another (`isApprox` is not transitive), which never happens for a
real rig, whose cameras are either co-centered (panoramic) or centimeters apart.

**Evidence.** `tests/estimators/generalized_pose.rs`: COLMAP's
`EstimateGeneralizedRelativePose.Nominal`, which covers panoramic and non-panoramic rigs, and
`EstimateStructureLessAbsolutePose.PanoramicWorldCameras` pass 1:1.

## 263. The generalized pose estimators return no pose for empty or misaligned input

**What differs.** In estimators/generalized_pose.cc, `ThrowCheckCameras` dereferences
`std::minmax_element` of the camera index list, and `EstimateStructureLessAbsolutePose` calls
`IsPanoramicRig`, which dereferences the first element of the index set: both are undefined
behavior for an empty list. `EstimateGeneralizedRelativePose` also reads `camera_idxs1[i]` and
`camera_idxs2[i]` for every point without checking their lengths. In
`estimators/generalized_pose.rs`, an empty index list has nothing to check,
`estimate_structure_less_absolute_pose` returns `Ok(None)` (COLMAP's `false`) for no
correspondences, as the other two estimators already do, and
`estimate_generalized_relative_pose` fails a check (`camera_idxs1.len() == num_points`) when
an index list is shorter or longer than the points.

**Why.** An out-of-range read is a bug, not behavior to reproduce. Returning "no pose" for no
correspondences matches what COLMAP's other two estimators do, and colmap-sharp makes the same
choice (in comments, not as a numbered entry). Results for valid input are unchanged.

**Evidence.** `tests/estimators/rust_only_generalized_pose.rs`
(`rust_only_generalized_pose_empty_correspondences_find_no_pose`,
`rust_only_generalized_relative_pose_short_camera_idxs_is_an_error`).

## 264. Two-view geometry estimation does not log

**What differs.** COLMAP's `EstimateTwoViewGeometry` emits a one-time `LOG_FIRST_N(WARNING)`
when it marks a pair degenerate because `force_H_use` was asked of a non-pinhole camera, and
another when it marks an uncalibrated fisheye pair degenerate (asking for a focal length prior);
`MaybeDecomposeRelativePoses` logs `LOG(INFO)` lines with its decomposed and failed counts and
its run time. `colmap_rust::estimators::two_view_geometry` returns the same geometries without
these lines, and drops the `Timer` that only fed the last one.

**Why.** The core crate has no logging facility yet (COLMAP's glog is not ported); the lines are
diagnostic only and every returned geometry, configuration and stored pose is unchanged. When a
logger lands, this entry goes away (as entry 211 and the same note on
`geometry::pose_prior::gravity_from_exif_orientation` do).

**Evidence.** `tests/estimators/two_view_geometry.rs`
`estimate_two_view_geometry_uncalibrated_fisheye_is_degenerate` and
`estimate_two_view_geometry_force_h_use_with_fisheye_is_degenerate`, and
`tests/estimators/two_view_geometry_misc.rs` `maybe_decompose_relative_poses_*`
(`two_view_geometry_test.cc` 1:1) pass.

## 265. The database is one in-memory type, not a factory over SQLite files

**What differs.** COLMAP's `Database` (scene/database.h) is an abstract class whose only
implementation is `SqliteDatabase` (database_sqlite.cc), opened through a path-keyed factory
registry (`Database::Open`, `Database::Register`); `kInMemorySqliteDatabasePath` opens an
in-memory SQLite database. `colmap_rust::scene::database::Database` is one concrete type that
keeps its tables in memory: `Database::new()` stands for `Open(kInMemorySqliteDatabasePath)`,
and there is no `Open(path)`, no `Register` and no file. Its methods take `&self` (the tables
sit behind a mutex, so a database can be shared between threads like COLMAP's `shared_ptr`) and
return `Result`, so a closed database fails every call as COLMAP's does. `BeginTransaction` /
`EndTransaction` only check that the database is open: every call is applied at once, and
COLMAP never rolls a transaction back. `DatabaseTransaction` still holds the transaction lock,
so only one is active at a time.

**Why.** SQLite database files are out of scope (PORTING_PLAN.md: SQLite is native code), and a
trait with a single implementation would only add indirection. Everything the SQLite schema
makes observable is reproduced: AUTOINCREMENT ids for rigs, cameras, frames and images (one more
than the largest id the table ever held, explicit ids included; clearing does not reset it), the
pose priors' plain INTEGER PRIMARY KEY (one more than the largest current id), the UNIQUE
indices (image names, a sensor in at most one rig as reference and one as non-reference sensor,
a data id in at most one frame, one pose prior per data id, one keypoints / descriptors /
matches / two-view geometry row per image or pair), the foreign keys with their cascades
(clearing rigs drops their frames, clearing images their keypoints and descriptors, clearing
cameras fails while images exist), rows read in id order and pairs in pair id order, UPDATE of a
missing row doing nothing, reads of a missing row returning defaults, and the two-view geometry
columns (`tri_angle` is not stored, so a read returns its default).

**Evidence.** `tests/scene/database.rs` (database_test.cc 1:1; the three `OpenFile*` cases are
skipped as file-based) and `tests/scene/rust_only_database.rs`. Same as colmap-sharp's
`InMemoryDatabase` (its `Scene/InMemoryDatabase.cs` header).

## 266. A database write that violates a constraint stores nothing

**What differs.** COLMAP's `WriteRig` / `UpdateRig` and `WriteFrame` / `UpdateFrame` issue one
INSERT for the rig or frame row and then one per sensor or data id, each in SQLite's autocommit
mode; when a later INSERT violates a constraint (say, a sensor already in another rig), COLMAP
throws after the earlier rows were committed, leaving a rig or frame with part of its sensors or
data. Here every write checks all constraints first, so a failing call leaves the database
unchanged. Consequences:
- On successful writes the assigned ids, what is stored and every read are the same.
- After such a partially failed write COLMAP has consumed the new id (the row exists), so its
  next automatic id is one higher; here the id was never used and the next write gets it.
- The same calls fail, but the messages differ: COLMAP's `std::runtime_error` reads "SQLite
  error: constraint failed" (`sqlite3_errstr`); here it is `ErrorKind::RuntimeError` worded like
  SQLite's extended messages ("SQLite error: UNIQUE constraint failed: images.name", "SQLite
  error: FOREIGN KEY constraint failed"). `THROW_CHECK` sites keep COLMAP's check messages.
- An automatic id past `u32::MAX` fails ("the next id does not fit a 32-bit id") where COLMAP
  truncates SQLite's 64-bit rowid to the 32-bit id.

**Why.** The partially written state is an accident of statement-at-a-time execution, not
behavior any COLMAP caller relies on (all of them treat the exception as fatal), and reproducing
it would mean emulating SQLite's statement boundaries. A truncated id would silently alias an
existing row.

**Evidence.** `tests/scene/rust_only_database.rs` `rust_only_failed_rig_write_stores_nothing`
and `rust_only_image_names_are_unique_and_need_a_camera`; database_test.cc's cases that expect a
constraint violation to fail (`PosePrior`, `TwoViewGeometry`) pass. Same as colmap-sharp entry
23.

## 267. DatabaseCache iterates its objects in ascending id order

**What differs.** COLMAP's `DatabaseCache` keeps rigs, cameras, frames and images in
`NodeHashMap`s, whose iteration order depends on the hash-map backend of the build.
`colmap_rust::scene::database_cache::DatabaseCache` keeps them in `BTreeMap`s, so `rigs()`,
`cameras()`, `frames()` and `images()` enumerate in ascending id order, like `Reconstruction`
(entry 213). Where that order reaches an output: `find_image_with_name` returns the
smallest-id image when several share a name (possible only through `add_image`; database names
are unique), and the correspondence graph receives its images in ascending id order (the order
of `AddImage` calls is not observable). Pairs are added in the database's pair id order in
`load` and in the source graph's insertion order (entry 210) in `create_from_cache`, both as in
COLMAP. `Load` keeps COLMAP's `emplace` semantics: an object already in the cache is not
replaced.

**Why.** CLAUDE.md requires deterministic iteration. COLMAP relies on no particular order here
(its tests use order-free properties), so ascending id is the natural choice.

**Evidence.** `tests/scene/database_cache.rs` (database_cache_test.cc 1:1) passes with COLMAP's
expectations; `rust_only_create_from_cache_keeps_whole_frames` pins the filtering. Same as
colmap-sharp entry 33.

## 268. SynthesizeImages hands bitmaps to a sink instead of writing image files

**What differs.** COLMAP's `SynthesizeImages(options, reconstruction, image_path)` writes each
image to `image_path / image.Name()` with `Bitmap::Write` (OpenImageIO) and logs a failed
write. `colmap_rust::scene::synthetic_noise::synthesize_images(options, reconstruction,
write_image)` calls `write_image(image.name(), bitmap)` for each rendered RGB bitmap; the
pixels are the same.

**Why.** Image file encoding is not ported: the host owns image I/O (PORTING_PLAN.md, the
skipped `bitmap_test.cc` file cases). A sink lets a caller encode files with its own encoder or
keep the bitmaps in memory, natively and in the browser. Same as colmap-sharp entry 32.

**Evidence.** `tests/scene/synthetic_noise.rs` `synthesize_images_nominal` checks every image
reaches the sink with its camera's width and height (COLMAP's test reads each PNG back and
checks the same).

## 269. SynthesizeDataset rejects pose priors without a database

**What differs.** COLMAP's `SynthesizeDataset` with `prior_position` or `prior_gravity` set
writes each image's pose prior through `database->WritePosePrior` without checking the
pointer, so calling it without a database dereferences null (undefined behavior, in practice a
crash). `colmap_rust::scene::synthetic::synthesize_dataset` (the database-free form) returns
an `InvalidArgument` error, "SynthesizeDataset: pose priors need a database", at the first
image instead.

**Why.** A null dereference is a bug, not behavior to reproduce; an error is the only safe
answer, and the database form behaves exactly as COLMAP's.

**Evidence.** `tests/scene/synthetic_database.rs`
`rust_only_synthesize_dataset_pose_priors_need_a_database`.

## 273. TranscribeImageIdsToDatabase changes nothing when it fails

**What differs.** COLMAP's `Reconstruction::TranscribeImageIdsToDatabase` renames the images
one by one while it looks their names up in the database, so when a name is missing
(`LOG(FATAL_THROW)`) or two images map to one id (`THROW_CHECK`), the images before it already
carry their new ids while the frames and tracks still refer to the old ones: the reconstruction
is left broken. `colmap_rust::scene::reconstruction_database::transcribe_image_ids_to_database`
looks every id up, and checks every frame data id and track element has a mapping, before it
changes anything, so a failure returns the error and leaves the reconstruction as it was. The
successful result is the same.

**Why.** A half-applied renumbering is a bug (`IsValid` fails afterwards); checking first costs
one extra pass and makes the error recoverable.

**Evidence.** `tests/scene/reconstruction_database.rs`
`rust_only_transcribe_image_ids_to_database_unknown_name_changes_nothing`; COLMAP's
TranscribeImageIdsToDatabase test passes 1:1.

## 274. ReadRigConfig requires exactly 4 rotation and 3 translation entries

**What differs.** COLMAP's `ReadRigConfig` (scene/rig.cc) writes the entries of
`cam_from_rig_rotation` and `cam_from_rig_translation` into a fixed-size `Eigen::Vector4d` and
`Rigid3d::translation()` by running index, without checking the count.
`colmap_rust::scene::rig::read_rig_config` returns a "Check failed" error ("... must have 4
entries" / "... must have 3 entries") unless there are exactly 4 and 3 entries.

**Why.** Fewer entries leave COLMAP with uninitialized coefficients and more entries write out
of bounds; both are undefined behavior with no result to match. Every well-formed config reads
the same. Same as colmap-sharp entry 80.

**Evidence.** `tests/scene/rig.rs` (rig_test.cc 1:1) passes, including
`read_rig_config_nominal`, which reads a 4-entry rotation and a 3-entry translation;
`tests/scene/rust_only_rig.rs` `rust_only_read_rig_config_pose_needs_four_and_three_entries`.

## 275. The rig config JSON reader limits nesting depth

**What differs.** COLMAP reads rig configs with `boost::property_tree::read_json`, whose parser
recurses once per nested array or object without a limit, so a deeply nested file overflows the
stack. `colmap_rust::util::json::parse_json`, the core's own reader that stands in for it,
returns an error ("nesting deeper than 512 levels") past `MAX_DEPTH` (512) nested arrays and
objects. Every document within the limit parses as Boost's parser does (strict JSON).

**Why.** A stack overflow is a crash, natively and in the browser (whose wasm stack is small);
an error is the only safe answer. Real rig configs nest three levels.

**Evidence.** `tests/util/rust_only_json.rs` `rust_only_json_nesting_limit` (512 levels parse,
513 fail, a million fail without a crash).

## 276. ApplyRigConfig hands the reconstruction its rigs and frames in id order

**What differs.** COLMAP's `UpdateRigsAndFramesFromDatabase` (scene/rig.cc) collects the
reconstruction's rigs and frames in `NodeHashMap`s and passes them to
`Reconstruction::SetRigsAndFrames` in the maps' iteration order, and `AddFrame` registers posed
frames in that order, so it decides the order of `RegFrameIds()`.
`colmap_rust::scene::rig_apply::apply_rig_config` collects them in `BTreeMap`s, so the frames
are registered in ascending frame id order.

**Why.** In COLMAP 4.2.0 `NodeHashMap` is `boost::unordered_node_map` or `std::unordered_map`,
depending on the build (util/hash_containers.h). Either way its iteration order is
implementation-defined: it depends on the build configuration, the hash and the standard
library, so there is no single order to match, and CLAUDE.md requires a deterministic order.
The sets of rigs, frames and registered frames do not depend on it. Same as colmap-sharp
entry 81.

**Evidence.** `tests/scene/rig.rs` `apply_rig_config_*` (rig_test.cc 1:1) pass with COLMAP's
counts of rigs, frames and registered frames, none of which depends on the order;
`tests/scene/rust_only_rig.rs`
`rust_only_apply_rig_config_derives_sensor_from_rig_from_reconstruction` pins the ascending
registration order.

## 277. Reconstruction alignment and merging use deterministic orders and do not log

**What differs.** Three choices in `estimators/alignment.cc` depend on hash-map iteration order
in COLMAP:
- `AlignReconstructionsViaPoints` counts, per source 3D point, how often each target 3D point is
  seen along its track in a `FlatHashMap` and takes `std::max_element` over it, so among target
  points with the same count the hash order picks one.
  `colmap_rust::estimators::alignment::align_reconstructions_via_points` keeps the counts in
  first-seen order along the track and takes the first maximum.
- `MergeReconstructions` copies the missing source images in the iteration order of a
  `FlatHashSet`, which decides the order their frames are registered in the target
  (`RegFrameIds()`). `estimators::alignment::merge_reconstructions` copies them in the source's
  `RegImageIds()` order. It merges the source points in ascending id order (entry 213), which
  decides the ids the new target points get.
- `AlignReconstructionToOrigRigScales` sums the per-rig scales over a `NodeHashMap` of rigs;
  `align_reconstruction_to_orig_rig_scales` takes a `BTreeMap` and sums in ascending rig id
  order, so the floating-point sum is the same on every run.

COLMAP's `LOG(WARNING)` (too few pose priors, no prior covariance), `LOG(ERROR)` (inconsistent
image id and name in a merge), `LOG(INFO)` and `VLOG` lines are not emitted (the core has no
logger); the return values are COLMAP's.

**Why.** The hash orders are implementation-defined (they depend on the build's hash-map
backend and its history), so there is no single COLMAP order to match, and CLAUDE.md requires a
deterministic order. The selected point, the set of merged images and points and the scale
mean do not depend on the order beyond ties and summation rounding. Same as colmap-sharp
entry 49.

**Evidence.** `tests/estimators/alignment.rs` (`alignment_test.cc` 1:1) passes with COLMAP's
values and tolerances, including `MergeReconstructions`' counts of rigs, cameras, frames,
images, points and observations and `AlignReconstructionToOrigRigScales`' recovered rig
extrinsics.

## 278. Rotation averaging lays out its linear system in ascending id order

**What differs.** COLMAP's `RotationAveragingProblem` allocates frame parameters in the
iteration order of a `FlatHashSet<frame_t>`, camera (unknown cam_from_rig) parameters in the
order of a `NodeHashMap<camera_t, int>`, and constraint rows in the order of
`PoseGraph::ValidEdges()` (a `NodeHashMap`); `ComputeResiduals` and the IRLS weights walk the
pair constraints in `NodeHashMap<image_pair_t, ...>` order, which is also the order of the
jitter draws near the +-pi boundary of the 1-DOF residual. The gauge is fixed at the *first*
active frame in hash order (the first gravity-aligned one, if any).
`colmap_rust::estimators::rotation_averaging` allocates frames and cameras in ascending id
order, rows in ascending pair id order, visits constraints in that order, and fixes the gauge
at the smallest-id (gravity-aligned, if any) active frame.

The driver code does the same where COLMAP walks hash containers: the maximum spanning tree
numbers the active images in ascending id order (COLMAP: `FlatHashSet` order, and node 0 is
the tree's root, which is held at the identity rotation), adds edges in ascending pair id
order (which breaks weight ties) and visits each node's children in ascending id order; the
stratified gravity subset copies its edges in ascending pair id order;
`EstimateRotations` registers the frames in ascending image id order (which decides the order
of `reg_frame_ids`); `InitializeRigRotationsFromImages` walks frames in ascending id order,
which fixes the order in which quaternion samples are summed; `CreateExpandedReconstruction`
numbers its singleton rigs in ascending rig id (then sensor) order and its new frames in
ascending frame id (then data id) order.

COLMAP's `BuildPairConstraints` looks a pair's frame parameter up with `operator[]`, which
silently inserts index 0 for a frame outside the active set (`ComputeResiduals` then throws
from `.at()`). Here `RotationAveragingProblem::new` fails for such a pair instead. Callers
invalidate pairs outside the active set first, so this changes no successful run.

Same as colmap-sharp entry 44.

**Why.** CLAUDE.md's hash-order rule: the layout changes the floating-point grouping of the
normal equations and the Cholesky ordering, the jitter draw order, and the gauge (the solution
is only defined up to a global rotation, so a different fixed frame gives a globally rotated
result). COLMAP relies on none of these: its tests compare gauge-invariant relative rotations.

**Evidence.** `tests/estimators/rotation_averaging.rs` (rotation_averaging_test.cc 1:1, all 15
tests) passes with COLMAP's tolerances, including `DeterministicRandomSeed` (bit-identical
repeat solves) and the 1e-12 degree `RidgeRegularizationDoesNotBiasSolution`.

## 280. Reading a truncated binary PLY mesh fails inside the texcoord lists

**What differs.** COLMAP's `ReadPlyMesh` (util/ply.cc) checks `file.good()` after every vertex,
face count and face index it reads from a binary file, but not after a face's texcoord count
or its six UVs. A file cut there leaves those values indeterminate: usually the count is not 6
and the read fails with "Expected 6 texture coordinates per triangular face", a cut inside a
middle face's UVs fails at the next face, and a cut inside the last face's UVs loads the mesh
with garbage UVs and no error. `colmap_rust::util::ply::read_ply_mesh` checks the count and
every UV and fails with "Unexpected end of PLY file at face i" for the face whose texcoords are
cut, in all three cases. Same as colmap-sharp entry 113.

**Why.** The same robustness gap as a short binary stream (`util/endian.rs` header): a
half-written mesh must not load with garbage texture coordinates, and the error should name the
cause and the face. Well-formed files read identically.

**Evidence.** Reading of ply.cc (the unchecked `file.read` calls for `num_texcoords` and `uv`).
`tests/util/rust_only_ply.rs` `rust_only_truncated_binary_texcoords_fail` cuts a one-face
textured mesh before the texcoord count and after two of its six UVs (the last-face case); the
ported `ply_test.cc` cases, which read complete files, pass unchanged.

## 281. The color extractors take images from the host instead of reading image files

**What differs.** COLMAP's `Reconstruction::ExtractColorsForImage(image_id, path)` and
`ExtractColorsForAllImages(path, num_threads)` read each image from `path / image.Name()` with
`Bitmap::Read(..., as_rgb = true)` (OpenImageIO); a file that cannot be read makes the first
return `false` and is skipped (with a log warning) by the second.
`colmap_rust::scene::reconstruction_ply` takes a `load_image(name) -> Option<Bitmap>` callback
instead: `None` is the unreadable file. `extract_colors_for_all_images` has no `num_threads`
and accumulates each point's color sum in registered-image order, which is COLMAP's result
with one thread (with more, COLMAP adds per-thread partial sums in a scheduling-dependent
order, which can move a mean that lands near a rounding tie).

**Why.** Image file decoding is not ported: the host owns image I/O, natively and in the
browser (as for `SynthesizeImages`, entry 268). The core is single-threaded (CLAUDE.md,
Threading).

**Evidence.** `tests/scene/reconstruction_ply.rs` `reconstruction_extract_colors_for_all_images`
(COLMAP's test with the deleted image file as a missing map entry) and
`tests/scene/rust_only_reconstruction_ply.rs`
`rust_only_extract_colors_for_image_colors_black_points_only`.

## 282. PoseGraph orders equally large frame components by smallest frame id

**What differs.** COLMAP's `PoseGraph::ConnectedFrameComponents` collects the frames of the
valid edges in a `FlatHashSet`, finds the components (whose order follows that set, entry 42)
and orders them with an unstable `std::sort` by descending size, so the order of equally large
components is unspecified. `LargestConnectedFrameComponent` keeps the first strictly largest
component in the same hash order. `colmap_rust::scene::pose_graph::PoseGraph` hands
`find_connected_components` the frames in ascending id order and sorts stably, so equally large
components come out by their smallest frame id, and `largest_connected_frame_component` returns
that same first one. `connected_image_ids_for_frame_components` follows the same component
order. The edges live in a `BTreeMap` (ascending pair id) where COLMAP's `NodeHashMap` has its
hash order; no output of the pose graph itself depends on it. Same as colmap-sharp entry 38.

**Why.** CLAUDE.md's stable-sort and hash-order rules: the tie order decides which component
global SfM treats as the largest. COLMAP relies on no particular order: `pose_graph_test.cc`
uses components of different sizes and compares sets.

**Evidence.** `tests/scene/pose_graph.rs` (pose_graph_test.cc 1:1) passes with COLMAP's
expectations; `tests/scene/rust_only_pose_graph.rs`
`rust_only_equal_size_components_ordered_by_smallest_frame_id` pins the tie order and that the
largest component equals the first one.

## 283. ReconstructionManager::write breaks point-count ties by index

**What differs.** COLMAP's `ReconstructionManager::Write` orders the models with `std::sort` by
descending `NumPoints3D` and writes them to `0/`, `1/`, ...; the order of models with the same
point count is unspecified. `colmap_rust::scene::reconstruction_manager::ReconstructionManager`
sorts stably, so equal-sized models keep their manager order. Same as colmap-sharp entry 34.

**Why.** CLAUDE.md's stable-sort rule: the tie order decides which folder a model lands in.
libc++'s `std::sort` uses insertion sort for short ranges, which also keeps equal elements in
order, so for the handful of models a mapper produces the result is the same as COLMAP's; only
a long run of equal counts could differ.

**Evidence.** `tests/scene/rust_only_reconstruction_manager.rs`
`rust_only_write_orders_by_point_count_then_index` writes models with point counts 1, 2, 1, 2
and reads back sources 1, 3, 0, 2.

## 284. Scene and reconstruction clustering break sort ties deterministically

**What differs.** Three `std::sort` calls leave equal keys in an unspecified order:
`SceneClustering::PartitionHierarchicalCluster` sorts the overlap-candidate edges of each
child by descending weight, `SceneClustering::PartitionFlatCluster` sorts each image's
related images by descending weight (scene_clustering.cc), and `EstablishStrongClusters`
(reconstruction_clustering.cc) sorts the union-find clusters by descending size after
collecting them from a `NodeHashMap`, so equal-size clusters are numbered in hash order.
`colmap_rust::scene::scene_clustering` keeps equal weights in edge order (a stable sort), and
`colmap_rust::scene::reconstruction_clustering` numbers equal-size clusters by ascending
smallest frame id. When a weight tie straddles the overlap budget, a different overlap image
can be chosen; equal-size reconstruction clusters can get swapped ids. Same as colmap-sharp
entry 93.

**Why.** The translation rules require an explicit, deterministic tie-break wherever tie
order reaches an output. Edge order is what the macOS SDK's libc++ `std::sort` gives for short
lists: lengths 2-5 go through `__sort3/4/5`, which never reorder equal elements under these
`>` comparators; lengths 6-23 go through insertion sort, which is stable; lengths of 24 or more
go through pdqsort, which is not. So an overlap list or related-image list of 24+ entries with
tied weights can come out in a different order than COLMAP's; every such list in COLMAP's tests
has at most 12 entries. Reproducing pdqsort and absl's or Boost's hash order would tie the
result to one library version.

**Evidence.** All eight `scene_clustering_test.cc` cases (`tests/scene/scene_clustering.rs`)
and all nine `reconstruction_clustering_test.cc` cases
(`tests/scene/reconstruction_clustering.rs`) pass;
`tests/scene/rust_only_reconstruction_clustering.rs`
`rust_only_equal_size_clusters_numbered_by_smallest_frame_id` pins the cluster numbering (and
fails when the tie-break is reversed).

## 290. Flat scene clusters are ordered by size, then smallest image id

**What differs.** `SceneClustering::PartitionFlatCluster` sorts the child clusters with the
comparator `size(a) >= size(b) && min(a) < min(b)`, which is not a strict weak ordering
(`std::sort` then has undefined behavior) and dereferences `min_element` of an empty cluster.
`colmap_rust::scene::scene_clustering` orders the children by what COLMAP's comment says it
intends: descending size, then ascending smallest image id, with empty clusters last (stable
among themselves). Only the order of `root_cluster().child_clusters` / `leaf_clusters()` is
affected; each child's overlap images depend only on its own members. In practice the orders do
differ: with 3-5 children libc++'s `__sort3/4/5` only swap when the comparator says so, so a
child of size 2 with smallest id 1 stays ahead of a child of size 3 with smallest id 5 (neither
compares "less" than the other), while the port puts the larger child first. The flat overlap
loop also stops once `j` passes the longest related-image list of the child, where COLMAP keeps
counting up to `num_image_matches` (a negative value, "no limit" as `size_t`, would spin for
2^64 rounds); those rounds cannot add an image, so the result is the same. Same as colmap-sharp
entry 94.

**Why.** Undefined behavior cannot be ported; the comment states the intent.

**Evidence.** `tests/scene/rust_only_scene_clustering.rs`
`rust_only_flat_child_clusters_ordered_by_size_then_smallest_id` pins the order;
`scene_clustering_three_flat_clusters` and `_three_flat_clusters_two_overlap` (1:1,
order-insensitive as in COLMAP) pass.

## 291. SceneClustering::create hands the image pairs over in Boost hash order

**What differs.** COLMAP's `SceneClustering::Create` builds the edge list by iterating the
`NodeHashMap` from `NumMatchesBetweenAllImages()`, so the edges come in hash order.
`colmap_rust::scene::scene_clustering::SceneClustering::create` models the order of COLMAP's
default backend, which CMake picks when Boost >= 1.84 is found (`cmake/FindDependencies.cmake`):
`boost::unordered_node_map` with the identity `std::hash<uint64_t>`. Boost marks that hash
non-avalanching (`hash_traits.hpp`), so its table mixes it with `mulx(h) = lo ^ hi` of
`h * 0x9E3779B97F4A7C15` (`detail/mulx.hpp`, ported under BSL-1.0, `THIRD_PARTY_NOTICES.md`),
puts the key in group `mix >> size_index` and iterates groups in ascending order. The port
orders the pairs by that mixed value (then by pair id) instead of the correspondence graph's
order, which is ascending pair id. This is Boost's order up to the order of the keys that share
a group (Boost keeps insertion order within a group and moves overflow to later groups), which
depends on the table size and the insertion history. Same as colmap-sharp entry 120.

**Why.** The edge order decides every weight tie: the vertex numbering of the graph cut
(entry 46) and which overlap images each child cluster gets (entry 284). COLMAP relies, without
saying so, on that order being unrelated to image ids. In ascending order every tie breaks
toward the same few low image ids, and a database numbers the images of one rig frame
consecutively, so sibling clusters can overlap in a single frame. A zero-baseline rig's images
of one frame share a projection center, so the two halves have too few distinct centers in
common to estimate the similarity that merges them (colmap-sharp's hierarchical pipeline test
on a panoramic rig ended with two reconstructions until this order). Modelling the default
Boost backend reproduces the property, and most of the order, and stays deterministic.

**Evidence.** Tier C. `tests/scene/rust_only_scene_clustering.rs`
`rust_only_create_spreads_tied_overlap_across_frames` gets at least three shared frames with
the Boost order and fails with the ascending order (two shared frames, 1 and 11). The ported
`scene_clustering_test.cc` cases call `partition` directly and are unaffected.

## 295. The covariant detector rejects an image too small for its octaves

**What differs.** `VlCovDet::put_image` (`feature/vlfeat/covdet.rs`) returns `Err` for an image
whose shorter side is below 16 pixels with the default first octave (-1), or too small for the
configured first octave; VLFeat's `vl_covdet_put_image` goes on: a 1-pixel side converts
`log2(0) = -inf` to an integer, and with a last octave below 0 `vl_scalespace_put_image`
upsamples octave -1 from an octave 0 the scale space never allocated (`scalespace.rs`'s
`put_image` checks this). It also returns `Err` for an image buffer of the wrong length, which
VLFeat cannot check.

**Why.** VLFeat's behaviour there is undefined (out-of-bounds writes), so there is no result to
match; an error the caller can report is the safe equivalent. COLMAP's covariant extractor
never gets there: it rejects such images first (colmap-sharp's `CovariantSift.cs` throws for a
side below `MinimumImageSide`), so no COLMAP result is lost.

**Evidence.** `tests/feature/rust_only_vl_covdet.rs` `rust_only_vl_covdet_rejects_bad_input`.

## 300. The small dense kernels use Ceres' naive loops for every shape

**What differs.** Ceres' `small_blas.h` products (`MatrixMatrixMultiply`,
`MatrixTransposeMatrixMultiply`, `MatrixVectorMultiply`, `MatrixTransposeVectorMultiply`) use
Eigen's fixed-size kernels when all sizes are compile-time constants (the template
specializations the Schur eliminator picks from the Jacobian's static block sizes, for COLMAP's
bundle adjustment typically `<2, 3, 6>`) and the "Naive" loops otherwise.
`colmap_rust::solver::small_blas` always computes each output element as a sum over k in
increasing order into a temporary that starts at 0.0, then stores it with the operation
(=, +=, -=): the naive loops' rounding. The bundle adjustment shapes (2 or 3 inner rows,
inner dimension 3) go through `small_blas_fixed.rs`, whose const-generic kernels unroll the k
loop but keep that order, so every shape rounds exactly as the naive loops do. Eigen's
fixed-size products may sum in another order (vectorized pairs), so results agree with Ceres'
to rounding, not bit for bit. Same as colmap-sharp entry 35 (item 2, the kernel part).

**Why.** Eigen is replaced rather than ported (CLAUDE.md contract 2). One summation order for
every shape makes the Schur solvers' results independent of which kernel a shape dispatches
to, which the later determinism tests rely on. Bundle adjustment is Tier C.

**Evidence.** `tests/solver/rust_only_small_blas.rs` (Ceres' `small_blas_test.cc` via
colmap-sharp: every product, operation, stride and start offset against MatrixXd within
5 * epsilon) and `tests/solver/rust_only_small_blas_fixed.rs` (the fixed kernels give the
naive loops' bits for every operation, with negative zeros and strided, offset output blocks;
reversing the fixed kernels' k order makes it fail).

## 301. Inverting a PSD block that is not positive definite gives NaN

**What differs.** Ceres' `InvertPSDMatrix` (assume_full_rank = true) is
`selfadjointView<Upper>().llt().solve(Identity)`; for a block that is not positive definite,
Eigen's LLT stops and the solve returns unspecified values. `small_blas::invert_upper_psd`
(and `_in_place`, which `BlockRandomAccessDiagonalMatrix::invert` uses for the SCHUR_JACOBI
preconditioner, and which the Schur eliminator will use for E'E) fills the whole inverse with
NaN as soon as a Cholesky pivot is not positive or is NaN. Same as colmap-sharp entry 35
(item 5).

**Why.** Eigen's partial result cannot be reproduced without porting Eigen, and it is garbage
either way. NaN makes the linear solve, and so the trust-region step, invalid; the minimizer
then rejects the step and shrinks the radius, which is what Ceres' minimizer does when the
garbage happens to be non-finite.

**Evidence.** `tests/solver/rust_only_sparse_matrix.rs`
`rust_only_invert_upper_psd_matches_llt_and_rejects_indefinite` (an indefinite 2x2 block
gives all NaN; an SPD block with garbage below the diagonal matches `linalg::Llt` within
1e-12, in place bit-identical to out of place). The ported
`BlockRandomAccessDiagonalMatrixTest.Invert` checks the positive definite case at 1e-14.

## 305. The Ceres manifolds and polynomial balancing sum norms left to right

**What differs.** Ceres evaluates the norms in its manifolds through Eigen
(`delta.norm()`, `x.norm()`, the Householder vector's `squaredNorm()` and dot product in
`sphere_manifold_functions.h` / `householder_vector.h`) and the 1-norms of
`BalanceCompanionMatrix` through `lpNorm<1>()`, all of which Eigen may vectorize and reduce in
packet lanes. `colmap_rust::solver::{sphere_manifold, ceres_polynomial}` sum the squares and
absolute values left to right in coefficient order. The quaternion manifolds' norms are
`std::hypot`'s three-argument form, which the port follows exactly (`math::fns::hypot3`). Results
can differ from COLMAP's Ceres in the last bits. Tier C for the manifolds, like the solver that
consumes them; Tier B for the polynomial roots, which already go through `linalg::EigenSolver`.

**Why.** Eigen is MPL-2.0 and not ported (contract rule 2), and its packet order depends on the
target's SIMD width and the compiler, so there is no single order to match. A sequential order
gives the same bits on every target, native and wasm. Same as colmap-sharp entry 115 (its
`Solver/Manifolds.cs` part).

**Evidence.** `tests/solver/rust_only_manifold.rs` (colmap-sharp's `ManifoldTests`, plus the
sphere's poles) checks Plus/Minus round trips to 1e-12 and the Jacobians against finite
differences to 1e-8; `tests/solver/ceres_polynomial.rs` (Ceres' `polynomial_test.cc`) passes at
Ceres' own tolerances.

## 306. A Problem owns its parameter values and names blocks by id

**What differs.** Ceres' `Problem` keys a parameter block by the user's `double*`, reads the
user's memory during evaluation and writes the solution back into it. `colmap_rust::solver::Problem`
copies the values into one arena when a block is added (`add_parameter_block` returns a
`ParameterBlockId`), evaluates and solves on that arena, and the caller reads the result back
with `Problem::parameter_block`. Every other call that takes a `double*` in Ceres takes the id.
Consequences: blocks cannot alias or be re-added with another size, so Ceres' aliasing and
same-pointer checks do not exist; `AddResidualBlock` does not add unknown blocks on the fly (a
residual block names ids the problem handed out, an unknown id is an error);
`GetParameterBlocks` lists blocks in the order they were added, not in address order (the one
COLMAP caller, `GetOtherParams` in `estimators/covariance.cc`, uses that order only to lay out
the factorization; same as colmap-sharp entry 53); messages that print a block's pointer print
its id. Cost functions, losses and manifolds are `Arc`s, so the ownership options
(`Problem::Options::*_ownership`) have nothing to control. Where Ceres CHECK-fails or
`LOG(FATAL)`s (unknown block, duplicate blocks, size mismatch, a manifold of the wrong size),
the call returns the error with Ceres' message. Block removal, `EvaluateResidualBlock` and the
evaluation callback are not ported: COLMAP never calls them.

**Why.** Rust gives a block no stable address to key on without `unsafe`, and a problem that
borrowed the caller's arrays would need every block borrowed mutably for the problem's whole
life, which COLMAP's shared camera and rig blocks rule out. Owning the values costs one copy in
and one copy out per solve (Ceres copies them into its state vector anyway) and keeps the
solver free of lifetimes. Insertion order is deterministic, where address order is an accident
of the allocator.

**Evidence.** `tests/solver/problem.rs` (Ceres' `problem_test.cc`, the cases that apply; the
skipped ones are listed in its header with their reason) and `tests/solver/problem_evaluate.rs`
(`ProblemEvaluateTest`, exact values) pass.

## 307. A ParameterBlockOrdering lists a group's blocks in the order they were added to the problem

**What differs.** Ceres' `ParameterBlockOrdering` (`OrderedGroups<double*>`) keeps each group
in a `std::set<double*>`, so a group's blocks come out in heap-address order, and the Schur
solvers lay out the eliminated blocks (and the reduced camera system's columns) in that order.
`colmap_rust::solver::OrderedGroups` is Ceres' class over `ParameterBlockId`, so a group's
blocks come out in id order: the order they were added to the problem. Groups are still visited
in increasing id order, and every other rule (moving an element, `Reverse`'s numbering, the
counts) is Ceres'. colmap-sharp keeps each group in the order its blocks were added to the
group (its entry 36); id order is the same wherever a caller adds blocks to the problem and to
the ordering in the same order, as COLMAP's `GlobalPositioner` does.

**Why.** Address order is an accident of the allocator, so no run-to-run order exists to match;
the ids make Ceres' own `std::set` rule deterministic. The effect on results is rounding-level
(the order of additions into the reduced system), within the Tier C outcome bar.

**Evidence.** `tests/solver/ordered_groups.rs` (Ceres' `ordered_groups_test.cc`, unchanged
values) passes; none of its cases depends on the order inside a group.
