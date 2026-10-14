// Port of the MatchGuidedSiftFeaturesCPU geometry cases of colmap/feature/sift_test.cc (same
// test names, values and checks) for colmap_rust::feature::sift::guided_matching: the
// essential-matrix path with distorted, spherical and unprojectable keypoints, and the
// shared-focal pairs. Each TEST runs its Test* helper with the CPU matcher factory of
// sift_matching.rs (the same .cc file, split for the 800-line limit). The GPU instantiations
// of these helpers are skipped (SiftGPU, excluded by licence; PORTING_PLAN.md).

use colmap_rust::feature::matcher::{FeatureMatcher, FeatureMatcherImage};
use colmap_rust::feature::types::{FeatureDescriptors, FeatureKeypoint};
use colmap_rust::geometry::essential_matrix::{
    essential_matrix_from_pose, fundamental_from_essential_matrix,
};
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Matrix3d, Quaterniond, Vector2d, Vector3d};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::util::cancellation::CancelToken;

use super::sift_matching::{
    create_cpu_matcher_with_index_cache, create_empty_descriptors,
    create_random_feature_descriptors, create_reversed_descriptors, expect_reversed_inlier_matches,
    kp,
};

type MatcherFactory = fn(&[FeatureMatcherImage<'_>]) -> Box<dyn FeatureMatcher>;

fn image<'a>(
    image_id: u32,
    camera: &'a Camera,
    keypoints: &'a [FeatureKeypoint],
    descriptors: &'a FeatureDescriptors,
) -> FeatureMatcherImage<'a> {
    FeatureMatcherImage {
        image_id,
        camera: Some(camera),
        keypoints: Some(keypoints),
        descriptors: Some(descriptors),
        pose_prior: None,
    }
}

/// `Eigen::Vector2f` of a projected point, as a keypoint.
fn kp_of(point: [f32; 2]) -> FeatureKeypoint {
    kp(point[0], point[1])
}

/// `camera.ImgFromCam(point).value().cast<float>()`.
fn img_from_cam_f(camera: &Camera, point: Vector3d) -> [f32; 2] {
    let p = camera.img_from_cam(point, true).unwrap();
    [p.x as f32, p.y as f32]
}

fn match_guided(
    matcher: &mut dyn FeatureMatcher,
    max_error: f64,
    image1: &FeatureMatcherImage<'_>,
    image2: &FeatureMatcherImage<'_>,
    two_view_geometry: &mut TwoViewGeometry,
) {
    matcher
        .match_guided(
            max_error,
            image1,
            image2,
            two_view_geometry,
            &CancelToken::new(),
        )
        .unwrap();
}

fn x_translation() -> Rigid3d {
    Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0))
}

/// The OPENCV camera with strong radial and tangential distortion of sift_test.cc.
fn distorted_opencv(focal_length: f64) -> Camera {
    let mut camera =
        Camera::create_from_model_id(1, CameraModelId::OpenCV, focal_length, 100, 200).unwrap();
    camera.params[4] = -0.5; // k1
    camera.params[5] = 0.5; // k2
    camera.params[6] = -0.5; // p1
    camera
}

/// `TestGuidedMatchingWithCameraDistortion`.
fn test_guided_matching_with_camera_distortion(matcher_factory: MatcherFactory) {
    // Test guided matching with essential matrix using calibrated cameras. This exercises the
    // code path that uses normalized coordinates. Use the OPENCV model with strong radial and
    // tangential distortion. Its params are fx, fy, cx, cy, k1, k2, p1, p2. The distortion is
    // strong enough that the pixel-coordinate fundamental matrix finds no matches, but must
    // stay invertible over the keypoints used below, which is what bounds how large these
    // coefficients can be; p2 is left at zero for that reason.
    let camera = distorted_opencv(100.0);

    // Two points on the epipolar line (v=0 in normalized coordinates).
    let img_point11 = img_from_cam_f(&camera, Vector3d::new(-0.5, 0.1, 1.0));
    let img_point12 = img_from_cam_f(&camera, Vector3d::new(0.4, -0.1, 1.0));
    let img_point21 = img_from_cam_f(&camera, Vector3d::new(0.3, -0.1, 1.0));
    let img_point22 = img_from_cam_f(&camera, Vector3d::new(-0.4, 0.1, 1.0));

    let descriptors0 = create_empty_descriptors();
    let descriptors1 = create_random_feature_descriptors(2);
    let descriptors2 = create_reversed_descriptors(&descriptors1);
    let keypoints0: [FeatureKeypoint; 0] = [];
    let keypoints1 = [kp_of(img_point11), kp_of(img_point12)];
    let keypoints2 = [kp_of(img_point21), kp_of(img_point22)];
    let image0 = image(0, &camera, &keypoints0, &descriptors0);
    let image1 = image(1, &camera, &keypoints1, &descriptors1);
    let image2 = image(2, &camera, &keypoints2, &descriptors2);

    let mut matcher = matcher_factory(&[image0, image1, image2]);

    let mut two_view_geometry = TwoViewGeometry::default();
    let e = essential_matrix_from_pose(&x_translation());
    let k = camera.calibration_matrix().unwrap();
    two_view_geometry.e = Some(e);
    two_view_geometry.f = Some(fundamental_from_essential_matrix(&k, &e, &k));

    const MAX_ERROR: f64 = 1.0;

    // With uncalibrated cameras, the fundamental matrix is used with pixel coordinates and no
    // matches are expected to be found due to strong distortion.
    two_view_geometry.config = ConfigurationType::Uncalibrated;
    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image1,
        &image2,
        &mut two_view_geometry,
    );
    assert_eq!(two_view_geometry.inlier_matches.len(), 0);

    // With calibrated cameras, the essential matrix is used with normalized coordinates and
    // matches are expected to be found.
    two_view_geometry.config = ConfigurationType::Calibrated;
    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image1,
        &image2,
        &mut two_view_geometry,
    );
    expect_reversed_inlier_matches(&two_view_geometry);

    two_view_geometry.config = ConfigurationType::Calibrated;
    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image0,
        &image2,
        &mut two_view_geometry,
    );
    assert_eq!(two_view_geometry.inlier_matches.len(), 0);
}

fn equirectangular_camera() -> Camera {
    Camera::create_from_model_id(1, CameraModelId::Equirectangular, 0.0, 512, 256).unwrap()
}

/// `TestGuidedMatchingSpherical`: guided matching for a spherical camera, with
/// correspondences deliberately in the back hemisphere. Those pixels have no normalized image
/// plane representation at all - CamFromImg fails for them - so they are only matchable via
/// the full-sphere bearing.
fn test_guided_matching_spherical(matcher_factory: MatcherFactory) {
    let camera = equirectangular_camera();
    let cam2_from_cam1 = x_translation();

    // Both points are behind both cameras, i.e. in the back hemisphere.
    let point3d1 = Vector3d::new(0.3, 0.1, -2.0);
    let point3d2 = Vector3d::new(-0.25, -0.15, -2.5);

    let project = |point3d: Vector3d| {
        let image_point = camera.img_from_cam(point3d, true).unwrap();
        // The premise of this test: these pixels are unprojectable through the normalized
        // image plane. If this ever starts failing, the test is no longer exercising the back
        // hemisphere.
        assert!(camera.cam_from_img(image_point).is_none());
        [image_point.x as f32, image_point.y as f32]
    };

    let img_point11 = project(point3d1);
    let img_point12 = project(point3d2);
    let img_point21 = project(cam2_from_cam1 * point3d2);
    let img_point22 = project(cam2_from_cam1 * point3d1);

    let descriptors1 = create_random_feature_descriptors(2);
    let descriptors2 = create_reversed_descriptors(&descriptors1);
    let keypoints1 = [kp_of(img_point11), kp_of(img_point12)];
    let keypoints2 = [kp_of(img_point21), kp_of(img_point22)];
    let image1 = image(1, &camera, &keypoints1, &descriptors1);
    let image2 = image(2, &camera, &keypoints2, &descriptors2);

    // Same as image2, but with the second correspondence replaced by a decoy far off the
    // epipolar great circle, which must be rejected.
    //
    // This is the load-bearing assertion. Recovering the matches above is necessary but not
    // sufficient: when every keypoint is unprojectable, the old sentinel mapped them all to
    // the same location, so the guided filter scored every pair identically and degenerated
    // into accepting everything - and plain descriptor matching then produced the right
    // answer anyway. Only a decoy that the filter must actively reject distinguishes "the
    // epipolar constraint is evaluated correctly for back-hemisphere rays" from "the
    // constraint has stopped constraining anything".
    let img_point_decoy = project(cam2_from_cam1 * Vector3d::new(2.0, -1.5, -0.5));
    let keypoints3 = [kp_of(img_point21), kp_of(img_point_decoy)];
    let image3 = image(3, &camera, &keypoints3, &descriptors2);

    let mut matcher = matcher_factory(&[image1, image2, image3]);

    let mut two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        e: Some(essential_matrix_from_pose(&cam2_from_cam1)),
        ..TwoViewGeometry::default()
    };

    const MAX_ERROR: f64 = 4.0;

    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image1,
        &image2,
        &mut two_view_geometry,
    );
    expect_reversed_inlier_matches(&two_view_geometry);

    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image1,
        &image3,
        &mut two_view_geometry,
    );
    assert_eq!(two_view_geometry.inlier_matches.len(), 1);
    assert_eq!(two_view_geometry.inlier_matches[0].point2d_idx1, 1);
    assert_eq!(two_view_geometry.inlier_matches[0].point2d_idx2, 0);
}

/// `TestGuidedMatchingSphericalMixedHemispheres`: one correspondence in the front hemisphere
/// and one in the back, to verify the two are handled by the same code path rather than being
/// swapped.
fn test_guided_matching_spherical_mixed_hemispheres(matcher_factory: MatcherFactory) {
    let camera = equirectangular_camera();
    let cam2_from_cam1 = x_translation();

    let point3d_front = Vector3d::new(0.2, 0.1, 2.0);
    let point3d_back = Vector3d::new(-0.25, -0.15, -2.5);

    let project = |point3d: Vector3d| img_from_cam_f(&camera, point3d);

    let img_point11 = project(point3d_front);
    let img_point12 = project(point3d_back);
    let img_point21 = project(cam2_from_cam1 * point3d_back);
    let img_point22 = project(cam2_from_cam1 * point3d_front);

    let as_d = |p: [f32; 2]| Vector2d::new(f64::from(p[0]), f64::from(p[1]));
    assert!(camera.cam_from_img(as_d(img_point11)).is_some());
    assert!(camera.cam_from_img(as_d(img_point12)).is_none());

    let descriptors1 = create_random_feature_descriptors(2);
    let descriptors2 = create_reversed_descriptors(&descriptors1);
    let keypoints1 = [kp_of(img_point11), kp_of(img_point12)];
    let keypoints2 = [kp_of(img_point21), kp_of(img_point22)];
    let image1 = image(1, &camera, &keypoints1, &descriptors1);
    let image2 = image(2, &camera, &keypoints2, &descriptors2);

    // Replaces the back-hemisphere correspondence with a decoy off the epipolar great circle,
    // so that the filter has to actively reject it. See the comment in
    // test_guided_matching_spherical for why this is what discriminates.
    let img_point_decoy = project(cam2_from_cam1 * Vector3d::new(2.0, -1.5, -0.5));
    let keypoints3 = [kp_of(img_point_decoy), kp_of(img_point22)];
    let image3 = image(3, &camera, &keypoints3, &descriptors2);

    let mut matcher = matcher_factory(&[image1, image2, image3]);

    let mut two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        e: Some(essential_matrix_from_pose(&cam2_from_cam1)),
        ..TwoViewGeometry::default()
    };

    const MAX_ERROR: f64 = 4.0;

    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image1,
        &image2,
        &mut two_view_geometry,
    );
    expect_reversed_inlier_matches(&two_view_geometry);

    // Only the front-hemisphere correspondence survives.
    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image1,
        &image3,
        &mut two_view_geometry,
    );
    assert_eq!(two_view_geometry.inlier_matches.len(), 1);
    assert_eq!(two_view_geometry.inlier_matches[0].point2d_idx1, 0);
    assert_eq!(two_view_geometry.inlier_matches[0].point2d_idx2, 1);
}

/// `TestGuidedMatchingUnprojectableKeypoints`: a keypoint the camera cannot unproject must be
/// rejected outright. It used to be relocated to a (1e6, 1e6) sentinel, which does not reject:
/// the Sampson error is a ratio whose numerator and denominator scale together, so the
/// residual converges to the finite squared distance from the partner to the epipolar line of
/// the point at infinity in direction (1, 1, 0). Any partner near that line was therefore
/// silently accepted.
fn test_guided_matching_unprojectable_keypoints(matcher_factory: MatcherFactory) {
    let camera = distorted_opencv(100.0);

    // Well inside a region where the iterative undistortion does not converge.
    let unprojectable = Vector2d::new(50.0, 150.0);
    assert!(camera.cam_from_img(unprojectable).is_none());

    let project = |point3d: Vector3d| {
        let image_point = camera.img_from_cam(point3d, true).unwrap();
        // Everything except the sentinel keypoint must be a normal, usable keypoint, or the
        // test would pass for the wrong reason.
        assert!(camera.cam_from_img(image_point).is_some());
        [image_point.x as f32, image_point.y as f32]
    };

    // A translation with tx == ty, so that the epipolar line of the sentinel direction
    // (1, 1, 0) passes through the image center and the decoy below can sit on it at a
    // well-behaved location.
    let cam2_from_cam1 = Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 1.0, 1.0));
    let e: Matrix3d = essential_matrix_from_pose(&cam2_from_cam1);

    let point3d = Vector3d::new(-0.3, -0.2, 2.0);
    let img_point_good1 = project(point3d);
    let img_point_good2 = project(cam2_from_cam1 * point3d);

    // The old (1e6, 1e6) sentinel converges to the direction (1, 1, 0); its epipolar line is
    // where spurious matches used to concentrate, so the decoy is placed exactly on it.
    let sentinel_line = e * Vector3d::new(1.0, 1.0, 0.0);
    let decoy_x = 0.2;
    let decoy_y = -(sentinel_line.x * decoy_x + sentinel_line.z) / sentinel_line.y;
    let img_point_decoy = project(Vector3d::new(decoy_x, decoy_y, 1.0));

    let descriptors1 = create_random_feature_descriptors(2);
    let descriptors2 = create_reversed_descriptors(&descriptors1);
    let keypoints1 = [
        kp(unprojectable.x as f32, unprojectable.y as f32),
        kp_of(img_point_good1),
    ];
    let keypoints2 = [kp_of(img_point_good2), kp_of(img_point_decoy)];
    let image1 = image(1, &camera, &keypoints1, &descriptors1);
    let image2 = image(2, &camera, &keypoints2, &descriptors2);

    let mut matcher = matcher_factory(&[image1, image2]);

    let mut two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        e: Some(e),
        ..TwoViewGeometry::default()
    };

    match_guided(&mut *matcher, 1.0, &image1, &image2, &mut two_view_geometry);

    // Only the good pair survives; the unprojectable keypoint 0 matches nothing.
    assert_eq!(two_view_geometry.inlier_matches.len(), 1);
    assert_eq!(two_view_geometry.inlier_matches[0].point2d_idx1, 1);
    assert_eq!(two_view_geometry.inlier_matches[0].point2d_idx2, 0);
}

#[test]
fn match_guided_sift_features_cpu_essential_matrix() {
    test_guided_matching_with_camera_distortion(create_cpu_matcher_with_index_cache);
}

#[test]
fn match_guided_sift_features_cpu_spherical() {
    test_guided_matching_spherical(create_cpu_matcher_with_index_cache);
}

#[test]
fn match_guided_sift_features_cpu_spherical_mixed_hemispheres() {
    test_guided_matching_spherical_mixed_hemispheres(create_cpu_matcher_with_index_cache);
}

#[test]
fn match_guided_sift_features_cpu_unprojectable_keypoints() {
    test_guided_matching_unprojectable_keypoints(create_cpu_matcher_with_index_cache);
}

/// `TestGuidedMatchingSharedFocal`: an UNCALIBRATED pair carrying solver-estimated intrinsics
/// (camera1/camera2) is guided-matched via the essential matrix in normalized coordinates,
/// using those estimated intrinsics rather than the images' cameras, whose focal length is
/// only a placeholder. Distortion is strong enough that the pixel-coordinate F path finds
/// nothing, and the placeholder focal is wrong enough that normalizing with it finds nothing
/// either, so the test passes only if the estimated camera is the one used.
///
/// As elsewhere on the E path, E is taken to relate undistorted rays.
fn test_guided_matching_shared_focal(matcher_factory: MatcherFactory) {
    const ESTIMATED_FOCAL: f64 = 100.0;
    const PLACEHOLDER_FOCAL: f64 = 500.0;
    // Same distortion as in test_guided_matching_with_camera_distortion: strong, but
    // invertible over the keypoints used below.
    let camera = distorted_opencv(ESTIMATED_FOCAL);

    // The camera as stored in the database: same model and distortion, but the focal length
    // has not been recovered yet.
    let mut placeholder_camera = camera.clone();
    placeholder_camera
        .set_focal_length(PLACEHOLDER_FOCAL)
        .unwrap();

    // Two points on the epipolar line (v=0 in normalized coordinates).
    let img_point11 = img_from_cam_f(&camera, Vector3d::new(-0.5, 0.1, 1.0));
    let img_point12 = img_from_cam_f(&camera, Vector3d::new(0.4, -0.1, 1.0));
    let img_point21 = img_from_cam_f(&camera, Vector3d::new(0.3, -0.1, 1.0));
    let img_point22 = img_from_cam_f(&camera, Vector3d::new(-0.4, 0.1, 1.0));

    let descriptors1 = create_random_feature_descriptors(2);
    let descriptors2 = create_reversed_descriptors(&descriptors1);
    let keypoints1 = [kp_of(img_point11), kp_of(img_point12)];
    let keypoints2 = [kp_of(img_point21), kp_of(img_point22)];
    let image1 = image(1, &placeholder_camera, &keypoints1, &descriptors1);
    let image2 = image(2, &placeholder_camera, &keypoints2, &descriptors2);

    let mut matcher = matcher_factory(&[image1, image2]);

    let e = essential_matrix_from_pose(&x_translation());
    let k = camera.calibration_matrix().unwrap();
    let mut two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Uncalibrated,
        e: Some(e),
        camera1: Some(camera.clone()),
        camera2: Some(camera.clone()),
        // F = K^-T E K^-1, as the estimator populates it for this config.
        f: Some(fundamental_from_essential_matrix(&k, &e, &k)),
        ..TwoViewGeometry::default()
    };

    const MAX_ERROR: f64 = 1.0;

    // Matches are found only by normalizing with the estimated intrinsics. The complementary
    // case, an UNCALIBRATED pair without them falling back to the F path, is covered by
    // test_guided_matching_with_camera_distortion.
    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image1,
        &image2,
        &mut two_view_geometry,
    );
    expect_reversed_inlier_matches(&two_view_geometry);
}

/// `TestGuidedMatchingSharedFocalPerPairFocal`: the normalizing camera is not a function of
/// the image alone: a shared-focal pair carries a focal length estimated per pair, so the same
/// image matched against different partners must be renormalized. Guards the GPU matcher's
/// feature-location cache, which keys on the image id.
fn test_guided_matching_shared_focal_per_pair_focal(matcher_factory: MatcherFactory) {
    const FOCAL_A: f64 = 100.0;
    const FOCAL_B: f64 = 200.0;
    let camera_a =
        Camera::create_from_model_id(1, CameraModelId::SimplePinhole, FOCAL_A, 100, 200).unwrap();
    let camera_b =
        Camera::create_from_model_id(2, CameraModelId::SimplePinhole, FOCAL_B, 100, 200).unwrap();

    // image1's pixels normalize to y = +-0.1 under camera_a, and to half that, y = +-0.05,
    // under camera_b. The relative pose is a pure x-translation, so a match requires the
    // partner's normalized y to agree.
    let img_point11 = img_from_cam_f(&camera_a, Vector3d::new(-0.5, 0.1, 1.0));
    let img_point12 = img_from_cam_f(&camera_a, Vector3d::new(0.4, -0.1, 1.0));
    // Partner for the camera_a pair.
    let img_point21 = img_from_cam_f(&camera_a, Vector3d::new(0.3, -0.1, 1.0));
    let img_point22 = img_from_cam_f(&camera_a, Vector3d::new(-0.4, 0.1, 1.0));
    // Partner for the camera_b pair.
    let img_point31 = img_from_cam_f(&camera_b, Vector3d::new(0.3, -0.05, 1.0));
    let img_point32 = img_from_cam_f(&camera_b, Vector3d::new(-0.4, 0.05, 1.0));

    let descriptors1 = create_random_feature_descriptors(2);
    let descriptors2 = create_reversed_descriptors(&descriptors1);
    let descriptors3 = create_reversed_descriptors(&descriptors1);
    let keypoints1 = [kp_of(img_point11), kp_of(img_point12)];
    let keypoints2 = [kp_of(img_point21), kp_of(img_point22)];
    let keypoints3 = [kp_of(img_point31), kp_of(img_point32)];
    let image1 = image(1, &camera_a, &keypoints1, &descriptors1);
    let image2 = image(2, &camera_a, &keypoints2, &descriptors2);
    let image3 = image(3, &camera_a, &keypoints3, &descriptors3);

    let mut matcher = matcher_factory(&[image1, image2, image3]);

    let mut two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Uncalibrated,
        e: Some(essential_matrix_from_pose(&x_translation())),
        ..TwoViewGeometry::default()
    };

    const MAX_ERROR: f64 = 1.0;

    two_view_geometry.camera1 = Some(camera_a.clone());
    two_view_geometry.camera2 = Some(camera_a.clone());
    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image1,
        &image2,
        &mut two_view_geometry,
    );
    expect_reversed_inlier_matches(&two_view_geometry);

    // Same image1, different estimated focal: stale normalized keypoints from the previous
    // call would put image1 at y = +-0.1 instead of +-0.05, far outside the ~1/f normalized
    // threshold.
    two_view_geometry.camera1 = Some(camera_b.clone());
    two_view_geometry.camera2 = Some(camera_b.clone());
    match_guided(
        &mut *matcher,
        MAX_ERROR,
        &image1,
        &image3,
        &mut two_view_geometry,
    );
    expect_reversed_inlier_matches(&two_view_geometry);
}

#[test]
fn match_guided_sift_features_cpu_shared_focal() {
    test_guided_matching_shared_focal(create_cpu_matcher_with_index_cache);
}

#[test]
fn match_guided_sift_features_cpu_shared_focal_per_pair_focal() {
    test_guided_matching_shared_focal_per_pair_focal(create_cpu_matcher_with_index_cache);
}
