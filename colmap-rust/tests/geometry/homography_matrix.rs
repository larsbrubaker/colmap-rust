// Port of COLMAP's src/colmap/geometry/homography_matrix_test.cc. Random values come from
// COLMAP's seeded PRNG (math::random_eigen), seeded with 0 like gtest_main.

use colmap_rust::geometry::homography_matrix::{
    decompose_homography_matrix, homography_matrix_from_pose, pose_from_homography_matrix,
};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Matrix3d, Quaterniond, Vector3d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::random_eigen_matrix3d;

use super::{eigen_matrix_near, expect_near, rigid3d_near};

fn max_abs(m: &Matrix3d) -> f64 {
    m.as_slice().iter().fold(0.0, |a, &v| a.max(v.abs()))
}

// Note that the test case values are obtained from OpenCV.
// The literals are COLMAP's, verbatim (some carry more digits than a double holds).
#[allow(clippy::excessive_precision)]
#[test]
fn decompose_homography_matrix_nominal() {
    let mut h = Matrix3d::new(
        2.649157564634028,
        4.583875997496426,
        70.694447785121326,
        -1.072756858861583,
        3.533262150437228,
        1513.656999614321649,
        0.001303887589576,
        0.003042206876298,
        1.0,
    );
    h *= 3.0;

    let k = Matrix3d::new(640.0, 0.0, 320.0, 0.0, 640.0, 240.0, 0.0, 0.0, 1.0);

    let (cams2_from_cams1, normals) = decompose_homography_matrix(&h, &k, &k);

    assert_eq!(cams2_from_cams1.len(), 4);
    assert_eq!(normals.len(), 4);

    let ref_rotation = Matrix3d::new(
        0.43307983549125,
        0.545749113549648,
        -0.717356090899523,
        -0.85630229674426,
        0.497582023798831,
        -0.138414255706431,
        0.281404038139784,
        0.67421809131173,
        0.682818960388909,
    );
    let ref_translation = Vector3d::new(1.826751712278038, 1.264718492450820, 0.195080809998819);
    let ref_normal = Vector3d::new(-0.244875830334816, -0.480857890778889, -0.841909446789566);

    const EPS: f64 = 1e-6;
    let ref_solution_exists = (0..4).any(|i| {
        (cams2_from_cams1[i].rotation.to_rotation_matrix() - ref_rotation).norm() < EPS
            && (cams2_from_cams1[i].translation - ref_translation).norm() < EPS
            && (normals[i] - ref_normal).norm() < EPS
    });
    assert!(ref_solution_exists);
}

#[test]
fn decompose_homography_matrix_random() {
    set_prng_seed(0);
    const NUM_ITERS: usize = 100;
    let epsilon = 1e-6;
    let identity = Matrix3d::identity();

    for _ in 0..NUM_ITERS {
        let h = random_eigen_matrix3d();

        if h.determinant().abs() < epsilon {
            continue;
        }

        let (cams2_from_cams1, normals) = decompose_homography_matrix(&h, &identity, &identity);

        assert_eq!(cams2_from_cams1.len(), 4);
        assert_eq!(normals.len(), 4);

        // Test that each candidate rotation is a rotation
        for cam2_from_cam1 in &cams2_from_cams1 {
            let r = cam2_from_cam1.rotation.to_rotation_matrix();
            let orthog_error = r.transpose() * r - identity;

            // Check that the rotation is an orthogonal matrix
            assert!(max_abs(&orthog_error) < epsilon);

            // Check determinant is 1
            expect_near(r.determinant(), 1.0, epsilon);
        }
    }
}

#[test]
fn pose_from_homography_matrix_nominal() {
    let k1 = Matrix3d::identity();
    let k2 = Matrix3d::identity();
    let ref_rotation = Quaterniond::new(1.0, 0.1, 0.2, 0.3).normalized();
    let ref_translation = Vector3d::new(1.0, 0.0, 0.0);
    let ref_normal = Vector3d::new(0.0, 0.0, -1.0);
    let h = homography_matrix_from_pose(
        &k1,
        &k2,
        &ref_rotation.to_rotation_matrix(),
        ref_translation,
        ref_normal,
        1.0,
    )
    .unwrap();

    let rays1 = [
        Vector3d::new(0.1, 0.1, 1.0).normalized(),
        Vector3d::new(0.4, 0.1, 1.0).normalized(),
        Vector3d::new(0.1, 0.4, 1.0).normalized(),
        Vector3d::new(0.4, 0.4, 1.0).normalized(),
        Vector3d::new(0.0, 0.0, 1.0).normalized(),
    ];

    let mut rays2 = Vec::new();
    for &ray1 in &rays1 {
        let ray2 = h * ray1;
        assert!(ray2.z > 0.0);
        rays2.push(ray2.normalized());
    }

    let (cam2_from_cam1, normal, points3d) =
        pose_from_homography_matrix(&h, &k1, &k2, &rays1, &rays2).unwrap();

    assert!(rigid3d_near(
        &Rigid3d::new(
            cam2_from_cam1.rotation,
            cam2_from_cam1.translation.normalized()
        ),
        &Rigid3d::new(ref_rotation, ref_translation.normalized()),
        1e-6,
        1e-6
    ));

    assert!(eigen_matrix_near(&normal, &ref_normal, 1e-5));
    assert_eq!(points3d.len(), rays1.len());
}

#[test]
fn homography_matrix_from_pose_pure_rotation() {
    let k1 = Matrix3d::identity();
    let k2 = Matrix3d::identity();
    let r = Matrix3d::identity();
    let t = Vector3d::new(0.0, 0.0, 0.0);
    let n = Vector3d::new(-1.0, 0.0, 0.0);
    let d = f64::INFINITY;
    let h = homography_matrix_from_pose(&k1, &k2, &r, t, n, d).unwrap();
    assert_eq!(h, Matrix3d::identity());
}

#[test]
fn homography_matrix_from_pose_planar_scene() {
    let k1 = Matrix3d::identity();
    let k2 = Matrix3d::identity();
    let r = Matrix3d::identity();
    let t = Vector3d::new(1.0, 0.0, 0.0);
    let n = Vector3d::new(-1.0, 0.0, 0.0);
    let d = 1.0;
    let h = homography_matrix_from_pose(&k1, &k2, &r, t, n, d).unwrap();
    let h_ref = Matrix3d::new(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    assert_eq!(h, h_ref);
}
