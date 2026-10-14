// Port of colmap/scene/two_view_geometry_test.cc, 1:1 (same test names, values and checks),
// for colmap_rust::scene::two_view_geometry::TwoViewGeometry. COLMAP's
// `EigenMatrixNear(rhs)` matcher (default tolerance, Eigen's `dummy_precision()` 1e-12) is
// `eigen_matrix_near` over the coefficients.

use colmap_rust::feature::types::FeatureMatch;
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Matrix3d, Quaterniond, Vector3d};
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Port of COLMAP's `EigenMatrixNear(rhs)` matcher with Eigen's default precision (1e-12):
/// when `rhs.isZero()`, `lhs.norm() <= tol`; otherwise Eigen's `lhs.isApprox(rhs, tol)`,
/// `||lhs - rhs|| <= tol * min(||lhs||, ||rhs||)`.
fn eigen_matrix_near(lhs: &[f64], rhs: &[f64]) -> bool {
    const TOL: f64 = 1e-12;
    assert_eq!(lhs.len(), rhs.len());
    if rhs.iter().all(|&v| v.abs() <= TOL) {
        return l2(lhs) <= TOL;
    }
    let diff: Vec<f64> = lhs.iter().zip(rhs).map(|(a, b)| a - b).collect();
    l2(&diff) <= TOL * l2(lhs).min(l2(rhs))
}

fn expect_matrix_identity(matrix: Option<Matrix3d>) {
    let matrix = matrix.expect("matrix has a value");
    assert!(eigen_matrix_near(
        matrix.as_slice(),
        Matrix3d::identity().as_slice()
    ));
}

fn expect_state(two_view_geometry: &TwoViewGeometry, translation: [f64; 3], matches: [u32; 4]) {
    assert_eq!(two_view_geometry.config, ConfigurationType::Calibrated);
    expect_matrix_identity(two_view_geometry.f);
    expect_matrix_identity(two_view_geometry.e);
    expect_matrix_identity(two_view_geometry.h);
    assert!(two_view_geometry.cam2_from_cam1.is_some());
    let cam2_from_cam1 = two_view_geometry.cam2_from_cam1.unwrap();
    assert!(eigen_matrix_near(
        &cam2_from_cam1.rotation.coeffs().to_array(),
        &Quaterniond::identity().coeffs().to_array()
    ));
    assert!(eigen_matrix_near(
        &cam2_from_cam1.translation.to_array(),
        &translation
    ));
    assert_eq!(two_view_geometry.inlier_matches[0].point2d_idx1, matches[0]);
    assert_eq!(two_view_geometry.inlier_matches[0].point2d_idx2, matches[1]);
    assert_eq!(two_view_geometry.inlier_matches[1].point2d_idx1, matches[2]);
    assert_eq!(two_view_geometry.inlier_matches[1].point2d_idx2, matches[3]);
}

#[test]
fn two_view_geometry_default() {
    let two_view_geometry = TwoViewGeometry::default();
    assert_eq!(two_view_geometry.config, ConfigurationType::Undefined);
    assert!(two_view_geometry.f.is_none());
    assert!(two_view_geometry.e.is_none());
    assert!(two_view_geometry.h.is_none());
    assert!(two_view_geometry.cam2_from_cam1.is_none());
    assert!(two_view_geometry.inlier_matches.is_empty());
}

#[test]
fn two_view_geometry_invert() {
    let mut two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        f: Some(Matrix3d::identity()),
        e: Some(Matrix3d::identity()),
        h: Some(Matrix3d::identity()),
        cam2_from_cam1: Some(Rigid3d::new(
            Quaterniond::identity(),
            Vector3d::new(0.0, 1.0, 2.0),
        )),
        ..TwoViewGeometry::default()
    };
    two_view_geometry
        .inlier_matches
        .resize(2, FeatureMatch::default());
    two_view_geometry.inlier_matches[0] = FeatureMatch::new(0, 1);
    two_view_geometry.inlier_matches[1] = FeatureMatch::new(2, 3);

    two_view_geometry.invert();
    expect_state(&two_view_geometry, [-0.0, -1.0, -2.0], [1, 0, 3, 2]);

    two_view_geometry.invert();
    expect_state(&two_view_geometry, [0.0, 1.0, 2.0], [0, 1, 2, 3]);
}
