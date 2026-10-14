// Rust-only test (PoseLib's gp3p has no unit test and COLMAP tests it only through
// GP3PEstimator and RANSAC; port of colmap-sharp's C#-only `Gp3pTests`) of
// colmap_rust::estimators::solvers::poselib::gp3p and re3q3::solve_rotation. The expected
// poses are the output of PoseLib fa7280f's gp3p built with clang++ and Eigen 3.4 in
// colmap-sharp's scratch harness on the same input. PoseLib pre-rotates the problem by a
// std::rand rotation, so its solutions vary with the seed at the 1e-10 level (seeds 1..5 all
// gave these four poses); the port draws that rotation from a fixed seed
// (docs/CPP_DIVERGENCES.md entry 221). Tier B: the solution set is compared
// order-insensitively, since its order depends on the random pre-rotation. The PoseLib values
// are kept digit for digit, hence the excessive_precision allowance.

#![allow(clippy::excessive_precision)]

use colmap_rust::estimators::solvers::poselib::gp3p::gp3p;
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Vector3d};

fn max_abs(m: Matrix3d) -> f64 {
    m.to_column_major()
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

/// Three rays from distinct origins; gp3p must return the same four poses as C++ PoseLib (to
/// 1e-8), one of which is the ground truth.
#[test]
fn rust_only_matches_poselib_solution_set() {
    let r = AngleAxisd::new(0.7, Vector3d::new(0.3, -0.5, 0.8).normalized()).to_rotation_matrix();
    let t = Vector3d::new(0.2, -0.4, 1.5);
    let p = [
        Vector3d::new(0.1, 0.0, 0.0),
        Vector3d::new(-0.2, 0.3, 0.1),
        Vector3d::new(0.0, -0.1, 0.4),
    ];
    let big_x = [
        Vector3d::new(1.0, 0.5, 4.0),
        Vector3d::new(-1.2, 0.3, 5.0),
        Vector3d::new(0.4, -0.9, 3.0),
    ];
    let x: Vec<Vector3d> = (0..3)
        .map(|i| (r * big_x[i] + t - p[i]).normalized())
        .collect();

    let mut output = Vec::new();
    let n = gp3p(&p, &x, &big_x, &mut output);

    // PoseLib output, seed 2 (row-major R, then t).
    let expected = [
        (
            Matrix3d::new(
                0.78643831294203415,
                -0.55660005156953363,
                -0.26778939958422493,
                0.48461296604437926,
                0.82483142522211972,
                -0.2912102215028109,
                0.38296873642447177,
                0.099244660102391982,
                0.91841463640482346,
            ),
            Vector3d::new(
                0.20000000000001467,
                -0.40000000000002667,
                1.5000000000000004,
            ),
        ),
        (
            Matrix3d::new(
                -0.3115249563007132,
                0.89677384268464011,
                -0.31424333991106707,
                -0.6225007385407948,
                -0.44245516960232001,
                -0.64554337840949683,
                -0.71794500633266001,
                -0.0054861615730680824,
                0.69607820675064291,
            ),
            Vector3d::new(1.5938683320549281, 3.9649710278062558, -6.5890832575233063),
        ),
        (
            Matrix3d::new(
                -0.092892833061926972,
                -0.17092584302288621,
                0.98089514105873965,
                -0.97539153337595708,
                -0.18222090593954188,
                -0.12412452641227711,
                0.1999556905522531,
                -0.96828709462921603,
                -0.14979260392383642,
            ),
            Vector3d::new(-3.2590970512765134, 2.1199117032871451, -3.7918864601009084),
        ),
        (
            Matrix3d::new(
                0.058701789199897658,
                0.20584854725259422,
                -0.97682161909876219,
                0.29517353781895167,
                0.93117681433605992,
                0.21396804203990566,
                0.9536386540593742,
                -0.30089220002675948,
                -0.0060992988844521534,
            ),
            Vector3d::new(3.3727273062725383, -2.2987449769690533, 4.9474907494556586),
        ),
    ];

    let mut missing = Vec::new();
    for (er, et) in expected {
        let found = output.iter().any(|pose| {
            let d = (pose.t - et).cwise_abs();
            max_abs(pose.r() - er) < 1e-8 && d.x.max(d.y).max(d.z) < 1e-8
        });
        if !found {
            missing.push(format!("R={er:?} t={et:?}"));
        }
    }

    assert_eq!(n, 4);
    assert_eq!(output.len(), 4);
    assert!(missing.is_empty(), "{missing:#?}");
}
