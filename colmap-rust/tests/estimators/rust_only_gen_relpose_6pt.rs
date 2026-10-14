// Rust-only tests (COLMAP tests PoseLib's gen_relpose_6pt only through GR6PEstimator, not yet
// ported) of colmap_rust::estimators::solvers::poselib::gen_relpose_6pt and the quaternion
// step it refines with. Tier B: on random minimal problems between two generalized cameras
// (rays with distinct origins), the true metric pose is among the solutions and the output
// vector is cleared first. (Cheirality filters the solutions before PoseLib's Newton
// refinement, so a refined spurious solution need not stay cheiral.)

use colmap_rust::estimators::solvers::poselib::camera_pose::{
    quat_exp, quat_multiply, quat_step_pre, quat_to_rotmat,
};
use colmap_rust::estimators::solvers::poselib::essential::check_cheirality_generalized;
use colmap_rust::estimators::solvers::poselib::gen_relpose_6pt::gen_relpose_6pt;
use colmap_rust::estimators::solvers::poselib::CameraPose;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Vector3d, Vector4d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::random_eigen_vector3d;

/// Pose tolerance. Well-conditioned problems agree to 1e-12 after the Newton refinement;
/// COLMAP's GR6PEstimator test allows 1e-2.
const TOL: f64 = 1e-6;

struct Problem {
    p1: Vec<Vector3d>,
    x1: Vec<Vector3d>,
    p2: Vec<Vector3d>,
    x2: Vec<Vector3d>,
}

/// Six correspondences between two rigs whose rays start at random origins in [-1, 1]^3.
fn random_problem(rig2_from_rig1: &Rigid3d) -> Problem {
    let mut problem = Problem {
        p1: Vec::new(),
        x1: Vec::new(),
        p2: Vec::new(),
        x2: Vec::new(),
    };
    for _ in 0..6 {
        let point1 = Vector3d::new(
            random_uniform_real(-2.0, 2.0),
            random_uniform_real(-2.0, 2.0),
            random_uniform_real(4.0, 8.0),
        );
        let point2 = *rig2_from_rig1 * point1;
        let origin1 = random_eigen_vector3d();
        let origin2 = random_eigen_vector3d();
        problem.p1.push(origin1);
        problem.x1.push((point1 - origin1).normalized());
        problem.p2.push(origin2);
        problem.x2.push((point2 - origin2).normalized());
    }
    problem
}

#[test]
fn rust_only_gen_relpose_6pt_recovers_true_pose() {
    set_prng_seed(6);
    let trials = 50;
    let mut found_count = 0;
    for _ in 0..trials {
        let axis = random_eigen_vector3d().normalized();
        let rotation = AngleAxisd::new(random_uniform_real(-0.5, 0.5), axis).to_quaternion();
        let rig2_from_rig1 = Rigid3d::new(rotation, random_eigen_vector3d());
        let pr = random_problem(&rig2_from_rig1);

        let mut output = vec![CameraPose::identity()];
        let n = gen_relpose_6pt(&pr.p1, &pr.x1, &pr.p2, &pr.x2, &mut output);
        assert_eq!(n, output.len(), "the output is cleared first");
        assert!(n <= 64);

        let r_true = rig2_from_rig1.rotation.to_rotation_matrix();
        let mut found = false;
        for pose in &output {
            assert!((pose.q.norm() - 1.0).abs() < 1e-10);
            let r_error = (pose.r() - r_true).norm();
            let t_error = (pose.t - rig2_from_rig1.translation).norm();
            found |= r_error < TOL && t_error < TOL;
        }
        if found {
            found_count += 1;
        }
    }
    assert_eq!(found_count, trials);
}

#[test]
fn rust_only_check_cheirality_generalized() {
    // Rig 2 is rig 1 moved one unit along -x; the rays start one unit apart along y.
    let pose = CameraPose::new(
        Vector4d::new(1.0, 0.0, 0.0, 0.0),
        Vector3d::new(1.0, 0.0, 0.0),
    );
    let point1 = Vector3d::new(0.0, 0.0, 5.0);
    let point2 = point1 + pose.t;
    let (p1, p2) = (Vector3d::new(0.0, 1.0, 0.0), Vector3d::new(0.0, -1.0, 0.0));
    let x1 = (point1 - p1).normalized();
    let x2 = (point2 - p2).normalized();
    assert!(check_cheirality_generalized(&pose, p1, x1, p2, x2, 0.0));
    assert!(!check_cheirality_generalized(&pose, p1, -x1, p2, -x2, 0.0));
    // Both rays reach the point after sqrt(26) ~ 5.1 units.
    assert!(check_cheirality_generalized(&pose, p1, x1, p2, x2, 5.0));
    assert!(!check_cheirality_generalized(&pose, p1, x1, p2, x2, 5.2));
}

#[test]
fn rust_only_quat_step_pre() {
    set_prng_seed(7);
    for scale in [1.0, 1e-7] {
        let q = Vector4d::new(
            random_uniform_real(-1.0, 1.0),
            random_uniform_real(-1.0, 1.0),
            random_uniform_real(-1.0, 1.0),
            random_uniform_real(-1.0, 1.0),
        )
        .normalized();
        // Both branches of quat_exp (above and below |w| = 1e-6) are unit rotations of |w|.
        let w = random_eigen_vector3d() * scale;
        let e = quat_exp(w);
        assert!((e.norm() - 1.0).abs() < 1e-15);
        let expected = AngleAxisd::new(w.norm(), w.normalized()).to_rotation_matrix();
        assert!((quat_to_rotmat(e) - expected).norm() < 1e-14);

        let stepped = quat_step_pre(q, w);
        assert_eq!(stepped, quat_multiply(e, q));
        let expected = expected * quat_to_rotmat(q);
        assert!((quat_to_rotmat(stepped) - expected).norm() < 1e-14);
    }
}
