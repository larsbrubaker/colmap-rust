// Rust-only (COLMAP has no test for Eigen itself): Quaterniond and AngleAxisd against known
// rotations and round trips (quaternion <-> matrix <-> angle-axis). Port of colmap-sharp's
// `ColmapSharp.Tests/LinearAlgebra/QuaternionTests.cs`, plus slerp. The bit-level comparison
// against Eigen itself is `rust_only_rotation_oracle.rs`; these pin the conventions a reader
// can check by hand (w-first constructor, xyzw coeffs, Hamilton product order, the sign
// convention of from_rotation_matrix).

use colmap_rust::linalg::{AngleAxisd, Matrix3d, Quaterniond, Vector3d, Vector4d};
use colmap_rust::math::fns;
use std::f64::consts::{FRAC_1_SQRT_2 as S, PI};

fn sample_rotations() -> Vec<Quaterniond> {
    vec![
        Quaterniond::identity(),
        Quaterniond::new(S, S, 0.0, 0.0),
        Quaterniond::new(0.0, 1.0, 0.0, 0.0),
        Quaterniond::new(0.0, 0.0, 1.0, 0.0),
        Quaterniond::new(0.0, 0.0, 0.0, 1.0),
        Quaterniond::new(0.0, S, S, 0.0),
        Quaterniond::new(0.5, -0.5, 0.5, 0.5),
        Quaterniond::new(0.9, 0.1, -0.2, 0.3).normalized(),
        Quaterniond::new(0.05, 0.7, -0.1, 0.7).normalized(),
        Quaterniond::new(0.01, -0.3, 0.9, 0.2).normalized(),
    ]
}

#[test]
fn rust_only_constructor_is_wxyz_coeffs_are_xyzw() {
    let q = Quaterniond::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(q.w, 1.0);
    assert_eq!(q.x, 2.0);
    assert_eq!(q.coeffs(), Vector4d::new(2.0, 3.0, 4.0, 1.0));
    assert_eq!(Quaterniond::from_coeffs(q.coeffs()), q);
    assert_eq!(q.vec(), Vector3d::new(2.0, 3.0, 4.0));
}

#[test]
fn rust_only_ninety_degree_rotations() {
    let about_z = Quaterniond::new(S, 0.0, 0.0, S);
    let about_x = Quaterniond::new(S, S, 0.0, 0.0);
    let rz = Matrix3d::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    assert!((about_z * Vector3d::unit_x()).is_approx(Vector3d::unit_y()));
    assert!((about_x * Vector3d::unit_y()).is_approx(Vector3d::unit_z()));
    assert!(about_z.to_rotation_matrix().is_approx(rz));
    assert!(AngleAxisd::new(PI / 2.0, Vector3d::unit_z())
        .to_rotation_matrix()
        .is_approx(rz));
    // Composition: (a * b) applies b first. Z then X takes x -> y -> z.
    assert!(((about_x * about_z) * Vector3d::unit_x()).is_approx(Vector3d::unit_z()));
    assert!((about_z.angular_distance(Quaterniond::identity()) - PI / 2.0).abs() <= 1e-15);
}

#[test]
fn rust_only_rotating_a_vector_matches_the_rotation_matrix() {
    let v = Vector3d::new(0.3, -1.7, 2.2);
    for q in sample_rotations() {
        assert!(
            (q * v).is_approx_with(q.to_rotation_matrix() * v, 1e-14),
            "{q:?}"
        );
    }
}

#[test]
fn rust_only_quaternion_matrix_round_trip() {
    for q in sample_rotations() {
        let back = Quaterniond::from_rotation_matrix(q.to_rotation_matrix());
        // q and -q are the same rotation; from_rotation_matrix picks the sign by its branch.
        let negated = Quaterniond::new(-q.w, -q.x, -q.y, -q.z);
        assert!(back.is_approx(q) || back.is_approx(negated), "{q:?}");
    }
}

#[test]
fn rust_only_from_rotation_matrix_sign_convention() {
    // Positive trace: w is the positive component.
    let q = Quaterniond::new(-S, 0.0, 0.0, S);
    assert!(Quaterniond::from_rotation_matrix(q.to_rotation_matrix()).w > 0.0);
    // Half turns (trace -1): the component on the largest diagonal entry is positive.
    let half_y = Matrix3d::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
    let half_x = Matrix3d::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
    let half_z = Matrix3d::new(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0);
    assert_eq!(
        Quaterniond::from_rotation_matrix(half_y),
        Quaterniond::new(0.0, 0.0, 1.0, 0.0)
    );
    assert_eq!(
        Quaterniond::from_rotation_matrix(half_x),
        Quaterniond::new(0.0, 1.0, 0.0, 0.0)
    );
    assert_eq!(
        Quaterniond::from_rotation_matrix(half_z),
        Quaterniond::new(0.0, 0.0, 0.0, 1.0)
    );
}

#[test]
fn rust_only_angle_axis_round_trips() {
    for q in sample_rotations() {
        let aa = AngleAxisd::from_quaternion(q);
        let r = q.to_rotation_matrix();
        assert!((0.0..=PI).contains(&aa.angle), "{q:?}");
        assert!(aa.to_rotation_matrix().is_approx_with(r, 1e-14), "{q:?}");
        assert!(
            aa.to_quaternion()
                .to_rotation_matrix()
                .is_approx_with(r, 1e-14),
            "{q:?}"
        );
        let from_matrix = AngleAxisd::from_rotation_matrix(r);
        assert!(
            from_matrix.to_rotation_matrix().is_approx_with(r, 1e-14),
            "{q:?}"
        );
    }
}

#[test]
fn rust_only_angle_axis_negative_w_flips_the_axis() {
    // -q is the same rotation; the angle stays in [0, pi] and the axis flips instead.
    let q = Quaterniond::new(-fns::cos(0.3), fns::sin(0.3), 0.0, 0.0);
    let aa = AngleAxisd::from_quaternion(q);
    assert!((aa.angle - 0.6).abs() <= 1e-15);
    assert_eq!(aa.axis, -Vector3d::unit_x());
    assert_eq!(
        AngleAxisd::from_quaternion(Quaterniond::identity()),
        AngleAxisd::new(0.0, Vector3d::unit_x())
    );
    assert!(aa.is_approx(AngleAxisd::new(0.6, -Vector3d::unit_x())));
}

#[test]
fn rust_only_angle_axis_tiny_vector_part_keeps_its_angle_and_axis() {
    // Tiny (but nonzero) vector parts must not collapse to angle 0 about x. The angles are the
    // pycolmap/Eigen values (rust_only_rotation_oracle pins them bit for bit).
    let tiny = AngleAxisd::from_quaternion(Quaterniond::new(-1.0, 1e-17, -2e-17, 0.0));
    let underflowing = AngleAxisd::from_quaternion(Quaterniond::new(1.0, 1e-200, 0.0, 0.0));
    assert_eq!(tiny.angle, 4.47213595499958e-17);
    // w < 0 flips the axis: -(1, -2, 0) / sqrt(5).
    assert!(tiny
        .axis
        .is_approx_with(Vector3d::new(-1.0, 2.0, 0.0) / fns::sqrt(5.0), 1e-15));
    assert_eq!(underflowing.angle, 2e-200);
    assert_eq!(underflowing.axis, Vector3d::unit_x());
}

#[test]
fn rust_only_normalized_conjugate_inverse() {
    let q = Quaterniond::new(1.0, 2.0, 3.0, 4.0);
    let zero = Quaterniond::new(0.0, 0.0, 0.0, 0.0);
    assert!((q.normalized().norm() - 1.0).abs() <= 1e-15);
    assert_eq!(zero.normalized(), zero);
    assert!((q * q.inverse()).is_approx(Quaterniond::identity()));
    assert_eq!(q.conjugate(), Quaterniond::new(1.0, -2.0, -3.0, -4.0));
    assert_eq!(
        q.inverse(),
        Quaterniond::new(1.0 / 30.0, -2.0 / 30.0, -3.0 / 30.0, -4.0 / 30.0)
    );
    assert_eq!(zero.inverse(), zero);
}

#[test]
fn rust_only_from_two_vectors() {
    // from_two_vectors maps the first direction onto the second along the shortest arc.
    let a = Vector3d::new(1.0, 2.0, 3.0);
    let b = Vector3d::new(-2.0, 0.5, 1.0);
    let q = Quaterniond::from_two_vectors(a, b);
    assert!((q * a.normalized() - b.normalized()).norm() < 1e-15);
    assert!((q.norm() - 1.0).abs() < 1e-15);
    // Shortest arc: the rotation angle is the angle between the vectors.
    let c = a.normalized().dot(b.normalized());
    assert!((q.w - fns::sqrt((1.0 + c) / 2.0)).abs() < 1e-15);
}

#[test]
fn rust_only_from_two_vectors_opposite() {
    // Exactly and nearly opposite vectors take the half-turn branch (1 + c < 1e-8,
    // docs/CPP_DIVERGENCES.md entry 4), which must still map the first direction onto the
    // second to rounding accuracy and return a unit quaternion. Covers every choice of the
    // least-aligned coordinate axis and both sides of the threshold.
    let directions = [
        Vector3d::new(1.0, 0.0, 0.0),
        Vector3d::new(0.0, -1.0, 0.0),
        Vector3d::new(0.0, 0.0, 1.0),
        Vector3d::new(1.0, 2.0, 3.0),
        Vector3d::new(-3.0, 0.5, 0.25),
        Vector3d::new(0.1, -4.0, 2.0),
    ];
    // Perpendicular offsets giving 1 + c = 0, ~5e-13, ~5e-11 and ~5e-9 (inside the branch) and
    // ~2e-8 (just outside it).
    let offsets = [0.0, 1e-6, 1e-5, 1e-4, 2e-4];
    let mut failures = Vec::new();
    for direction in directions {
        let u = direction.normalized();
        let perpendicular = u.cross(Vector3d::new(0.3, -0.7, 0.2)).normalized();
        for offset in offsets {
            let target = -u + offset * perpendicular;
            let q = Quaterniond::from_two_vectors(direction, target);
            let error = (q * u - target.normalized()).norm();
            // Outside the branch Melax's formula (the one COLMAP's Eigen uses there) loses
            // accuracy as 1 + c shrinks, ~1e-8 at 1 + c = 2e-8 (the reason for the branch), so
            // the bound there is looser.
            let tolerance = if offset < 2e-4 { 1e-15 * 8.0 } else { 1e-7 };
            if error > tolerance || (q.norm() - 1.0).abs() > tolerance {
                failures.push(format!(
                    "{direction:?} offset {offset}: error {error}, norm {}",
                    q.norm()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn rust_only_slerp() {
    let a = Quaterniond::identity();
    let b = Quaterniond::new(S, 0.0, 0.0, S); // 90 degrees about z
    assert_eq!(a.slerp(0.0, b), a);
    assert!(a.slerp(1.0, b).is_approx(b));
    let half = a.slerp(0.5, b);
    let expected = AngleAxisd::new(PI / 4.0, Vector3d::unit_z()).to_quaternion();
    assert!(half.is_approx_with(expected, 1e-15), "{half:?}");
    // Shortest arc: interpolating towards -b is the same as towards b.
    let negated_b = Quaterniond::new(-b.w, -b.x, -b.y, -b.z);
    assert!(a.slerp(0.5, negated_b).is_approx_with(expected, 1e-15));
    // Equal inputs take the linear branch and return the input.
    assert_eq!(b.slerp(0.3, b), b);
}
