// Port of COLMAP's src/colmap/geometry/triangulation_test.cc. Random values come from
// COLMAP's seeded PRNG (math::random_eigen), seeded with 0 like gtest_main. COLMAP's
// `bool` + out-pointer results are `Option` here (`EXPECT_TRUE` -> `Some`).

use std::f64::consts::PI;

use colmap_rust::geometry::triangulation::{
    calculate_angle_between_vectors, calculate_triangulation_angle, calculate_triangulation_angles,
    triangulate_mid_point, triangulate_multi_view_point, triangulate_multi_view_point_from_rays,
    triangulate_point, triangulate_point_from_rays,
};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Quaterniond, Vector2d, Vector3d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};

use super::{eigen_matrix_near, expect_near};

fn nominal_points() -> [Vector3d; 6] {
    [
        Vector3d::new(0.0, 0.1, 0.1),
        Vector3d::new(0.0, 1.0, 3.0),
        Vector3d::new(0.0, 1.0, 2.0),
        Vector3d::new(0.01, 0.2, 3.0),
        Vector3d::new(-1.0, 0.1, 1.0),
        Vector3d::new(0.1, 0.1, 0.2),
    ]
}

fn bearing_points() -> [Vector3d; 5] {
    [
        Vector3d::new(0.0, 0.1, 0.1),
        Vector3d::new(0.0, 1.0, 3.0),
        Vector3d::new(-1.0, 0.1, 1.0),
        Vector3d::new(0.1, 0.1, -0.5),  // behind cam1 (negative Z)
        Vector3d::new(0.2, -0.3, -2.0), // behind cam1
    ]
}

#[test]
fn triangulate_point_nominal() {
    let cam1_from_world = Rigid3d::identity();
    for z in 0..5 {
        let qz = f64::from(z) / 5.0;
        for tx in (0..10).step_by(2) {
            let cam2_from_world = Rigid3d::new(
                Quaterniond::new(0.2, 0.3, 0.4, qz),
                Vector3d::new(f64::from(tx), 2.0, 3.0),
            );
            for point3d in nominal_points() {
                let point1 = (cam1_from_world * point3d).hnormalized();
                let point2 = (cam2_from_world * point3d).hnormalized();

                let tri_point3d = triangulate_point(
                    &cam1_from_world.to_matrix(),
                    &cam2_from_world.to_matrix(),
                    point1,
                    point2,
                )
                .expect("triangulated");

                assert!(eigen_matrix_near(&point3d, &tri_point3d, 1e-10));
            }
        }
    }
}

#[test]
fn triangulate_point_parallel_rays() {
    assert!(triangulate_point(
        &Rigid3d::identity().to_matrix(),
        &Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0)).to_matrix(),
        Vector2d::new(0.0, 0.0),
        Vector2d::new(0.0, 0.0),
    )
    .is_none());
}

#[test]
fn triangulate_point_bearings() {
    let cam1_from_world = Rigid3d::identity();
    let cam2_from_world = Rigid3d::new(
        Quaterniond::new(0.21, 0.31, 0.41, 0.1),
        Vector3d::new(1.0, 2.0, 3.0),
    );

    for point3d in bearing_points() {
        let cam_ray1 = (cam1_from_world * point3d).normalized();
        let cam_ray2 = (cam2_from_world * point3d).normalized();

        let tri_point3d = triangulate_point_from_rays(
            &cam1_from_world.to_matrix(),
            &cam2_from_world.to_matrix(),
            cam_ray1,
            cam_ray2,
        )
        .expect("triangulated");
        assert!(eigen_matrix_near(&point3d, &tri_point3d, 1e-9));
    }
}

#[test]
fn triangulate_point_bearings_parallel_rays() {
    assert!(triangulate_point_from_rays(
        &Rigid3d::identity().to_matrix(),
        &Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0)).to_matrix(),
        Vector3d::new(0.0, 0.0, 1.0),
        Vector3d::new(0.0, 0.0, 1.0),
    )
    .is_none());
}

#[test]
fn triangulate_mid_point_nominal() {
    set_prng_seed(0);
    const NUM_TRIALS: usize = 10;
    for _ in 0..NUM_TRIALS {
        let cam1_from_world = Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d());
        let cam2_from_world = Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d());
        let point3d = random_eigen_vector3d();
        let cam_ray1 = (cam1_from_world * point3d).normalized();
        let cam_ray2 = (cam2_from_world * point3d).normalized();

        let point3d_in_cam1 = triangulate_mid_point(
            &(cam2_from_world * cam1_from_world.inverse()),
            cam_ray1,
            cam_ray2,
        )
        .expect("triangulated");
        let point3d_in_world = cam1_from_world.inverse() * point3d_in_cam1;
        assert!(eigen_matrix_near(&point3d, &point3d_in_world, 1e-10));
    }
}

#[test]
fn triangulate_mid_point_non_perfect_intersection() {
    let cam1_from_world = Rigid3d::new(Quaterniond::identity(), Vector3d::new(-1.0, 0.0, 0.0));
    let cam2_from_world = Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0));
    let expected_point3d = Vector3d::new(0.0, 0.0, 5.0);
    let point3d_in_cam1 = triangulate_mid_point(
        &(cam2_from_world * cam1_from_world.inverse()),
        (cam1_from_world * (expected_point3d + Vector3d::new(0.0, 0.1, 0.0))).normalized(),
        (cam2_from_world * (expected_point3d + Vector3d::new(0.0, -0.1, 0.0))).normalized(),
    )
    .expect("triangulated");
    assert!(eigen_matrix_near(
        &point3d_in_cam1,
        &(cam1_from_world * expected_point3d),
        1e-3
    ));
}

#[test]
fn triangulate_mid_point_parallel_rays() {
    assert!(triangulate_mid_point(
        &Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0)),
        Vector3d::new(0.0, 0.0, 1.0),
        Vector3d::new(0.0, 0.0, 1.0),
    )
    .is_none());
}

#[test]
fn triangulate_mid_point_behind_cameras() {
    assert!(triangulate_mid_point(
        &Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0)),
        Vector3d::new(0.1, 0.0, 1.0).normalized(),
        Vector3d::new(-0.1, 0.0, 1.0).normalized(),
    )
    .is_none());
}

#[test]
fn triangulate_multi_view_point_nominal() {
    let cam1_from_world = Rigid3d::identity();
    for z in 0..5 {
        let qz = f64::from(z) / 5.0;
        for tx in (0..10).step_by(2) {
            let tx = f64::from(tx);
            let cam2_from_world = Rigid3d::new(
                Quaterniond::new(0.21, 0.31, 0.41, qz),
                Vector3d::new(tx, 2.0, 3.0),
            );
            let cam3_from_world = Rigid3d::new(
                Quaterniond::new(0.2, 0.3, 0.4, qz),
                Vector3d::new(tx, 2.1, 3.1),
            );
            for point3d in nominal_points() {
                let points = [
                    (cam1_from_world * point3d).hnormalized(),
                    (cam2_from_world * point3d).hnormalized(),
                    (cam3_from_world * point3d).hnormalized(),
                ];
                let cams_from_world = [
                    cam1_from_world.to_matrix(),
                    cam2_from_world.to_matrix(),
                    cam3_from_world.to_matrix(),
                ];

                let tri_point3d = triangulate_multi_view_point(&cams_from_world, &points)
                    .unwrap()
                    .expect("triangulated");

                assert!(eigen_matrix_near(&point3d, &tri_point3d, 1e-10));
            }
        }
    }
}

// The 3D bearing overload recovers the same points as the 2D overload, and
// additionally handles back-hemisphere rays (negative Z in the camera frame)
// that the 2D (u, v, 1) representation cannot encode -- as produced by
// omnidirectional (e.g. EQUIRECTANGULAR) cameras.
#[test]
fn triangulate_multi_view_point_bearings() {
    let cam1_from_world = Rigid3d::identity();
    let cam2_from_world = Rigid3d::new(
        Quaterniond::new(0.21, 0.31, 0.41, 0.1),
        Vector3d::new(1.0, 2.0, 3.0),
    );
    let cam3_from_world = Rigid3d::new(
        Quaterniond::new(0.2, 0.3, 0.4, 0.05),
        Vector3d::new(2.0, 2.1, 3.1),
    );
    let cams_from_world = [
        cam1_from_world.to_matrix(),
        cam2_from_world.to_matrix(),
        cam3_from_world.to_matrix(),
    ];

    for point3d in bearing_points() {
        let cam_rays = [
            (cam1_from_world * point3d).normalized(),
            (cam2_from_world * point3d).normalized(),
            (cam3_from_world * point3d).normalized(),
        ];

        let tri_point3d = triangulate_multi_view_point_from_rays(&cams_from_world, &cam_rays)
            .unwrap()
            .expect("triangulated");
        assert!(eigen_matrix_near(&point3d, &tri_point3d, 1e-9));
    }
}

fn each_near(values: &[f64], target: f64, tol: f64) {
    for &v in values {
        expect_near(v, target, tol);
    }
}

#[test]
fn calculate_triangulation_angle_nominal() {
    let tvec1 = Vector3d::new(0.0, 0.0, 0.0);
    let tvec2 = Vector3d::new(0.0, 1.0, 0.0);

    expect_near(
        calculate_triangulation_angle(tvec1, tvec2, Vector3d::new(0.0, 0.0, 100.0)),
        0.009999666687,
        1e-8,
    );
    expect_near(
        calculate_triangulation_angle(tvec1, tvec2, Vector3d::new(0.0, 0.0, 50.0)),
        0.019997333973,
        1e-8,
    );
    expect_near(
        calculate_triangulation_angles(tvec1, tvec2, &[Vector3d::new(0.0, 0.0, 100.0)])[0],
        0.009999666687,
        1e-8,
    );
    expect_near(
        calculate_triangulation_angles(tvec1, tvec2, &[Vector3d::new(0.0, 0.0, 50.0)])[0],
        0.019997333973,
        1e-8,
    );
    // Parallel rays.
    each_near(
        &calculate_triangulation_angles(
            Vector3d::zeros(),
            Vector3d::zeros(),
            &[
                Vector3d::new(0.0, 0.0, 0.0),
                Vector3d::new(50.0, 0.0, 0.0),
                Vector3d::new(0.0, 50.0, 0.0),
                Vector3d::new(0.0, 0.0, 50.0),
            ],
        ),
        0.0,
        1e-6,
    );
    // Orthogonal rays.
    each_near(
        &calculate_triangulation_angles(
            Vector3d::zeros(),
            Vector3d::new(50.0, 0.0, 50.0),
            &[Vector3d::new(50.0, 0.0, 0.0), Vector3d::new(0.0, 0.0, 50.0)],
        ),
        PI / 2.0,
        1e-6,
    );
    // Opposing rays.
    each_near(
        &calculate_triangulation_angles(
            Vector3d::zeros(),
            Vector3d::new(0.0, 0.0, 50.0),
            &[
                Vector3d::new(0.0, 0.0, 0.0),
                Vector3d::new(0.0, 0.0, 50.0),
                Vector3d::new(0.0, 0.0, 25.0),
                Vector3d::new(0.0, 0.0, -25.0),
                Vector3d::new(0.0, 0.0, 75.0),
            ],
        ),
        0.0,
        1e-6,
    );
}

#[test]
fn calculate_angle_between_vectors_parallel_vectors() {
    let v1 = Vector3d::new(1.0, 0.0, 0.0);
    let v2 = Vector3d::new(2.0, 0.0, 0.0);
    expect_near(calculate_angle_between_vectors(v1, v2), 0.0, 1e-10);
}

#[test]
fn calculate_angle_between_vectors_opposite_vectors() {
    let v1 = Vector3d::new(1.0, 0.0, 0.0);
    let v2 = Vector3d::new(-1.0, 0.0, 0.0);
    expect_near(calculate_angle_between_vectors(v1, v2), PI, 1e-10);
}

#[test]
fn calculate_angle_between_vectors_perpendicular_vectors() {
    let v1 = Vector3d::new(1.0, 0.0, 0.0);
    let v2 = Vector3d::new(0.0, 1.0, 0.0);
    expect_near(calculate_angle_between_vectors(v1, v2), PI / 2.0, 1e-10);
}

#[test]
fn calculate_angle_between_vectors_perpendicular_vectors_different_magnitudes() {
    let v1 = Vector3d::new(3.0, 0.0, 0.0);
    let v2 = Vector3d::new(0.0, 5.0, 0.0);
    expect_near(calculate_angle_between_vectors(v1, v2), PI / 2.0, 1e-10);
}

#[test]
fn calculate_angle_between_vectors_zero_vector() {
    let v1 = Vector3d::new(1.0, 0.0, 0.0);
    let v2 = Vector3d::new(0.0, 0.0, 0.0);
    expect_near(calculate_angle_between_vectors(v1, v2), 0.0, 1e-10);
    expect_near(calculate_angle_between_vectors(v2, v1), 0.0, 1e-10);
}

#[test]
fn calculate_angle_between_vectors_both_zero_vectors() {
    let v1 = Vector3d::new(0.0, 0.0, 0.0);
    let v2 = Vector3d::new(0.0, 0.0, 0.0);
    expect_near(calculate_angle_between_vectors(v1, v2), 0.0, 1e-10);
}

#[test]
fn calculate_angle_between_vectors_forty_five_degrees() {
    let v1 = Vector3d::new(1.0, 0.0, 0.0);
    let v2 = Vector3d::new(1.0, 1.0, 0.0);
    expect_near(calculate_angle_between_vectors(v1, v2), PI / 4.0, 1e-10);
}

#[test]
fn calculate_angle_between_vectors_identical_vectors() {
    let v1 = Vector3d::new(1.0, 2.0, 3.0);
    let v2 = Vector3d::new(1.0, 2.0, 3.0);
    expect_near(calculate_angle_between_vectors(v1, v2), 0.0, 1e-10);
}
