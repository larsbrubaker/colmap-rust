//! Port of COLMAP's `colmap/estimators/cost_functions/sampson_error.h`, via colmap-sharp's
//! `Estimators/CostFunctions/SampsonError.cs`: the scalar-generic (`f64` or
//! [`crate::solver::Jet`]) essential matrix of a relative pose, the signed Sampson and tangent
//! Sampson errors, and the two Ceres autodiff cost functors built on them
//! ([`SampsonErrorCostFunctor`], [`TangentSampsonErrorCostFunctor`], wrapped by
//! [`AutoDiffCostFunction`]). [`super::tiny_sampson_error`]'s one-sided focal functor reuses
//! the generic pieces for its autodiff form.
//!
//! Translation notes:
//! - `Eigen::Matrix<T, 3, 3>` is a row-major `[[T; 3]; 3]`.
//! - C++ casts the measurements (points, rays, unprojection Jacobians) to `T`, so they meet
//!   the parameter-dependent `E` through Jet-Jet arithmetic; so do these functions, through
//!   [`Scalar::from_f64`] (colmap-sharp used the mixed Jet/double operators instead, which
//!   differ only in the sign of a zero derivative).
//! - Eigen evaluates these small fixed-size products coefficient by coefficient and reduces
//!   each inner product with its unrolled, non-vectorized redux, which splits in halves: a
//!   3-term sum is `e0 + (e1 + e2)` and a 4-term squared norm `(e0 + e1) + (e2 + e3)`. The
//!   grouping is kept.
//! - The zero-denominator test compares the value part, as Ceres' Jet `operator==` does.
//!
//! Tier: scalar formulas (Tier A up to Eigen's grouping, which is kept); the tests pin them
//! within COLMAP's tolerances. Tests: `tests/estimators/tiny_sampson_error.rs`
//! (`TinyTangentSampsonErrorCostFunctor.MatchesAutodiffCostFunctor`).

use crate::geometry::pose::CamRayWithJac;
use crate::linalg::{Matrix3x2d, Vector2d, Vector3d};
use crate::solver::{AutoDiffCostFunction, AutoDiffFunctor, Scalar};
use crate::Result;

/// `Eigen::Matrix<T, 3, 3>`, row-major.
pub type Matrix3T<T> = [[T; 3]; 3];

/// Eigen's 3-term inner product `a0 b0 + a1 b1 + a2 b2`, reduced as `e0 + (e1 + e2)`.
#[inline]
fn dot3<T: Scalar>(a: [T; 3], b: [T; 3]) -> T {
    a[0] * b[0] + (a[1] * b[1] + a[2] * b[2])
}

/// `T(v)` of a double 3-vector.
#[inline]
fn cast3<T: Scalar>(v: Vector3d) -> [T; 3] {
    [T::from_f64(v.x), T::from_f64(v.y), T::from_f64(v.z)]
}

/// `M * v`.
fn mat_vec<T: Scalar>(m: &Matrix3T<T>, v: [T; 3]) -> [T; 3] {
    [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)]
}

/// `M^T * v`.
fn mat_t_vec<T: Scalar>(m: &Matrix3T<T>, v: [T; 3]) -> [T; 3] {
    let col = |c: usize| [m[0][c], m[1][c], m[2][c]];
    [dot3(col(0), v), dot3(col(1), v), dot3(col(2), v)]
}

/// `J^T * v` for a 3 x 2 `J` cast to `T`.
fn jac_t_vec<T: Scalar>(j: &Matrix3x2d, v: [T; 3]) -> [T; 2] {
    let col = |c: usize| {
        [
            T::from_f64(j[(0, c)]),
            T::from_f64(j[(1, c)]),
            T::from_f64(j[(2, c)]),
        ]
    };
    [dot3(col(0), v), dot3(col(1), v)]
}

/// Eigen's `toRotationMatrix()` of the quaternion with coefficients (x, y, z, w), with
/// Eigen's own products (`txy = ty * x`, ...) and grouping.
pub fn quaternion_to_rotation_matrix<T: Scalar>(x: T, y: T, z: T, w: T) -> Matrix3T<T> {
    let two = T::from_f64(2.0);
    let one = T::from_f64(1.0);
    let tx = two * x;
    let ty = two * y;
    let tz = two * z;
    let twx = tx * w;
    let twy = ty * w;
    let twz = tz * w;
    let txx = tx * x;
    let txy = ty * x;
    let txz = tz * x;
    let tyy = ty * y;
    let tyz = tz * y;
    let tzz = tz * z;
    [
        [one - (tyy + tzz), txy - twz, txz + twy],
        [txy + twz, one - (txx + tzz), tyz - twx],
        [txz - twy, tyz + twx, one - (txx + tyy)],
    ]
}

/// Port of `colmap::EssentialMatrixFromPoseParams`: `E = [t]_x R` from a relative pose in the
/// Rigid3d parameter layout [qx, qy, qz, qw, tx, ty, tz], generic so it works under
/// autodiff.
pub fn essential_matrix_from_pose_params<T: Scalar>(cam2_from_cam1: &[T]) -> Matrix3T<T> {
    let p = cam2_from_cam1;
    let r = quaternion_to_rotation_matrix(p[0], p[1], p[2], p[3]);
    // Matrix representation of the cross product t x R.
    let zero = T::from_f64(0.0);
    let t_x = [
        [zero, -p[6], p[5]],
        [p[6], zero, -p[4]],
        [-p[5], p[4], zero],
    ];
    let col = |c: usize| [r[0][c], r[1][c], r[2][c]];
    let (c0, c1, c2) = (col(0), col(1), col(2));
    [
        [dot3(t_x[0], c0), dot3(t_x[0], c1), dot3(t_x[0], c2)],
        [dot3(t_x[1], c0), dot3(t_x[1], c1), dot3(t_x[1], c2)],
        [dot3(t_x[2], c0), dot3(t_x[2], c1), dot3(t_x[2], c2)],
    ]
}

/// `sqrt(squaredNorm())` of a 4-vector, Eigen's halves-first reduction, then `num / norm`, or
/// zero when the norm vanishes.
fn signed_error<T: Scalar>(num: T, denom: [T; 4]) -> T {
    let squared_norm =
        (denom[0] * denom[0] + denom[1] * denom[1]) + (denom[2] * denom[2] + denom[3] * denom[3]);
    let denom_norm = squared_norm.sqrt();
    if denom_norm.value() == 0.0 {
        return T::from_f64(0.0);
    }
    num / denom_norm
}

/// Port of `colmap::SampsonError`: signed Sampson error under an essential/fundamental
/// matrix. `point1`/`point2` are points on the image plane, taken as 2D because the error is
/// not invariant to the scale of the homogeneous representative, so only (x, y, 1) is
/// correct here. For rays use [`tangent_sampson_error`]. Returns 0 when the denominator
/// vanishes.
pub fn sampson_error<T: Scalar>(e: &Matrix3T<T>, point1: [T; 2], point2: [T; 2]) -> T {
    let one = T::from_f64(1.0);
    let point2_homogeneous = [point2[0], point2[1], one];
    let epipolar_line1 = mat_vec(e, [point1[0], point1[1], one]);
    let num = dot3(point2_homogeneous, epipolar_line1);
    let col = |c: usize| [e[0][c], e[1][c], e[2][c]];
    let denom = [
        dot3(point2_homogeneous, col(0)),
        dot3(point2_homogeneous, col(1)),
        epipolar_line1[0],
        epipolar_line1[1],
    ];
    signed_error(num, denom)
}

/// Port of `colmap::TangentSampsonError`: signed tangent Sampson error of one correspondence
/// under `e`, in pixels, using the unprojection Jacobians `j_ray1 = d(ray1)/d(pixel1)` and
/// `j_ray2`. Returns 0 when the denominator vanishes. See
/// `geometry::essential_matrix::compute_squared_tangent_sampson_error`.
pub fn tangent_sampson_error<T: Scalar>(
    e: &Matrix3T<T>,
    cam_ray1: [T; 3],
    j_ray1: &Matrix3x2d,
    cam_ray2: [T; 3],
    j_ray2: &Matrix3x2d,
) -> T {
    let e_ray1 = mat_vec(e, cam_ray1);
    let et_ray2 = mat_t_vec(e, cam_ray2);
    let num = dot3(cam_ray2, e_ray1);
    let d1 = jac_t_vec(j_ray1, et_ray2);
    let d2 = jac_t_vec(j_ray2, e_ray1);
    signed_error(num, [d1[0], d1[1], d2[0], d2[1]])
}

/// Port of `colmap::SampsonErrorCostFunctor`: refines a relative pose by the Sampson error of
/// image-plane point correspondences. The pose is [qx, qy, qz, qw, tx, ty, tz] with the
/// translation on the unit sphere, so it needs a sphere manifold on tvec. For calibrated
/// rays with unprojection Jacobians use [`TangentSampsonErrorCostFunctor`], which is
/// pixel-accurate for any central model.
#[derive(Clone, Copy, Debug)]
pub struct SampsonErrorCostFunctor {
    point1: Vector2d,
    point2: Vector2d,
}

impl SampsonErrorCostFunctor {
    /// The functor of one correspondence.
    pub fn new(point1: Vector2d, point2: Vector2d) -> Self {
        Self { point1, point2 }
    }

    /// C++ `Create`: the autodiff cost function, 1 residual, one 7-parameter block.
    pub fn create(point1: Vector2d, point2: Vector2d) -> Result<AutoDiffCostFunction<Self, 1, 7>> {
        AutoDiffCostFunction::new(Self::new(point1, point2), &[7])
    }
}

impl AutoDiffFunctor for SampsonErrorCostFunctor {
    fn call<T: Scalar>(&self, cam2_from_cam1: &[T], residuals: &mut [T]) -> bool {
        let e = essential_matrix_from_pose_params(cam2_from_cam1);
        let cast2 = |p: Vector2d| [T::from_f64(p.x), T::from_f64(p.y)];
        residuals[0] = sampson_error(&e, cast2(self.point1), cast2(self.point2));
        true
    }
}

/// Port of `colmap::TangentSampsonErrorCostFunctor`: refines a relative pose by the
/// pixel-unit tangent Sampson error of calibrated ray correspondences with unprojection
/// Jacobians. Pose layout matches [`SampsonErrorCostFunctor`]. Pixel-accurate for any
/// central model.
#[derive(Clone, Copy, Debug)]
pub struct TangentSampsonErrorCostFunctor {
    cam_ray1_with_jac: CamRayWithJac,
    cam_ray2_with_jac: CamRayWithJac,
}

impl TangentSampsonErrorCostFunctor {
    /// The functor of one correspondence.
    pub fn new(cam_ray1_with_jac: CamRayWithJac, cam_ray2_with_jac: CamRayWithJac) -> Self {
        Self {
            cam_ray1_with_jac,
            cam_ray2_with_jac,
        }
    }

    /// C++ `Create`: the autodiff cost function, 1 residual, one 7-parameter block.
    pub fn create(
        cam_ray1_with_jac: CamRayWithJac,
        cam_ray2_with_jac: CamRayWithJac,
    ) -> Result<AutoDiffCostFunction<Self, 1, 7>> {
        AutoDiffCostFunction::new(Self::new(cam_ray1_with_jac, cam_ray2_with_jac), &[7])
    }
}

impl AutoDiffFunctor for TangentSampsonErrorCostFunctor {
    fn call<T: Scalar>(&self, cam2_from_cam1: &[T], residuals: &mut [T]) -> bool {
        let e = essential_matrix_from_pose_params(cam2_from_cam1);
        residuals[0] = tangent_sampson_error(
            &e,
            cast3(self.cam_ray1_with_jac.ray),
            &self.cam_ray1_with_jac.jacobian,
            cast3(self.cam_ray2_with_jac.ray),
            &self.cam_ray2_with_jac.jacobian,
        );
        true
    }
}
