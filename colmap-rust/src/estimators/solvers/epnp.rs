//! Port of the `EPNPEstimator` of COLMAP's `src/colmap/estimators/solvers/absolute_pose.h`
//! and `.cc` (the rest of that file is the sibling [`super::absolute_pose`]): EPnP for the
//! Perspective-N-Point problem from four or more 2D-3D correspondences. Port of
//! colmap-sharp's `Estimators/Solvers/Epnp.cs`. Tests: `tests/estimators/absolute_pose.rs`
//! (`AbsolutePose.EPNP`, `AbsolutePose.EPNP_BrokenSolveSignCase`).
//!
//! The algorithm is based on the following paper:
//!
//!    Lepetit, Vincent, Francesc Moreno-Noguer, and Pascal Fua.
//!    "Epnp: An accurate o (n) solution to the pnp problem."
//!    International journal of computer vision 81.2 (2009): 155-166.
//!
//! COLMAP's implementation is based on the authors' original open-source release, ported to
//! Eigen with several improvements; this is a port of COLMAP's.
//!
//! Tier B: the pose goes through several SVDs and a QR.
//!
//! Translation notes:
//! - COLMAP keeps the working state (points, control points, alphas, camera-frame points) in
//!   members of the estimator, so an `EPNPEstimator` cannot run two estimates at once. Here
//!   it lives in a private `EpnpSolver` created per `estimate`, so the estimator is
//!   immutable and RANSAC may clone it freely; the arithmetic is unchanged (as colmap-sharp).
//! - `Eigen::JacobiSVD` becomes [`Svd3d`] for the 3x3 cases and [`JacobiSvd`] for the rest.
//! - COLMAP's estimator has no `Refine`, so LO-RANSAC uses it as a local estimator through
//!   [`crate::optim::EstimateAsLocal`].

use super::absolute_pose::{compute_squared_reprojection_error, ImgFromCamFunc, Point2DWithRay};
use crate::linalg::{
    ColPivHouseholderQr, JacobiSvd, Matrix3d, Matrix3x4d, MatrixXd, Svd3d, SvdOptions, Vector3d,
    VectorXd,
};
use crate::math::fns;
use crate::optim::Estimator;
use crate::{check_eq, check_ge, Result};

/// EPNP solver for the PNP (Perspective-N-Point) problem. The solver needs a minimum of 4
/// 2D-3D correspondences. Port of `colmap::EPNPEstimator`.
#[derive(Clone)]
pub struct EPNPEstimator {
    img_from_cam_func: ImgFromCamFunc,
}

impl EPNPEstimator {
    /// An estimator scoring in pixels through `img_from_cam_func`.
    pub fn new(img_from_cam_func: ImgFromCamFunc) -> Self {
        Self { img_from_cam_func }
    }
}

impl Estimator for EPNPEstimator {
    /// The 2D image feature observations.
    type X = Point2DWithRay;
    /// The observed 3D features in the world frame.
    type Y = Vector3d;
    /// The transformation from the world to the camera frame.
    type M = Matrix3x4d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 4;

    /// Estimate the most probable solution of the EPNP problem from a set of four or more
    /// 2D-3D point correspondences.
    fn estimate(
        &mut self,
        points2d: &[Point2DWithRay],
        points3d: &[Vector3d],
        cams_from_world: &mut Vec<Matrix3x4d>,
    ) -> Result<()> {
        check_ge!(points2d.len(), 4);
        check_eq!(points2d.len(), points3d.len());

        cams_from_world.clear();

        let mut solver = EpnpSolver::new(points2d, points3d, &self.img_from_cam_func);
        if let Some(cam_from_world) = solver.compute_pose()? {
            cams_from_world.push(cam_from_world);
        }
        Ok(())
    }

    /// Calculate the squared reprojection error given a set of 2D-3D point correspondences
    /// and a projection matrix.
    fn residuals(
        &mut self,
        points2d: &[Point2DWithRay],
        points3d: &[Vector3d],
        cam_from_world: &Matrix3x4d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        compute_squared_reprojection_error(
            points2d,
            points3d,
            cam_from_world,
            &self.img_from_cam_func,
            residuals,
        )
    }
}

type L6x10 = [[f64; 10]; 6];

/// The working state and steps of one EPnP estimate (COLMAP's private members).
struct EpnpSolver<'a> {
    points2d: &'a [Point2DWithRay],
    points3d: &'a [Vector3d],
    img_from_cam_func: &'a ImgFromCamFunc,
    pcs: Vec<Vector3d>,
    alphas: Vec<[f64; 4]>,
    cws: [Vector3d; 4],
    ccs: [Vector3d; 4],
}

impl<'a> EpnpSolver<'a> {
    fn new(
        points2d: &'a [Point2DWithRay],
        points3d: &'a [Vector3d],
        img_from_cam_func: &'a ImgFromCamFunc,
    ) -> Self {
        Self {
            points2d,
            points3d,
            img_from_cam_func,
            pcs: vec![Vector3d::zeros(); points2d.len()],
            alphas: vec![[0.0; 4]; points2d.len()],
            cws: [Vector3d::zeros(); 4],
            ccs: [Vector3d::zeros(); 4],
        }
    }

    /// `ComputePose`: `None` when the control points are degenerate.
    fn compute_pose(&mut self) -> Result<Option<Matrix3x4d>> {
        self.choose_control_points();

        if !self.compute_barycentric_coordinates() {
            return Ok(None);
        }

        let m = self.compute_m();
        let mtm = m.transpose_times_self();

        let svd = JacobiSvd::new(&mtm, SvdOptions::FULL_UV);
        let ut = svd.matrix_u().transpose();

        let l6x10 = compute_l6x10(&ut);
        let rho = self.compute_rho();

        let mut betas = [[0.0; 4]; 4];
        let mut reproj_errors = [0.0; 4];
        let mut rs = [Matrix3d::zeros(); 4];
        let mut ts = [Vector3d::zeros(); 4];

        betas[1] = find_betas_approx1(&l6x10, &rho);
        run_gauss_newton(&l6x10, &rho, &mut betas[1]);
        (reproj_errors[1], rs[1], ts[1]) = self.compute_rt(&ut, &betas[1])?;

        betas[2] = find_betas_approx2(&l6x10, &rho);
        run_gauss_newton(&l6x10, &rho, &mut betas[2]);
        (reproj_errors[2], rs[2], ts[2]) = self.compute_rt(&ut, &betas[2])?;

        betas[3] = find_betas_approx3(&l6x10, &rho);
        run_gauss_newton(&l6x10, &rho, &mut betas[3]);
        (reproj_errors[3], rs[3], ts[3]) = self.compute_rt(&ut, &betas[3])?;

        let mut best_idx = 1;
        if reproj_errors[2] < reproj_errors[1] {
            best_idx = 2;
        }
        if reproj_errors[3] < reproj_errors[best_idx] {
            best_idx = 3;
        }

        Ok(Some(Matrix3x4d::from_blocks(rs[best_idx], ts[best_idx])))
    }

    fn choose_control_points(&mut self) {
        let n = self.points3d.len();

        // Take C0 as the reference points centroid:
        let mut cw0 = Vector3d::zeros();
        for p in self.points3d {
            cw0 += *p;
        }
        cw0 /= n as f64;
        self.cws[0] = cw0;

        // PW0^T PW0 with PW0 the centered points as rows.
        let mut pw0 = MatrixXd::zeros(n, 3);
        for (i, p) in self.points3d.iter().enumerate() {
            let d = *p - cw0;
            pw0[(i, 0)] = d.x;
            pw0[(i, 1)] = d.y;
            pw0[(i, 2)] = d.z;
        }

        let pw0t_pw0 = pw0.transpose_times_self().to_matrix3d();
        let svd = Svd3d::compute(&pw0t_pw0);
        let d = svd.singular_values;
        let ut = svd.matrix_u.transpose();

        for i in 1..4 {
            let k = fns::sqrt(d[i - 1] / n as f64);
            self.cws[i] = cw0 + k * ut.row(i - 1);
        }
    }

    fn compute_barycentric_coordinates(&mut self) -> bool {
        let mut cc = MatrixXd::zeros(3, 3);
        for i in 0..3 {
            for j in 1..4 {
                cc[(i, j - 1)] = self.cws[j][i] - self.cws[0][i];
            }
        }

        if ColPivHouseholderQr::new(&cc).rank() < 3 {
            return false;
        }

        let cc_inv = cc.to_matrix3d().inverse();
        let cw0 = self.cws[0];

        for (alpha, p) in self.alphas.iter_mut().zip(self.points3d) {
            for j in 0..3 {
                alpha[1 + j] = cc_inv[(j, 0)] * (p.x - cw0.x)
                    + cc_inv[(j, 1)] * (p.y - cw0.y)
                    + cc_inv[(j, 2)] * (p.z - cw0.z);
            }
            alpha[0] = 1.0 - alpha[1] - alpha[2] - alpha[3];
        }

        true
    }

    fn compute_m(&self) -> MatrixXd {
        let mut m = MatrixXd::zeros(3 * self.points2d.len(), 12);
        for i in 0..self.points3d.len() {
            let ray = self.points2d[i].camera_ray;
            let alpha = &self.alphas[i];
            for j in 0..4 {
                m[(3 * i, 3 * j)] = 0.0;
                m[(3 * i, 3 * j + 1)] = -alpha[j] * ray.z;
                m[(3 * i, 3 * j + 2)] = alpha[j] * ray.y;

                m[(3 * i + 1, 3 * j)] = alpha[j] * ray.z;
                m[(3 * i + 1, 3 * j + 1)] = 0.0;
                m[(3 * i + 1, 3 * j + 2)] = -alpha[j] * ray.x;

                m[(3 * i + 2, 3 * j)] = -alpha[j] * ray.y;
                m[(3 * i + 2, 3 * j + 1)] = alpha[j] * ray.x;
                m[(3 * i + 2, 3 * j + 2)] = 0.0;
            }
        }
        m
    }

    fn compute_rho(&self) -> [f64; 6] {
        let c = &self.cws;
        [
            (c[0] - c[1]).squared_norm(),
            (c[0] - c[2]).squared_norm(),
            (c[0] - c[3]).squared_norm(),
            (c[1] - c[2]).squared_norm(),
            (c[1] - c[3]).squared_norm(),
            (c[2] - c[3]).squared_norm(),
        ]
    }

    /// `ComputeRT`: the pose for `betas` and its total reprojection error.
    fn compute_rt(&mut self, ut: &MatrixXd, betas: &[f64; 4]) -> Result<(f64, Matrix3d, Vector3d)> {
        self.compute_ccs(betas, ut);
        self.compute_pcs();

        self.solve_for_sign();

        let (r, t) = self.estimate_rt();

        Ok((self.compute_total_error(r, t)?, r, t))
    }

    fn compute_ccs(&mut self, betas: &[f64; 4], ut: &MatrixXd) {
        self.ccs = [Vector3d::zeros(); 4];

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..3 {
                    self.ccs[j][k] += betas[i] * ut[(11 - i, 3 * j + k)];
                }
            }
        }
    }

    fn compute_pcs(&mut self) {
        let ccs = &self.ccs;
        for (pc, alpha) in self.pcs.iter_mut().zip(&self.alphas) {
            for j in 0..3 {
                pc[j] = alpha[0] * ccs[0][j]
                    + alpha[1] * ccs[1][j]
                    + alpha[2] * ccs[2][j]
                    + alpha[3] * ccs[3][j];
            }
        }
    }

    fn solve_for_sign(&mut self) {
        if self.pcs[0].z < 0.0 {
            for cc in &mut self.ccs {
                *cc = -*cc;
            }
            for pc in &mut self.pcs {
                *pc = -*pc;
            }
        }
    }

    fn estimate_rt(&self) -> (Matrix3d, Vector3d) {
        let n = self.points3d.len() as f64;
        let mut pc0 = Vector3d::zeros();
        let mut pw0 = Vector3d::zeros();

        for (pc, pw) in self.pcs.iter().zip(self.points3d) {
            pc0 += *pc;
            pw0 += *pw;
        }
        pc0 /= n;
        pw0 /= n;

        let mut abt = Matrix3d::zeros();
        for (pc, pw) in self.pcs.iter().zip(self.points3d) {
            for j in 0..3 {
                abt[(j, 0)] += (pc[j] - pc0[j]) * (pw.x - pw0.x);
                abt[(j, 1)] += (pc[j] - pc0[j]) * (pw.y - pw0.y);
                abt[(j, 2)] += (pc[j] - pc0[j]) * (pw.z - pw0.z);
            }
        }

        let svd = Svd3d::compute(&abt);
        let abt_u = svd.matrix_u;
        let abt_v = svd.matrix_v;

        let mut r = row_products(abt_u, abt_v);

        if r.determinant() < 0.0 {
            let abt_v_prime = Matrix3d::from_columns(abt_v.col(0), abt_v.col(1), -abt_v.col(2));
            r = row_products(abt_u, abt_v_prime);
        }

        let t = pc0 - r * pw0;
        (r, t)
    }

    fn compute_total_error(&self, r: Matrix3d, t: Vector3d) -> Result<f64> {
        let cam_from_world = Matrix3x4d::from_blocks(r, t);

        let mut residuals = Vec::with_capacity(self.points2d.len());
        compute_squared_reprojection_error(
            self.points2d,
            self.points3d,
            &cam_from_world,
            self.img_from_cam_func,
            &mut residuals,
        )?;

        let mut error = 0.0;
        for residual in residuals {
            error += fns::sqrt(residual);
        }

        Ok(error)
    }
}

/// `R(i, j) = U.row(i) * V.row(j)^T`.
fn row_products(u: Matrix3d, v: Matrix3d) -> Matrix3d {
    let mut m = Matrix3d::zeros();
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = u.row(i).dot(v.row(j));
        }
    }
    m
}

fn compute_l6x10(ut: &MatrixXd) -> L6x10 {
    let mut l6x10 = [[0.0; 10]; 6];

    let mut dv = [[Vector3d::zeros(); 6]; 4];
    for (i, dvi) in dv.iter_mut().enumerate() {
        let (mut a, mut b) = (0, 1);
        for dvij in dvi.iter_mut() {
            *dvij = Vector3d::new(
                ut[(11 - i, 3 * a)] - ut[(11 - i, 3 * b)],
                ut[(11 - i, 3 * a + 1)] - ut[(11 - i, 3 * b + 1)],
                ut[(11 - i, 3 * a + 2)] - ut[(11 - i, 3 * b + 2)],
            );

            b += 1;
            if b > 3 {
                a += 1;
                b = a + 1;
            }
        }
    }

    for (i, row) in l6x10.iter_mut().enumerate() {
        row[0] = dv[0][i].dot(dv[0][i]);
        row[1] = 2.0 * dv[0][i].dot(dv[1][i]);
        row[2] = dv[1][i].dot(dv[1][i]);
        row[3] = 2.0 * dv[0][i].dot(dv[2][i]);
        row[4] = 2.0 * dv[1][i].dot(dv[2][i]);
        row[5] = dv[2][i].dot(dv[2][i]);
        row[6] = 2.0 * dv[0][i].dot(dv[3][i]);
        row[7] = 2.0 * dv[1][i].dot(dv[3][i]);
        row[8] = 2.0 * dv[2][i].dot(dv[3][i]);
        row[9] = dv[3][i].dot(dv[3][i]);
    }

    l6x10
}

/// Least-squares solution of `L6x10[:, cols] b = rho` by a full SVD (`JacobiSVD::solve`).
fn solve_columns(l6x10: &L6x10, rho: &[f64; 6], cols: &[usize]) -> VectorXd {
    let mut l = MatrixXd::zeros(6, cols.len());
    for (i, row) in l6x10.iter().enumerate() {
        for (j, &col) in cols.iter().enumerate() {
            l[(i, j)] = row[col];
        }
    }

    let svd = JacobiSvd::new(&l, SvdOptions::FULL_UV);
    svd.solve(&VectorXd::from_slice(rho))
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]
fn find_betas_approx1(l6x10: &L6x10, rho: &[f64; 6]) -> [f64; 4] {
    let b4 = solve_columns(l6x10, rho, &[0, 1, 3, 6]);
    let mut betas = [0.0; 4];

    if b4[0] < 0.0 {
        betas[0] = fns::sqrt(-b4[0]);
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    } else {
        betas[0] = fns::sqrt(b4[0]);
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }

    betas
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]
fn find_betas_approx2(l6x10: &L6x10, rho: &[f64; 6]) -> [f64; 4] {
    let b3 = solve_columns(l6x10, rho, &[0, 1, 2]);
    let mut betas = [0.0; 4];

    if b3[0] < 0.0 {
        betas[0] = fns::sqrt(-b3[0]);
        betas[1] = if b3[2] < 0.0 { fns::sqrt(-b3[2]) } else { 0.0 };
    } else {
        betas[0] = fns::sqrt(b3[0]);
        betas[1] = if b3[2] > 0.0 { fns::sqrt(b3[2]) } else { 0.0 };
    }

    if b3[1] < 0.0 {
        betas[0] = -betas[0];
    }

    betas[2] = 0.0;
    betas[3] = 0.0;
    betas
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]
fn find_betas_approx3(l6x10: &L6x10, rho: &[f64; 6]) -> [f64; 4] {
    let b5 = solve_columns(l6x10, rho, &[0, 1, 2, 3, 4]);
    let mut betas = [0.0; 4];

    if b5[0] < 0.0 {
        betas[0] = fns::sqrt(-b5[0]);
        betas[1] = if b5[2] < 0.0 { fns::sqrt(-b5[2]) } else { 0.0 };
    } else {
        betas[0] = fns::sqrt(b5[0]);
        betas[1] = if b5[2] > 0.0 { fns::sqrt(b5[2]) } else { 0.0 };
    }
    if b5[1] < 0.0 {
        betas[0] = -betas[0];
    }
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
    betas
}

fn run_gauss_newton(l: &L6x10, rho: &[f64; 6], betas: &mut [f64; 4]) {
    let mut a = MatrixXd::zeros(6, 4);
    let mut b = VectorXd::zeros(6);

    const NUM_ITERATIONS: usize = 5;
    for _ in 0..NUM_ITERATIONS {
        for i in 0..6 {
            let li = &l[i];
            a[(i, 0)] =
                2.0 * li[0] * betas[0] + li[1] * betas[1] + li[3] * betas[2] + li[6] * betas[3];
            a[(i, 1)] =
                li[1] * betas[0] + 2.0 * li[2] * betas[1] + li[4] * betas[2] + li[7] * betas[3];
            a[(i, 2)] =
                li[3] * betas[0] + li[4] * betas[1] + 2.0 * li[5] * betas[2] + li[8] * betas[3];
            a[(i, 3)] =
                li[6] * betas[0] + li[7] * betas[1] + li[8] * betas[2] + 2.0 * li[9] * betas[3];

            b[i] = rho[i]
                - (li[0] * betas[0] * betas[0]
                    + li[1] * betas[0] * betas[1]
                    + li[2] * betas[1] * betas[1]
                    + li[3] * betas[0] * betas[2]
                    + li[4] * betas[1] * betas[2]
                    + li[5] * betas[2] * betas[2]
                    + li[6] * betas[0] * betas[3]
                    + li[7] * betas[1] * betas[3]
                    + li[8] * betas[2] * betas[3]
                    + li[9] * betas[3] * betas[3]);
        }

        let x = ColPivHouseholderQr::new(&a).solve(&b);

        for (beta, dx) in betas.iter_mut().zip(x.as_slice()) {
            *beta += dx;
        }
    }
}
