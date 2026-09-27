//! `MatrixXd` arithmetic and conversions: coefficient-wise operators, matrix-matrix and
//! matrix-vector products, the transposed products `A^T * B`, `A^T * v` and `A^T * A` that
//! COLMAP's estimators form on tall constraint matrices without materializing `A^T`, and the
//! conversions to and from the fixed-size matrices.
//!
//! Port of colmap-sharp's `MatrixXd.Arithmetic.cs` and `MatrixXd.Conversions.cs` (MIT). The
//! type and its storage are in `matrix_x.rs`. Every product coefficient is a left-to-right sum
//! over k seeded with the k = 0 term, no FMA (Tier B against Eigen's blocked kernels,
//! docs/CPP_DIVERGENCES.md entry 20). Both sides of a conversion store column-major, so the
//! conversions are plain buffer copies.
//! Tests: `colmap-rust/tests/linalg/rust_only_dynamic_matrix.rs`.

use super::matrix_x::require_same_len;
use super::{dot, Matrix2d, Matrix3d, Matrix3x4d, Matrix4d, Matrix6d, MatrixXd, VectorXd};

impl MatrixXd {
    fn zip_with(&self, b: &MatrixXd, f: impl Fn(f64, f64) -> f64) -> MatrixXd {
        assert!(
            self.rows() == b.rows() && self.cols() == b.cols(),
            "Shape mismatch: {}x{} vs {}x{}.",
            self.rows(),
            self.cols(),
            b.rows(),
            b.cols()
        );
        let data = self
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(&x, &y)| f(x, y))
            .collect();
        MatrixXd::from_column_major_vec(self.rows(), self.cols(), data)
    }

    fn map_with(&self, f: impl Fn(f64) -> f64) -> MatrixXd {
        let data = self.as_slice().iter().map(|&x| f(x)).collect();
        MatrixXd::from_column_major_vec(self.rows(), self.cols(), data)
    }

    /// `self^T * b` without forming the transpose; both operands are walked down contiguous
    /// columns, so this is the fast way to build normal equations.
    pub fn transpose_times(&self, b: &MatrixXd) -> MatrixXd {
        require_same_len(self.rows(), b.rows());
        let mut r = MatrixXd::zeros(self.cols(), b.cols());
        for j in 0..b.cols() {
            let bj = b.column(j);
            for i in 0..self.cols() {
                r[(i, j)] = dot(self.column(i), bj);
            }
        }
        r
    }

    /// `self^T * v` without forming the transpose.
    pub fn transpose_times_vector(&self, v: &VectorXd) -> VectorXd {
        require_same_len(self.rows(), v.len());
        VectorXd::from_vec(
            (0..self.cols())
                .map(|i| dot(self.column(i), v.as_slice()))
                .collect(),
        )
    }

    /// The Gram matrix `self^T * self` (cols x cols). Each coefficient is computed once and
    /// mirrored, so the result is exactly symmetric.
    pub fn transpose_times_self(&self) -> MatrixXd {
        let n = self.cols();
        let mut r = MatrixXd::zeros(n, n);
        for j in 0..n {
            let cj = self.column(j);
            for i in j..n {
                let value = dot(self.column(i), cj);
                r[(i, j)] = value;
                r[(j, i)] = value;
            }
        }
        r
    }

    /// Converts to a `Matrix2d`; panics unless the shape is 2x2.
    pub fn to_matrix2d(&self) -> Matrix2d {
        self.require_shape(2, 2);
        Matrix2d::from_column_major(self.as_slice().try_into().expect("2x2"))
    }

    /// Converts to a `Matrix3d`; panics unless the shape is 3x3.
    pub fn to_matrix3d(&self) -> Matrix3d {
        self.require_shape(3, 3);
        Matrix3d::from_column_major(self.as_slice().try_into().expect("3x3"))
    }

    /// Converts to a `Matrix3x4d`; panics unless the shape is 3x4.
    pub fn to_matrix3x4d(&self) -> Matrix3x4d {
        self.require_shape(3, 4);
        Matrix3x4d::from_column_major(self.as_slice().try_into().expect("3x4"))
    }

    /// Converts to a `Matrix4d`; panics unless the shape is 4x4.
    pub fn to_matrix4d(&self) -> Matrix4d {
        self.require_shape(4, 4);
        Matrix4d::from_column_major(self.as_slice().try_into().expect("4x4"))
    }

    /// Converts to a `Matrix6d`; panics unless the shape is 6x6.
    pub fn to_matrix6d(&self) -> Matrix6d {
        self.require_shape(6, 6);
        Matrix6d::from_column_major(self.as_slice().try_into().expect("6x6"))
    }

    fn require_shape(&self, rows: usize, cols: usize) {
        assert!(
            self.rows() == rows && self.cols() == cols,
            "Expected a {rows}x{cols} matrix, got {}x{}.",
            self.rows(),
            self.cols()
        );
    }
}

macro_rules! from_fixed {
    ($t:ident) => {
        impl From<$t> for MatrixXd {
            fn from(m: $t) -> Self {
                MatrixXd::from_column_major($t::ROWS, $t::COLS, m.as_slice())
            }
        }
    };
}

from_fixed!(Matrix2d);
from_fixed!(Matrix3d);
from_fixed!(Matrix3x4d);
from_fixed!(Matrix4d);
from_fixed!(Matrix6d);

impl std::ops::Add for &MatrixXd {
    type Output = MatrixXd;
    fn add(self, b: &MatrixXd) -> MatrixXd {
        self.zip_with(b, |x, y| x + y)
    }
}

impl std::ops::Sub for &MatrixXd {
    type Output = MatrixXd;
    fn sub(self, b: &MatrixXd) -> MatrixXd {
        self.zip_with(b, |x, y| x - y)
    }
}

impl std::ops::Neg for &MatrixXd {
    type Output = MatrixXd;
    /// Negation; the same bits as colmap-sharp's `a * -1.0` for every non-NaN coefficient.
    fn neg(self) -> MatrixXd {
        self.map_with(|x| -x)
    }
}

impl std::ops::Mul<f64> for &MatrixXd {
    type Output = MatrixXd;
    fn mul(self, s: f64) -> MatrixXd {
        self.map_with(|x| x * s)
    }
}

impl std::ops::Mul<&MatrixXd> for f64 {
    type Output = MatrixXd;
    fn mul(self, m: &MatrixXd) -> MatrixXd {
        m * self
    }
}

impl std::ops::Div<f64> for &MatrixXd {
    type Output = MatrixXd;
    /// Divides each coefficient (no reciprocal), like Eigen.
    fn div(self, s: f64) -> MatrixXd {
        self.map_with(|x| x / s)
    }
}

impl std::ops::Mul for &MatrixXd {
    type Output = MatrixXd;
    /// Matrix product; each coefficient is a left-to-right sum over k from the k = 0 term.
    fn mul(self, b: &MatrixXd) -> MatrixXd {
        require_same_len(self.cols(), b.rows());
        let mut out = vec![0.0; self.rows() * b.cols()];
        if self.cols() > 0 {
            super::product(
                self.as_slice(),
                b.as_slice(),
                self.rows(),
                self.cols(),
                b.cols(),
                &mut out,
            );
        }
        MatrixXd::from_column_major_vec(self.rows(), b.cols(), out)
    }
}

impl std::ops::Mul<&VectorXd> for &MatrixXd {
    type Output = VectorXd;
    /// Matrix-vector product; each coefficient is a left-to-right sum over k.
    fn mul(self, v: &VectorXd) -> VectorXd {
        require_same_len(self.cols(), v.len());
        let mut out = vec![0.0; self.rows()];
        if self.cols() > 0 {
            super::product(
                self.as_slice(),
                v.as_slice(),
                self.rows(),
                self.cols(),
                1,
                &mut out,
            );
        }
        VectorXd::from_vec(out)
    }
}
