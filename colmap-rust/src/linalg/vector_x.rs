//! `VectorXd`: a heap-backed, dynamically sized column vector of doubles, the replacement for
//! `Eigen::VectorXd` (and `Eigen::Matrix<double, Dynamic, 1>`).
//!
//! Port of colmap-sharp's `ColmapSharp/LinearAlgebra/VectorXd.cs` (MIT), written there to
//! Eigen's documented semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Sibling
//! of [`super::MatrixXd`] (`matrix_x.rs`); the fixed-size `Vector2d`/`Vector3d`/`Vector4d`
//! convert to and from it. Arithmetic order follows the module contract in `linalg/mod.rs`:
//! reductions are left-to-right sums seeded with the first term, no FMA.
//!
//! Unlike the C# class, this is an owned value (`Clone` is the explicit deep copy, as there).
//! Length mismatches are programming errors and panic, like Eigen's asserts.
//! Tests: `colmap-rust/tests/linalg/rust_only_dynamic_matrix.rs`.

use super::{dot, is_approx_slices, maxi, Vector2d, Vector3d, Vector4d, DUMMY_PRECISION};
use crate::math::fns;

/// Dynamically sized column vector of doubles, Eigen's `VectorXd`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorXd {
    data: Vec<f64>,
}

impl VectorXd {
    /// A zero vector of length `n`, Eigen's `VectorXd::Zero(n)`.
    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    /// A vector holding a copy of `values`.
    pub fn from_slice(values: &[f64]) -> Self {
        Self {
            data: values.to_vec(),
        }
    }

    /// A vector that takes ownership of `values`.
    pub fn from_vec(values: Vec<f64>) -> Self {
        Self { data: values }
    }

    /// Eigen's `VectorXd::Ones(n)`.
    pub fn ones(n: usize) -> Self {
        Self::constant(n, 1.0)
    }

    /// Eigen's `VectorXd::Constant(n, value)`.
    pub fn constant(n: usize, value: f64) -> Self {
        Self {
            data: vec![value; n],
        }
    }

    /// Eigen's `VectorXd::Unit(n, i)`: zero except a 1 at index `i`.
    pub fn unit(n: usize, i: usize) -> Self {
        let mut v = Self::zeros(n);
        v.data[i] = 1.0;
        v
    }

    /// Number of coefficients, Eigen's `size()`.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True for a zero-length vector.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The coefficients, Eigen's `data()`.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The coefficients, writable.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Consumes the vector and returns its coefficients.
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    /// Converts to a `Vector2d`; panics unless the length is 2.
    pub fn to_vector2d(&self) -> Vector2d {
        self.require_len(2);
        Vector2d::new(self.data[0], self.data[1])
    }

    /// Converts to a `Vector3d`; panics unless the length is 3.
    pub fn to_vector3d(&self) -> Vector3d {
        self.require_len(3);
        Vector3d::new(self.data[0], self.data[1], self.data[2])
    }

    /// Converts to a `Vector4d`; panics unless the length is 4.
    pub fn to_vector4d(&self) -> Vector4d {
        self.require_len(4);
        Vector4d::new(self.data[0], self.data[1], self.data[2], self.data[3])
    }

    /// A copy of coefficients `[start, start + len)`, Eigen's `segment(start, len)`.
    pub fn segment(&self, start: usize, len: usize) -> Self {
        Self::from_slice(&self.data[start..start + len])
    }

    /// A copy of the first `n` coefficients, Eigen's `head(n)`.
    pub fn head(&self, n: usize) -> Self {
        self.segment(0, n)
    }

    /// A copy of the last `n` coefficients, Eigen's `tail(n)`.
    pub fn tail(&self, n: usize) -> Self {
        self.segment(self.len() - n, n)
    }

    /// Squared Euclidean norm, a left-to-right sum of squares.
    pub fn squared_norm(&self) -> f64 {
        dot(&self.data, &self.data)
    }

    /// Euclidean norm, `sqrt(squared_norm())`.
    pub fn norm(&self) -> f64 {
        fns::sqrt(self.squared_norm())
    }

    /// Dot product with a vector of the same length.
    pub fn dot(&self, other: &Self) -> f64 {
        self.require_len(other.len());
        dot(&self.data, &other.data)
    }

    /// This vector divided by its norm. Like Eigen's `normalized()`, a zero vector stays zero.
    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        let mut result = self.clone();
        if norm > 0.0 {
            for v in &mut result.data {
                *v /= norm;
            }
        }
        result
    }

    /// Largest absolute coefficient, Eigen's `lpNorm<Infinity>()` (0 when empty).
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0, |max, v| maxi(max, v.abs()))
    }

    /// Eigen's `isApprox` at [`DUMMY_PRECISION`].
    pub fn is_approx(&self, other: &Self) -> bool {
        self.is_approx_with(other, DUMMY_PRECISION)
    }

    /// Eigen's `isApprox`: `||a - b|| <= precision * min(||a||, ||b||)`; false when the
    /// lengths differ.
    pub fn is_approx_with(&self, other: &Self, precision: f64) -> bool {
        self.len() == other.len() && is_approx_slices(&self.data, &other.data, precision)
    }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    fn zip(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        self.require_len(other.len());
        Self {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }

    fn require_len(&self, len: usize) {
        assert_eq!(
            self.len(),
            len,
            "Vector length mismatch: {} vs {}.",
            self.len(),
            len
        );
    }
}

impl From<Vector2d> for VectorXd {
    fn from(v: Vector2d) -> Self {
        Self::from_slice(&[v.x, v.y])
    }
}

impl From<Vector3d> for VectorXd {
    fn from(v: Vector3d) -> Self {
        Self::from_slice(&[v.x, v.y, v.z])
    }
}

impl From<Vector4d> for VectorXd {
    fn from(v: Vector4d) -> Self {
        Self::from_slice(&[v.x, v.y, v.z, v.w])
    }
}

impl std::ops::Index<usize> for VectorXd {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

impl std::ops::IndexMut<usize> for VectorXd {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}

impl std::ops::Add for &VectorXd {
    type Output = VectorXd;
    fn add(self, b: &VectorXd) -> VectorXd {
        self.zip(b, |x, y| x + y)
    }
}

impl std::ops::Sub for &VectorXd {
    type Output = VectorXd;
    fn sub(self, b: &VectorXd) -> VectorXd {
        self.zip(b, |x, y| x - y)
    }
}

impl std::ops::Neg for &VectorXd {
    type Output = VectorXd;
    /// Negation; the same bits as colmap-sharp's `a * -1.0` for every non-NaN coefficient.
    fn neg(self) -> VectorXd {
        self.map(|x| -x)
    }
}

impl std::ops::Mul<f64> for &VectorXd {
    type Output = VectorXd;
    fn mul(self, s: f64) -> VectorXd {
        self.map(|x| x * s)
    }
}

impl std::ops::Mul<&VectorXd> for f64 {
    type Output = VectorXd;
    fn mul(self, v: &VectorXd) -> VectorXd {
        v * self
    }
}

impl std::ops::Div<f64> for &VectorXd {
    type Output = VectorXd;
    /// Divides each coefficient (no reciprocal), like Eigen.
    fn div(self, s: f64) -> VectorXd {
        self.map(|x| x / s)
    }
}

impl std::fmt::Display for VectorXd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:?}")?;
        }
        write!(f, "]")
    }
}
