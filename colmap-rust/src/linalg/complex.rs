//! Complex: a minimal double-precision complex number, the replacement for the
//! `System.Numerics.Complex` that colmap-sharp's `EigenSolver` (and its tests) use for
//! eigenvalues and eigenvectors (`std::complex<double>` in Eigen's `EigenSolver`).
//!
//! Only the operations [`EigenSolver`](super::EigenSolver) needs are here, each written to
//! give the same bits as the .NET operator colmap-sharp calls on finite operands:
//! - `+`, `-`, unary `-`, [`Complex::conj`] are componentwise.
//! - `Complex * Complex` is `(ac - bd) + (bc + ad) i` in that operand order.
//! - `Complex * f64`, `f64 * Complex`, `Complex / f64` scale each part; `Complex - f64` and
//!   `f64 - Complex` touch the real part only (the latter negates the imaginary part, so a
//!   `+0.0` imaginary part becomes `-0.0`, as in .NET).
//! - `Complex / Complex` is Smith's algorithm, branching on `|d| < |c|` exactly like .NET's
//!   `operator /`.
//! - [`Complex::abs`] is .NET's `Complex.Abs`: `large * sqrt(1 + (small/large)^2)`.
//!
//! .NET's scalar operators special-case non-finite operands; the solver never reaches them (it
//! rejects non-finite input up front), so they are not reproduced. Eigen (MPL-2.0) is not
//! ported. Tests: `colmap-rust/tests/linalg/rust_only_spectral_eigen.rs`.

use crate::math::fns;

/// A complex number `re + im i`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// `0 + 0i`.
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    /// `re + im i`.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// The real number `re + 0i` (.NET's implicit `double -> Complex`).
    pub const fn from_real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// The complex conjugate `re - im i`.
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// `|z|`, .NET's `Complex.Abs` (overflow-safe hypot).
    pub fn abs(self) -> f64 {
        let a = self.re.abs();
        let b = self.im.abs();
        let (small, large) = if a < b { (a, b) } else { (b, a) };
        if small == 0.0 {
            large
        } else if large == f64::INFINITY && !small.is_nan() {
            f64::INFINITY
        } else {
            let ratio = small / large;
            large * fns::sqrt(1.0 + ratio * ratio)
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, o: Complex) {
        *self = *self + o;
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.im * o.re + self.re * o.im,
        )
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Complex;
    fn mul(self, s: f64) -> Complex {
        Complex::new(self.re * s, self.im * s)
    }
}

impl std::ops::MulAssign<f64> for Complex {
    fn mul_assign(&mut self, s: f64) {
        *self = *self * s;
    }
}

impl std::ops::Mul<Complex> for f64 {
    type Output = Complex;
    fn mul(self, z: Complex) -> Complex {
        Complex::new(self * z.re, self * z.im)
    }
}

impl std::ops::Div<f64> for Complex {
    type Output = Complex;
    fn div(self, s: f64) -> Complex {
        Complex::new(self.re / s, self.im / s)
    }
}

impl std::ops::Sub<f64> for Complex {
    type Output = Complex;
    fn sub(self, s: f64) -> Complex {
        Complex::new(self.re - s, self.im)
    }
}

impl std::ops::Sub<Complex> for f64 {
    type Output = Complex;
    fn sub(self, z: Complex) -> Complex {
        Complex::new(self - z.re, -z.im)
    }
}

impl std::ops::Div for Complex {
    type Output = Complex;
    /// Smith's formula, .NET's branch structure and operation order.
    fn div(self, o: Complex) -> Complex {
        let (a, b, c, d) = (self.re, self.im, o.re, o.im);
        if d.abs() < c.abs() {
            let doc = d / c;
            Complex::new((a + b * doc) / (c + d * doc), (b - a * doc) / (c + d * doc))
        } else {
            let cod = c / d;
            Complex::new(
                (b + a * cod) / (d + c * cod),
                (-a + b * cod) / (d + c * cod),
            )
        }
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{}; {}>", self.re, self.im)
    }
}

/// A dense column-major matrix of [`Complex`] (colmap-sharp's `Complex[row, column]`), the
/// eigenvector matrix of [`EigenSolver`](super::EigenSolver). A coefficient is `m[(row, col)]`;
/// `column(j)` is column j as a slice.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexMatrixXd {
    rows: usize,
    cols: usize,
    data: Vec<Complex>,
}

impl ComplexMatrixXd {
    /// The rows x cols zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex::ZERO; rows * cols],
        }
    }

    /// The rows x cols matrix with every coefficient `value`.
    pub fn filled(rows: usize, cols: usize, value: Complex) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The coefficients, column-major.
    pub fn as_slice(&self) -> &[Complex] {
        &self.data
    }

    /// Column j.
    pub fn column(&self, j: usize) -> &[Complex] {
        &self.data[j * self.rows..(j + 1) * self.rows]
    }
}

impl std::ops::Index<(usize, usize)> for ComplexMatrixXd {
    type Output = Complex;
    fn index(&self, (row, col): (usize, usize)) -> &Complex {
        assert!(row < self.rows && col < self.cols, "index out of range");
        &self.data[col * self.rows + row]
    }
}

impl std::ops::IndexMut<(usize, usize)> for ComplexMatrixXd {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Complex {
        assert!(row < self.rows && col < self.cols, "index out of range");
        &mut self.data[col * self.rows + row]
    }
}
