//! The shared shape of every fixed-size matrix in `linalg`: column-major `[f64; R * C]`
//! storage, `(row, col)` indexing, element-wise operators, the Frobenius norm, `is_approx`
//! and the transpose (`matrix_common!`), plus the product impls (`matrix_product!`,
//! `matrix_vector_product!`). The types themselves live in `matrix_small.rs`, `matrix3.rs`,
//! `matrix4.rs` and `matrix6.rs`; the arithmetic helpers are in `linalg/mod.rs`.
//!
//! Written here (colmap-sharp spells each matrix out by hand; the semantics are the same):
//! element-wise operations are one IEEE operation per coefficient, scalar division divides
//! each coefficient (no reciprocal), and products are left-to-right sums from the first term
//! with no FMA (`linalg::product`).

/// Defines a column-major `$r x $c` matrix type `$t` whose transpose type is `$tt`.
macro_rules! matrix_common {
    ($(#[$meta:meta])* $t:ident, $tt:ident, $r:literal, $c:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $t {
            // Column-major: element (row, col) is at col * ROWS + row.
            data: [f64; $r * $c],
        }

        impl $t {
            /// Number of rows.
            pub const ROWS: usize = $r;
            /// Number of columns.
            pub const COLS: usize = $c;

            /// The zero matrix, Eigen's `Zero()`.
            pub const fn zeros() -> Self {
                Self { data: [0.0; $r * $c] }
            }

            /// The matrix with these coefficients in column-major order (Eigen's memory
            /// layout, `Map<const Matrix>`).
            pub const fn from_column_major(data: [f64; $r * $c]) -> Self {
                Self { data }
            }

            /// The coefficients in column-major order, Eigen's `data()`.
            pub const fn as_slice(&self) -> &[f64] {
                &self.data
            }

            /// A copy of the coefficients in column-major order.
            pub const fn to_column_major(self) -> [f64; $r * $c] {
                self.data
            }

            /// The transpose.
            pub fn transpose(self) -> $tt {
                let mut out = [0.0; $r * $c];
                for col in 0..$c {
                    for row in 0..$r {
                        // (row, col) here is (col, row) in the $c x $r transpose.
                        out[row * $c + col] = self.data[col * $r + row];
                    }
                }
                $tt::from_column_major(out)
            }

            /// Frobenius norm, Eigen's `norm()` on a matrix: the squares summed left to right
            /// in column-major order.
            pub fn norm(self) -> f64 {
                crate::linalg::frobenius_norm(&self.data)
            }

            /// Eigen's `isApprox` at [`crate::linalg::DUMMY_PRECISION`].
            pub fn is_approx(self, other: Self) -> bool {
                self.is_approx_with(other, crate::linalg::DUMMY_PRECISION)
            }

            /// Eigen's `isApprox` with the Frobenius norm:
            /// `||a - b|| <= precision * min(||a||, ||b||)`.
            pub fn is_approx_with(self, other: Self, precision: f64) -> bool {
                crate::linalg::is_approx_slices(&self.data, &other.data, precision)
            }

            fn map(self, f: impl Fn(f64) -> f64) -> Self {
                let mut data = self.data;
                for v in &mut data {
                    *v = f(*v);
                }
                Self { data }
            }

            fn zip(self, b: Self, f: impl Fn(f64, f64) -> f64) -> Self {
                let mut data = self.data;
                for (v, w) in data.iter_mut().zip(b.data) {
                    *v = f(*v, w);
                }
                Self { data }
            }
        }

        impl Default for $t {
            /// The zero matrix.
            fn default() -> Self {
                Self::zeros()
            }
        }

        /// Coefficient at `(row, col)`; panics when either is out of range.
        impl std::ops::Index<(usize, usize)> for $t {
            type Output = f64;
            fn index(&self, (row, col): (usize, usize)) -> &f64 {
                assert!(
                    row < $r && col < $c,
                    concat!(stringify!($t), " index ({}, {}) out of range"),
                    row,
                    col
                );
                &self.data[col * $r + row]
            }
        }

        impl std::ops::IndexMut<(usize, usize)> for $t {
            fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
                assert!(
                    row < $r && col < $c,
                    concat!(stringify!($t), " index ({}, {}) out of range"),
                    row,
                    col
                );
                &mut self.data[col * $r + row]
            }
        }

        impl std::ops::Add for $t {
            type Output = Self;
            fn add(self, b: Self) -> Self {
                self.zip(b, |x, y| x + y)
            }
        }

        impl std::ops::Sub for $t {
            type Output = Self;
            fn sub(self, b: Self) -> Self {
                self.zip(b, |x, y| x - y)
            }
        }

        impl std::ops::Neg for $t {
            type Output = Self;
            fn neg(self) -> Self {
                self.map(|x| -x)
            }
        }

        impl std::ops::Mul<f64> for $t {
            type Output = Self;
            fn mul(self, s: f64) -> Self {
                self.map(|x| x * s)
            }
        }

        impl std::ops::Mul<$t> for f64 {
            type Output = $t;
            fn mul(self, a: $t) -> $t {
                a * self
            }
        }

        /// Scalar division divides each coefficient, as Eigen does (no reciprocal).
        impl std::ops::Div<f64> for $t {
            type Output = Self;
            fn div(self, s: f64) -> Self {
                self.map(|x| x / s)
            }
        }

        impl std::ops::AddAssign for $t {
            fn add_assign(&mut self, b: Self) {
                *self = *self + b;
            }
        }

        impl std::ops::SubAssign for $t {
            fn sub_assign(&mut self, b: Self) {
                *self = *self - b;
            }
        }

        impl std::ops::MulAssign<f64> for $t {
            fn mul_assign(&mut self, s: f64) {
                *self = *self * s;
            }
        }

        impl std::ops::DivAssign<f64> for $t {
            fn div_assign(&mut self, s: f64) {
                *self = *self / s;
            }
        }
    };
}

/// `$a ($r x $k) * $b ($k x $c) -> $out ($r x $c)`, left-to-right sums (`linalg::product`).
macro_rules! matrix_product {
    ($a:ident, $b:ident, $out:ident, $r:literal, $k:literal, $c:literal) => {
        impl std::ops::Mul<$b> for $a {
            type Output = $out;
            fn mul(self, b: $b) -> $out {
                let mut out = [0.0; $r * $c];
                crate::linalg::product(self.as_slice(), b.as_slice(), $r, $k, $c, &mut out);
                $out::from_column_major(out)
            }
        }
    };
}

/// `$m ($r x $c) * $vin ($c) -> $vout ($r)`, each row a left-to-right dot product.
macro_rules! matrix_vector_product {
    ($m:ident, $vin:ident, $vout:ident, $r:literal, $c:literal) => {
        impl std::ops::Mul<crate::linalg::$vin> for $m {
            type Output = crate::linalg::$vout;
            fn mul(self, v: crate::linalg::$vin) -> crate::linalg::$vout {
                let mut out = [0.0; $r];
                crate::linalg::product(self.as_slice(), &v.to_array(), $r, $c, 1, &mut out);
                crate::linalg::$vout::from_array(out)
            }
        }
    };
}
