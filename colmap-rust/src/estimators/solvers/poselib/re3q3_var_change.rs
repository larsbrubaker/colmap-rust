//! PoseLib's random change of variables for a near-degenerate `re3q3` system, and the uniform
//! random rotation it and `re3q3_rotation` draw (`PoseLib/misc/re3q3.cc`, BSD-3-Clause,
//! Copyright (c) 2020 Viktor Larsson; see THIRD_PARTY_NOTICES.md; commit
//! fa7280fee27f97aff31ae7f98bab7f583fac7d08), through colmap-sharp's
//! `Estimators/Solvers/PoseLib/Re3q3.cs` (`SolveWithRandomVarChange`, `UnitRandomQuaternion`).
//! Split out of [`super::re3q3`], which calls it.
//!
//! PoseLib draws the change of variables from Eigen's `Quaternion::UnitRandom()` and
//! `setRandom()`, which read the C library's process-global `std::rand()`. Here the draw comes
//! from a fresh mt19937 with a fixed seed on every call, through libc++'s
//! `uniform_real_distribution`, with Shoemake's uniform rotation (Graphics Gems III) for the
//! quaternion, so the result depends only on the input and is thread-safe
//! (docs/CPP_DIVERGENCES.md entries 220 and 221).

use super::re3q3::{refine_3q3, solve};
use crate::linalg::{Quaterniond, Vector3d};
use crate::math::fns;
use crate::math::random::{libcxx, Mt19937};

/// The fixed seed of the random change of variables (docs/CPP_DIVERGENCES.md entry 220).
const RANDOM_VAR_CHANGE_SEED: u32 = 0;

/// `re3q3`'s retry after an affine change of variables `x = R x' + s`: solves the
/// transformed system without a further retry, maps the solutions back and refines them
/// against the original coefficients. Same contract as [`solve`].
pub(super) fn solve_with_random_var_change(coeffs: &[f64], solutions: &mut [f64]) -> usize {
    let mut rng = Mt19937::new(RANDOM_VAR_CHANGE_SEED);
    let rot = unit_random_quaternion(&mut rng).to_rotation_matrix();
    let shift = Vector3d::new(
        libcxx::uniform_real(&mut rng, -1.0, 1.0),
        libcxx::uniform_real(&mut rng, -1.0, 1.0),
        libcxx::uniform_real(&mut rng, -1.0, 1.0),
    )
    .normalized();

    // A is 3 x 4, column-major: A(r, c) = a[r + 3 c].
    let mut a = [0.0; 12];
    a[..9].copy_from_slice(&rot.to_column_major());
    a[9] = shift.x;
    a[10] = shift.y;
    a[11] = shift.z;

    // B is 10 x 10, column-major: B(r, c) = bm[r + 10 c].
    let mut bm = [0.0; 100];
    bm[0] = a[0] * a[0];
    bm[10] = 2.0 * a[0] * a[3];
    bm[20] = 2.0 * a[0] * a[6];
    bm[30] = a[3] * a[3];
    bm[40] = 2.0 * a[3] * a[6];
    bm[50] = a[6] * a[6];
    bm[60] = 2.0 * a[0] * a[9];
    bm[70] = 2.0 * a[3] * a[9];
    bm[80] = 2.0 * a[6] * a[9];
    bm[90] = a[9] * a[9];
    bm[1] = a[0] * a[1];
    bm[11] = a[0] * a[4] + a[3] * a[1];
    bm[21] = a[0] * a[7] + a[6] * a[1];
    bm[31] = a[3] * a[4];
    bm[41] = a[3] * a[7] + a[6] * a[4];
    bm[51] = a[6] * a[7];
    bm[61] = a[0] * a[10] + a[9] * a[1];
    bm[71] = a[3] * a[10] + a[9] * a[4];
    bm[81] = a[6] * a[10] + a[9] * a[7];
    bm[91] = a[9] * a[10];
    bm[2] = a[0] * a[2];
    bm[12] = a[0] * a[5] + a[3] * a[2];
    bm[22] = a[0] * a[8] + a[6] * a[2];
    bm[32] = a[3] * a[5];
    bm[42] = a[3] * a[8] + a[6] * a[5];
    bm[52] = a[6] * a[8];
    bm[62] = a[0] * a[11] + a[9] * a[2];
    bm[72] = a[3] * a[11] + a[9] * a[5];
    bm[82] = a[6] * a[11] + a[9] * a[8];
    bm[92] = a[9] * a[11];
    bm[3] = a[1] * a[1];
    bm[13] = 2.0 * a[1] * a[4];
    bm[23] = 2.0 * a[1] * a[7];
    bm[33] = a[4] * a[4];
    bm[43] = 2.0 * a[4] * a[7];
    bm[53] = a[7] * a[7];
    bm[63] = 2.0 * a[1] * a[10];
    bm[73] = 2.0 * a[4] * a[10];
    bm[83] = 2.0 * a[7] * a[10];
    bm[93] = a[10] * a[10];
    bm[4] = a[1] * a[2];
    bm[14] = a[1] * a[5] + a[4] * a[2];
    bm[24] = a[1] * a[8] + a[7] * a[2];
    bm[34] = a[4] * a[5];
    bm[44] = a[4] * a[8] + a[7] * a[5];
    bm[54] = a[7] * a[8];
    bm[64] = a[1] * a[11] + a[10] * a[2];
    bm[74] = a[4] * a[11] + a[10] * a[5];
    bm[84] = a[7] * a[11] + a[10] * a[8];
    bm[94] = a[10] * a[11];
    bm[5] = a[2] * a[2];
    bm[15] = 2.0 * a[2] * a[5];
    bm[25] = 2.0 * a[2] * a[8];
    bm[35] = a[5] * a[5];
    bm[45] = 2.0 * a[5] * a[8];
    bm[55] = a[8] * a[8];
    bm[65] = 2.0 * a[2] * a[11];
    bm[75] = 2.0 * a[5] * a[11];
    bm[85] = 2.0 * a[8] * a[11];
    bm[95] = a[11] * a[11];
    bm[6] = 0.0;
    bm[16] = 0.0;
    bm[26] = 0.0;
    bm[36] = 0.0;
    bm[46] = 0.0;
    bm[56] = 0.0;
    bm[66] = a[0];
    bm[76] = a[3];
    bm[86] = a[6];
    bm[96] = a[9];
    bm[7] = 0.0;
    bm[17] = 0.0;
    bm[27] = 0.0;
    bm[37] = 0.0;
    bm[47] = 0.0;
    bm[57] = 0.0;
    bm[67] = a[1];
    bm[77] = a[4];
    bm[87] = a[7];
    bm[97] = a[10];
    bm[8] = 0.0;
    bm[18] = 0.0;
    bm[28] = 0.0;
    bm[38] = 0.0;
    bm[48] = 0.0;
    bm[58] = 0.0;
    bm[68] = a[2];
    bm[78] = a[5];
    bm[88] = a[8];
    bm[98] = a[11];
    bm[9] = 0.0;
    bm[19] = 0.0;
    bm[29] = 0.0;
    bm[39] = 0.0;
    bm[49] = 0.0;
    bm[59] = 0.0;
    bm[69] = 0.0;
    bm[79] = 0.0;
    bm[89] = 0.0;
    bm[99] = 1.0;

    let mut coeffs_b = [0.0; 30];
    for k in 0..3 {
        for j in 0..10 {
            let mut sum = 0.0;
            for l in 0..10 {
                sum += coeffs[k + 3 * l] * bm[l + 10 * j];
            }

            coeffs_b[k + 3 * j] = sum;
        }
    }

    let n_sols = solve(&coeffs_b, solutions, false);

    // Revert change of variables
    for k in 0..n_sols {
        let s = rot * Vector3d::new(solutions[3 * k], solutions[3 * k + 1], solutions[3 * k + 2])
            + shift;
        solutions[3 * k] = s.x;
        solutions[3 * k + 1] = s.y;
        solutions[3 * k + 2] = s.z;
    }

    // In some cases the numerics are quite poor after the change of variables, so we do some
    // newton steps with the original coefficients.
    refine_3q3(coeffs, solutions, n_sols);

    n_sols
}

/// A uniformly distributed unit quaternion: K. Shoemake, "Uniform random rotations",
/// Graphics Gems III (1992), from three uniform draws in `[0, 1)`.
pub(super) fn unit_random_quaternion(rng: &mut Mt19937) -> Quaterniond {
    let u1 = libcxx::uniform_real(rng, 0.0, 1.0);
    let u2 = 2.0 * std::f64::consts::PI * libcxx::uniform_real(rng, 0.0, 1.0);
    let u3 = 2.0 * std::f64::consts::PI * libcxx::uniform_real(rng, 0.0, 1.0);
    let a = fns::sqrt(1.0 - u1);
    let b = fns::sqrt(u1);
    Quaterniond::new(
        b * fns::cos(u3),
        a * fns::sin(u2),
        a * fns::cos(u2),
        b * fns::sin(u3),
    )
}
