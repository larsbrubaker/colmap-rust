// Rust-only (COLMAP tests its sparse solvers only through sparse_cholesky_test.cc and
// least_absolute_deviations_test.cc, ported in tests/optim/): SimplicialCholesky and the AMD
// ordering in colmap_rust::linalg. Port of colmap-sharp's `ColmapSharp.Tests/LinearAlgebra/
// SimplicialCholeskyTests.cs`. Solutions are compared with the dense LLT on SPD matrices
// shaped like COLMAP's systems (2D grid Laplacians and a bundle-adjustment-like block-arrow
// normal matrix), for both factorization kinds, both orderings and both stored triangles. AMD
// is checked to be a permutation and to cut the fill of a grid Laplacian far below the
// natural ordering's, and a 100k-unknown Laplacian must factor with bounded fill and bounded
// ordering and factorization work, counted through `ordering_work()` and
// `factorization_work()` instead of timed, so machine load cannot fail it. Tier B.

use colmap_rust::linalg::amd_ordering::amd_ordering_of_graph;
use colmap_rust::linalg::{
    ComputationInfo, Llt, SimplicialCholesky, SimplicialCholeskyKind, SparseMatrixCsc,
    SparseOrdering, SparseTriplet, SymmetricPart, VectorXd,
};

use crate::net_random::NetRandom;
use crate::rust_only_sparse_matrix::random_sparse;

/// 5-point Laplacian of a width x height grid plus `shift` on the diagonal (SPD for
/// shift > 0).
fn grid_laplacian(width: usize, height: usize, shift: f64) -> SparseMatrixCsc {
    let mut triplets = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let mut diag = shift;
            let mut link = |j: usize| {
                triplets.push(SparseTriplet::new(i, j, -1.0));
                diag += 1.0;
            };
            if x > 0 {
                link(i - 1);
            }
            if x < width - 1 {
                link(i + 1);
            }
            if y > 0 {
                link(i - width);
            }
            if y < height - 1 {
                link(i + width);
            }
            triplets.push(SparseTriplet::new(i, i, diag));
        }
    }
    let n = width * height;
    SparseMatrixCsc::from_triplets(n, n, &triplets).unwrap()
}

/// Normal matrix J^T J + I of a synthetic BA problem: 6 parameters per camera, 3 per point,
/// every observation couples one camera with one point (a block-arrow pattern once points are
/// eliminated first, which AMD should discover).
fn block_arrow_normal_matrix(
    random: &mut NetRandom,
    cameras: usize,
    points: usize,
    observations_per_point: usize,
) -> SparseMatrixCsc {
    let n = 6 * cameras + 3 * points;
    let mut triplets = Vec::new();
    let mut row = 0;
    for p in 0..points {
        for _ in 0..observations_per_point {
            let c = random.next(cameras as i32) as usize;
            for _ in 0..2 {
                for k in 0..6 {
                    triplets.push(SparseTriplet::new(
                        row,
                        6 * c + k,
                        random.next_double() * 2.0 - 1.0,
                    ));
                }
                for k in 0..3 {
                    triplets.push(SparseTriplet::new(
                        row,
                        6 * cameras + 3 * p + k,
                        random.next_double() * 2.0 - 1.0,
                    ));
                }
                row += 1;
            }
        }
    }
    let jacobian = SparseMatrixCsc::from_triplets(row, n, &triplets).unwrap();
    jacobian.transpose_times_self().add_to_diagonal(1.0)
}

fn random_vector(random: &mut NetRandom, n: usize) -> VectorXd {
    VectorXd::from_vec((0..n).map(|_| random.next_double() * 2.0 - 1.0).collect())
}

fn relative_error(x: &VectorXd, reference: &VectorXd) -> f64 {
    (x - reference).norm() / reference.norm()
}

fn assert_matches_dense(a: &SparseMatrixCsc, random: &mut NetRandom) {
    let b = random_vector(random, a.rows());
    let reference = Llt::new(&a.to_dense()).solve(&b);
    for kind in [SimplicialCholeskyKind::Llt, SimplicialCholeskyKind::Ldlt] {
        for ordering in [SparseOrdering::Amd, SparseOrdering::Natural] {
            for part in [SymmetricPart::Lower, SymmetricPart::Upper] {
                // Only the named triangle is given, so reading the other one would fail.
                let stored = a.triangular_part(part);
                let mut solver = SimplicialCholesky::new(kind, ordering, part);
                solver.compute(&stored).unwrap();
                assert_eq!(solver.info(), ComputationInfo::Success);
                assert!(relative_error(&solver.solve(&b).unwrap(), &reference) < 1e-10);
            }
        }
    }
}

#[test]
fn rust_only_grid_laplacian_matches_dense_llt() {
    assert_matches_dense(&grid_laplacian(9, 7, 0.1), &mut NetRandom::new(1));
}

#[test]
fn rust_only_block_arrow_normal_matrix_matches_dense_llt() {
    let mut random = NetRandom::new(2);
    let a = block_arrow_normal_matrix(&mut random, 5, 20, 3);
    assert_matches_dense(&a, &mut random);
}

#[test]
fn rust_only_random_spd_matches_dense_llt() {
    let mut random = NetRandom::new(3);
    for _ in 0..5 {
        let j = random_sparse(&mut random, 40, 30, 0.08);
        assert_matches_dense(&j.transpose_times_self().add_to_diagonal(0.5), &mut random);
    }
}

#[test]
fn rust_only_amd_is_permutation_and_cuts_fill_on_grid_laplacian() {
    let a = grid_laplacian(40, 40, 1.0);
    let mut amd = SimplicialCholesky::new(
        SimplicialCholeskyKind::Llt,
        SparseOrdering::Amd,
        SymmetricPart::Lower,
    );
    let mut natural = SimplicialCholesky::new(
        SimplicialCholeskyKind::Llt,
        SparseOrdering::Natural,
        SymmetricPart::Lower,
    );
    amd.analyze_pattern(&a).unwrap();
    natural.analyze_pattern(&a).unwrap();

    let mut sorted = amd.permutation().to_vec();
    sorted.sort_unstable();

    // Natural order on a 40x40 grid fills the whole band of width 40 (about 64k entries);
    // minimum degree orderings land near 20k.
    assert_eq!(sorted, (0..a.rows()).collect::<Vec<_>>());
    assert!(natural.non_zeros_l() > 60_000);
    assert!(amd.non_zeros_l() < natural.non_zeros_l() / 2);
}

#[test]
fn rust_only_amd_orders_isolated_and_disconnected_nodes() {
    // Two components plus an isolated node; every node must appear exactly once.
    let adjacency: Vec<Vec<usize>> =
        vec![vec![1], vec![0, 2], vec![1], vec![], vec![5, 5, 4], vec![4]];
    let mut perm = amd_ordering_of_graph(&adjacency).unwrap();
    perm.sort_unstable();
    assert_eq!(perm, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn rust_only_factorize_reuses_symbolic_analysis_for_new_values() {
    let mut random = NetRandom::new(4);
    let a1 = block_arrow_normal_matrix(&mut random, 4, 15, 3);
    let mut a2 = a1.clone();
    for value in a2.values_mut() {
        *value *= 1.5;
    }
    let a3 = a2.add_to_diagonal(2.0);
    let mut solver = SimplicialCholesky::with_kind(SimplicialCholeskyKind::Ldlt);
    solver.analyze_pattern(&a1).unwrap();
    let b = random_vector(&mut random, a1.rows());

    assert!(solver.factorize(&a1).unwrap());
    assert!(
        relative_error(
            &solver.solve(&b).unwrap(),
            &Llt::new(&a1.to_dense()).solve(&b)
        ) < 1e-10
    );
    assert!(solver.factorize(&a2).unwrap());
    assert!(
        relative_error(
            &solver.solve(&b).unwrap(),
            &Llt::new(&a2.to_dense()).solve(&b)
        ) < 1e-10
    );

    // a1 already stores its whole diagonal, so add_to_diagonal keeps the pattern.
    assert!(solver.factorize(&a3).unwrap());
    assert!(
        relative_error(
            &solver.solve(&b).unwrap(),
            &Llt::new(&a3.to_dense()).solve(&b)
        ) < 1e-10
    );
    assert!(solver.factorize(&grid_laplacian(2, 2, 1.0)).is_err());
}

#[test]
fn rust_only_ldlt_solves_indefinite_and_llt_rejects_it() {
    // [[1, 2], [2, 1]] has eigenvalues 3 and -1.
    let a = SparseMatrixCsc::from_triplets(
        2,
        2,
        &[
            SparseTriplet::new(0, 0, 1.0),
            SparseTriplet::new(1, 0, 2.0),
            SparseTriplet::new(0, 1, 2.0),
            SparseTriplet::new(1, 1, 1.0),
        ],
    )
    .unwrap();
    let mut ldlt = SimplicialCholesky::with_kind(SimplicialCholeskyKind::Ldlt);
    ldlt.compute(&a).unwrap();
    let mut llt = SimplicialCholesky::with_kind(SimplicialCholeskyKind::Llt);
    llt.compute(&a).unwrap();
    let x = ldlt.solve(&VectorXd::from_slice(&[5.0, 4.0])).unwrap();
    assert_eq!(ldlt.info(), ComputationInfo::Success);
    assert!((x[0] - 1.0).abs() <= 1e-14);
    assert!((x[1] - 2.0).abs() <= 1e-14);
    assert_eq!(llt.info(), ComputationInfo::NumericalIssue);
    assert!(llt.solve(&VectorXd::from_slice(&[5.0, 4.0])).is_err());
}

// Guards against pathological cost at COLMAP's scale, counted rather than timed. The three
// counts are deterministic for this input and each catches one way the solve can go
// quadratic: the fill nnz(L) (a lost or broken ordering), the AMD ordering's own work (an
// element list or supervariable bucket that keeps growing), and the numeric factorization's
// inner-loop steps, which follow from the fill.
#[test]
fn rust_only_performance_100k_unknown_grid_laplacian_factors_with_bounded_work() {
    let a = grid_laplacian(316, 316, 1e-3);
    let b = random_vector(&mut NetRandom::new(5), a.rows());
    let mut solver = SimplicialCholesky::with_kind(SimplicialCholeskyKind::Llt);
    solver.compute(&a).unwrap();
    let x = solver.solve(&b).unwrap();
    println!(
        "100k Laplacian: nnz(L) = {}, ordering work {}, factorization work {}",
        solver.non_zeros_l(),
        solver.ordering_work(),
        solver.factorization_work()
    );

    assert_eq!(solver.info(), ComputationInfo::Success);
    assert!((&(&a * &x) - &b).norm() / b.norm() < 1e-10);
    // AMD gives 2,971,294 here; the natural (banded) ordering fills about n * 316 = 3.2e7.
    assert!(solver.non_zeros_l() < 4_000_000);
    // 6,651,849 measured in colmap-sharp, about 13 visits per stored entry of A (498,016);
    // linear in the input. A quadratic ordering would be on the order of n^2 = 1e10.
    assert!(solver.ordering_work() < 30 * a.non_zeros() as u64);
    // 216,199,331 measured in colmap-sharp (about 7 n^1.5, the grid's nested-dissection
    // rate); the natural ordering's band would take about n * 316^2 = 1e10.
    assert!(solver.factorization_work() < 300_000_000);
}

// The same input factored by colmap-sharp's SimplicialCholesky reports exactly these counts
// (its test comments record them). They are deterministic functions of the ordering and the
// elimination, so a difference means one of the two ports diverged.
#[test]
fn rust_only_100k_grid_laplacian_work_counts_match_colmap_sharp() {
    let a = grid_laplacian(316, 316, 1e-3);
    let mut solver = SimplicialCholesky::with_kind(SimplicialCholeskyKind::Llt);
    solver.compute(&a).unwrap();
    assert_eq!(solver.non_zeros_l(), 2_971_294);
    assert_eq!(solver.ordering_work(), 6_651_849);
    assert_eq!(solver.factorization_work(), 216_199_331);
}
