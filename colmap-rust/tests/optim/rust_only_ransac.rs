// Rust-only tests (no COLMAP counterpart) of colmap_rust::optim::{Ransac, LoRansac}. They
// run the checks of ransac_test.cc's / loransac_test.cc's SimilarityTransform,
// ParallelSimilarityTransform and ReproducibilityWithRandomSeed cases (ValidateReport: success,
// trials, exact inlier set, model within 1e-6) on the line estimators of test_estimators.rs,
// until SimilarityTransformEstimator<3> lands in Phase 6 and the 1:1 cases replace them. They
// also pin colmap-sharp's two C#-only RANSAC checks (docs/CPP_DIVERGENCES.md entries 140 and
// 141), COLMAP's trial counting, and LO-RANSAC's local refit, both
// through `EstimateAsLocal` and with a Refine-only `LocalEstimator`. Tier C (outcome).

use crate::test_estimators::{
    fit_line_lsq, generate_line_data, Line, LineEstimator, LineLsqEstimator, LineRefiner,
    LineTestData,
};
use colmap_rust::math::random::{default_prng_seed, set_prng_seed};
use colmap_rust::optim::{
    CombinationSampler, EstimateAsLocal, InlierSupport, InlierSupportMeasurer, LoRansac,
    ProgressiveSampler, Ransac, RansacOptions, RansacReport,
};

fn validate_report(report: &RansacReport<Line, InlierSupport>, data: &LineTestData) {
    assert!(report.success);
    assert!(report.num_trials > 0);

    assert_eq!(
        report.support.num_inliers,
        data.num_samples - data.num_outliers
    );
    for i in 0..data.num_samples {
        assert_eq!(report.inlier_mask[i], i >= data.num_outliers, "sample {i}");
    }

    let diff_a = data.expected.a - report.model.a;
    let diff_b = data.expected.b - report.model.b;
    assert!((diff_a * diff_a + diff_b * diff_b).sqrt() < 1e-6);
}

fn options(num_threads: i32, random_seed: i32) -> RansacOptions {
    RansacOptions {
        max_error: 10.0,
        random_seed,
        num_threads,
        ..Default::default()
    }
}

#[test]
fn rust_only_ransac_line_fit() {
    set_prng_seed(0);
    let data = generate_line_data(1000, 400, 0.0);
    let mut ransac =
        Ransac::<LineEstimator>::new(&options(1, default_prng_seed()), LineEstimator).unwrap();
    let report = ransac.estimate(&data.x, &data.y).unwrap();
    validate_report(&report, &data);
}

#[test]
fn rust_only_ransac_parallel_line_fit() {
    set_prng_seed(0);
    let data = generate_line_data(1000, 400, 0.0);
    let mut ransac =
        Ransac::<LineEstimator>::new(&options(4, default_prng_seed()), LineEstimator).unwrap();
    let report = ransac.estimate(&data.x, &data.y).unwrap();
    validate_report(&report, &data);
}

#[test]
fn rust_only_ransac_reproducibility_with_random_seed() {
    set_prng_seed(0);
    // Noisy inliers, so different minimal samples give different models.
    let data = generate_line_data(1000, 400, 0.1);

    let options1 = options(1, 42);
    let report1 = Ransac::<LineEstimator>::new(&options1, LineEstimator)
        .unwrap()
        .estimate(&data.x, &data.y)
        .unwrap();

    let mut options2 = options1;
    let report2 = Ransac::<LineEstimator>::new(&options2, LineEstimator)
        .unwrap()
        .estimate(&data.x, &data.y)
        .unwrap();

    assert!(report1.success);
    assert!(report2.success);

    // Results should be exactly the same.
    assert_eq!(report1.support.num_inliers, report2.support.num_inliers);
    assert_eq!(report1.inlier_mask, report2.inlier_mask);
    assert_eq!(report1.model, report2.model);

    // Now change the seed.
    options2.random_seed = 123;
    let report3 = Ransac::<LineEstimator>::new(&options2, LineEstimator)
        .unwrap()
        .estimate(&data.x, &data.y)
        .unwrap();

    assert!(report3.success);

    // Results should now differ.
    assert_ne!(report1.model, report3.model);
}

#[test]
fn rust_only_loransac_line_fit() {
    set_prng_seed(0);
    let data = generate_line_data(1000, 400, 0.0);
    let mut loransac = LoRansac::<LineEstimator, EstimateAsLocal<LineLsqEstimator>>::new(
        &options(1, default_prng_seed()),
        LineEstimator,
        EstimateAsLocal(LineLsqEstimator),
    )
    .unwrap();
    let report = loransac.estimate(&data.x, &data.y).unwrap();
    validate_report(&report, &data);
}

#[test]
fn rust_only_loransac_parallel_line_fit() {
    set_prng_seed(0);
    let data = generate_line_data(1000, 400, 0.0);
    let mut loransac = LoRansac::<LineEstimator, EstimateAsLocal<LineLsqEstimator>>::new(
        &options(4, default_prng_seed()),
        LineEstimator,
        EstimateAsLocal(LineLsqEstimator),
    )
    .unwrap();
    let report = loransac.estimate(&data.x, &data.y).unwrap();
    validate_report(&report, &data);
}

#[test]
fn rust_only_loransac_returns_local_refit_on_noisy_data() {
    // With noisy inliers, the least-squares refit on the inliers beats any two-point model's
    // residual sum, so LO-RANSAC must report the least-squares line over the true inliers.
    set_prng_seed(0);
    let data = generate_line_data(1000, 400, 0.1);
    let mut loransac = LoRansac::<LineEstimator, EstimateAsLocal<LineLsqEstimator>>::new(
        &options(1, 7),
        LineEstimator,
        EstimateAsLocal(LineLsqEstimator),
    )
    .unwrap();
    let report = loransac.estimate(&data.x, &data.y).unwrap();
    assert!(report.success);
    assert_eq!(report.support.num_inliers, 600);
    let expected = fit_line_lsq(&data.x[400..], &data.y[400..]).unwrap();
    assert_eq!(report.model, expected);

    let plain = Ransac::<LineEstimator>::new(&options(1, 7), LineEstimator)
        .unwrap()
        .estimate(&data.x, &data.y)
        .unwrap();
    assert!(report.support.residual_sum < plain.support.residual_sum);
}

#[test]
fn rust_only_ransac_counts_the_final_trial_increment() {
    // Five points with no three collinear and a tiny max error: every model has exactly its
    // two sample points as inliers, so the dynamic trial count (132 for 2 of 5 inliers)
    // never stops the loop before CombinationSampler's 10 combinations run out. COLMAP's
    // `trial_counter.fetch_add` that ends the loop is counted too, so num_trials is 11.
    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let y = [0.0, 1.0, 4.0, 9.0, 16.0];
    let opts = RansacOptions {
        max_error: 1e-3,
        ..Default::default()
    };
    let mut ransac = Ransac::<LineEstimator, InlierSupportMeasurer, CombinationSampler>::new(
        &opts,
        LineEstimator,
    )
    .unwrap();
    let report = ransac.estimate(&x, &y).unwrap();
    assert!(report.success);
    assert_eq!(report.num_trials, 11);
    assert_eq!(report.support.num_inliers, 2);
    // The first combination {0, 1} wins; later ties do not replace it (same inlier count and
    // a residual sum that is not smaller).
    assert_eq!(report.model, Line { a: 1.0, b: 0.0 });
}

#[test]
fn rust_only_progressive_sampler_index_past_end_fails_check() {
    // docs/CPP_DIVERGENCES.md entry 140 (colmap-sharp's
    // CSharpOnly_ProgressiveSamplerIndexPastEndFailsCheck): with exactly MIN_NUM_SAMPLES
    // pairs, PROSAC's first sample includes index total_num_samples; COLMAP reads past the
    // end of the data there, this port fails a check.
    set_prng_seed(0);
    let data = generate_line_data(2, 0, 0.0);
    let mut ransac = Ransac::<LineEstimator, InlierSupportMeasurer, ProgressiveSampler>::new(
        &options(1, -1),
        LineEstimator,
    )
    .unwrap();
    let err = ransac.estimate(&data.x, &data.y).unwrap_err();
    assert!(err.message().contains("Check failed"), "{}", err.message());
}

#[test]
fn rust_only_parallel_requires_random_sampler() {
    // docs/CPP_DIVERGENCES.md entry 141 (colmap-sharp's CSharpOnly_ParallelRequiresRandomSampler):
    // COLMAP rejects num_threads != 1 for any sampler but RandomSampler; the serial port keeps
    // that validation.
    let data = generate_line_data(10, 0, 0.0);
    let mut ransac = Ransac::<LineEstimator, InlierSupportMeasurer, CombinationSampler>::new(
        &options(2, -1),
        LineEstimator,
    )
    .unwrap();
    let err = ransac.estimate(&data.x, &data.y).unwrap_err();
    assert!(
        err.message()
            .contains("Parallel RANSAC only supports RandomSampler"),
        "{}",
        err.message()
    );

    let mut loransac = LoRansac::<
        LineEstimator,
        EstimateAsLocal<LineLsqEstimator>,
        InlierSupportMeasurer,
        CombinationSampler,
    >::new(
        &options(2, -1),
        LineEstimator,
        EstimateAsLocal(LineLsqEstimator),
    )
    .unwrap();
    let err = loransac.estimate(&data.x, &data.y).unwrap_err();
    assert!(err
        .message()
        .contains("Parallel LORANSAC only supports RandomSampler"));
}

#[test]
fn rust_only_ransac_options_check() {
    // RANSACOptions::Check runs in the constructor: max_error must be positive.
    assert!(Ransac::<LineEstimator>::new(&RansacOptions::default(), LineEstimator).is_err());
    let bad_threads = RansacOptions {
        max_error: 1.0,
        num_threads: 0,
        ..Default::default()
    };
    assert!(Ransac::<LineEstimator>::new(&bad_threads, LineEstimator).is_err());
}

#[test]
fn rust_only_loransac_refine_only_local_estimator() {
    // LineRefiner implements only LocalEstimator (like COLMAP's
    // FundamentalMatrixSampsonEstimator, which has Refine + Residuals and no Estimate): LO-RANSAC
    // must hand it the current best model and keep the refined copy.
    set_prng_seed(0);
    let data = generate_line_data(1000, 400, 0.1);
    let refiner = LineRefiner::default();
    let mut loransac =
        LoRansac::<LineEstimator, LineRefiner>::new(&options(1, 7), LineEstimator, refiner.clone())
            .unwrap();
    let report = loransac.estimate(&data.x, &data.y).unwrap();
    assert!(report.success);
    assert_eq!(report.support.num_inliers, 600);
    for i in 0..data.num_samples {
        assert_eq!(report.inlier_mask[i], i >= data.num_outliers, "sample {i}");
    }

    // Every Refine call started from a real model (never the default), and one Gauss-Newton
    // step on this linear problem lands on the least-squares line over the inliers.
    let initial_models = refiner.initial_models.borrow();
    assert!(!initial_models.is_empty());
    assert!(initial_models.iter().all(|m| *m != Line::default()));
    let expected = fit_line_lsq(&data.x[400..], &data.y[400..]).unwrap();
    assert!((report.model.a - expected.a).abs() < 1e-9);
    assert!((report.model.b - expected.b).abs() < 1e-9);
}
