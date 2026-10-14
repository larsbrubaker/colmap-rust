// Rust-only tests of colmap_rust::feature::sift::features_text (LoadSiftFeaturesFromTextFile),
// which COLMAP tests only through its feature importer controller (Phase 11): a well-formed
// file reads to its keypoints and truncated descriptor bytes, and every malformed input COLMAP
// rejects with a THROW_CHECK is an InvalidArgument error here.

use std::fmt::Write as _;

use colmap_rust::feature::sift::features_text::{
    load_sift_features_from_text, load_sift_features_from_text_file,
};
use colmap_rust::feature::types::{FeatureExtractorType, FeatureKeypoint};
use colmap_rust::ErrorKind;

/// A text file of `rows`, each `x y scale orientation` followed by 128 descriptor values.
fn sift_text(rows: &[([f32; 4], f32)]) -> String {
    let mut text = format!("{} 128\n", rows.len());
    for (keypoint, value) in rows {
        write!(
            text,
            "{} {} {} {}",
            keypoint[0], keypoint[1], keypoint[2], keypoint[3]
        )
        .unwrap();
        for j in 0..128 {
            write!(text, " {}", if j == 0 { *value } else { j as f32 }).unwrap();
        }
        text.push('\n');
    }
    text
}

#[test]
fn rust_only_load_sift_features_from_text_reads_keypoints_and_descriptors() {
    let text = sift_text(&[
        ([0.32, 0.12, 1.23, 1.0], 254.9),
        ([5.0, 6.0, 2.0, -0.5], 0.0),
    ]);
    let (keypoints, descriptors) = load_sift_features_from_text(&mut text.as_bytes()).unwrap();

    assert_eq!(keypoints.len(), 2);
    assert_eq!(
        keypoints[0],
        FeatureKeypoint::from_scale_orientation(0.32, 0.12, 1.23, 1.0).unwrap()
    );
    assert_eq!(
        keypoints[1],
        FeatureKeypoint::from_scale_orientation(5.0, 6.0, 2.0, -0.5).unwrap()
    );
    assert_eq!(descriptors.extractor_type, FeatureExtractorType::Sift);
    assert_eq!((descriptors.data.rows(), descriptors.data.cols()), (2, 128));
    // TruncateCast<float, uint8_t>: 254.9 truncates to 254.
    assert_eq!(descriptors.data[(0, 0)], 254);
    assert_eq!(descriptors.data[(1, 0)], 0);
    assert_eq!(descriptors.data[(1, 127)], 127);
}

#[test]
fn rust_only_load_sift_features_from_text_file_reads_a_file() {
    let path =
        std::env::temp_dir().join(format!("colmap_rust_sift_text_{}.txt", std::process::id()));
    std::fs::write(&path, sift_text(&[([1.0, 2.0, 3.0, 0.0], 9.0)])).unwrap();
    let result = load_sift_features_from_text_file(&path);
    std::fs::remove_file(&path).unwrap();
    let (keypoints, descriptors) = result.unwrap();
    assert_eq!(keypoints.len(), 1);
    assert_eq!(descriptors.data[(0, 0)], 9);

    let error = load_sift_features_from_text_file(&path).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
}

#[test]
fn rust_only_load_sift_features_from_text_rejects_malformed_input() {
    let reject = |text: &str| {
        let error = load_sift_features_from_text(&mut text.as_bytes()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{text:?}");
        error
    };
    // The header must read, and the dimension must be kSiftDescriptorDim (sift.h's own
    // example file, "2 4", is rejected by COLMAP too).
    reject("");
    reject("two 128\n");
    let error = reject("2 4\n0.32 0.12 1.23 1.0 1 2 3 4\n0.32 0.12 1.23 1.0 1 2 3 4\n");
    assert!(error
        .message()
        .contains("SIFT features must have kSiftDescriptorDim dimensions"));
    // A missing feature line reads empty (std::getline past the end), which fails.
    reject("1 128\n");
    // A short descriptor, and values outside [0, 255].
    let good = sift_text(&[([1.0, 2.0, 3.0, 0.0], 9.0)]);
    reject(&good.replace(" 127\n", "\n"));
    reject(&sift_text(&[([1.0, 2.0, 3.0, 0.0], 255.5)]));
    reject(&sift_text(&[([1.0, 2.0, 3.0, 0.0], -1.0)]));
}
