//! Port of `colmap::LoadSiftFeaturesFromTextFile` (`feature/sift.h` / `sift.cc`), which reads
//! SIFT features from a text file. Port of colmap-sharp's `Feature/SiftFeaturesText.cs`. Kept
//! apart from the extractor and matcher; the feature importer (Phase 11's
//! `controllers/feature_extraction`) is its caller. Tests: `tests/feature/rust_only_sift_text.rs`
//! (COLMAP tests it only through the importer controller).
//!
//! The format:
//!
//! ```text
//! LINE_0:            NUM_FEATURES DIM
//! LINE_1:            X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//! LINE_I:            ...
//! LINE_NUM_FEATURES: X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM
//! ```
//!
//! where the first line specifies the number of features and the descriptor dimensionality
//! followed by one line per feature: X, Y, SCALE, ORIENTATION are of type float and D_J
//! represent the descriptor in the range [0, 255].
//!
//! Tier A (exact). The stream reads (`std::getline`, `line_stream >> value` in the classic
//! locale) go through [`crate::util::line_tokens`], so a token parses as libc++ parses it
//! (`strtof` for floats), except that a token is taken whole and a line that is not UTF-8 is
//! an error (as for model files: the malformed-file divergence in the
//! `scene/reconstruction_io_utils.rs` header). The descriptor rows grow as lines are read
//! rather than being allocated up front from the header's count, so a corrupt count fails on
//! the first missing line instead of exhausting memory.

use std::io::Read;
use std::path::Path;

use super::SIFT_DESCRIPTOR_DIM;
use crate::feature::types::{
    FeatureDescriptors, FeatureExtractorType, FeatureKeypoint, FeatureKeypoints,
};
use crate::linalg::RowMajorMatrix;
use crate::math::utils::TruncateCast;
use crate::scene::reconstruction_io_utils::{open_read, stream_error};
use crate::util::check::{ColmapError, ErrorKind, Result};
use crate::util::line_tokens::{read_lines, LineTokens};
use crate::{check_eq, check_ge, check_le, fatal_throw};

/// Port of `LoadSiftFeaturesFromTextFile`: the keypoints and SIFT descriptors of the text
/// file at `path`. `Err` (COLMAP's `THROW_CHECK`) when the file cannot be opened or is
/// malformed.
pub fn load_sift_features_from_text_file(
    path: &Path,
) -> Result<(FeatureKeypoints, FeatureDescriptors)> {
    let mut file = open_read(path)?;
    load_sift_features_from_text(&mut file)
}

/// [`load_sift_features_from_text_file`] on an already open stream (for hosts without a file
/// system, such as the browser).
pub fn load_sift_features_from_text<R: Read + ?Sized>(
    stream: &mut R,
) -> Result<(FeatureKeypoints, FeatureDescriptors)> {
    let lines = read_lines(stream).map_err(stream_error)?;
    // `std::getline` past the end of the file reads an empty line.
    let mut lines = lines.into_iter();
    let mut next_line = || -> Result<String> {
        String::from_utf8(lines.next().unwrap_or_default()).map_err(|_| {
            ColmapError::new(
                ErrorKind::InvalidArgument,
                "SIFT feature text line is not valid UTF-8",
            )
        })
    };

    let header_line = next_line()?;
    let mut header_line_stream = LineTokens::new(&header_line);
    let header = header_line_stream
        .read_u32()
        .and_then(|num_features| Some((num_features, header_line_stream.read_u64()?)));
    let Some((num_features, dim)) = header else {
        fatal_throw!("Check failed: header_line_stream >> num_features >> dim ");
    };

    check_eq!(
        dim,
        SIFT_DESCRIPTOR_DIM as u64,
        "SIFT features must have kSiftDescriptorDim dimensions"
    );

    let mut keypoints = FeatureKeypoints::new();
    let mut data = Vec::new();
    for _ in 0..num_features {
        let line = next_line()?;
        let mut feature_line_stream = LineTokens::new(&line);

        let mut read4 = || -> Option<[f32; 4]> {
            Some([
                feature_line_stream.read_f32()?,
                feature_line_stream.read_f32()?,
                feature_line_stream.read_f32()?,
                feature_line_stream.read_f32()?,
            ])
        };
        let Some([x, y, scale, orientation]) = read4() else {
            fatal_throw!("Check failed: feature_line_stream >> x >> y >> scale >> orientation ");
        };
        keypoints.push(FeatureKeypoint::from_scale_orientation(
            x,
            y,
            scale,
            orientation,
        )?);

        // Descriptor
        for _ in 0..SIFT_DESCRIPTOR_DIM {
            let Some(value) = feature_line_stream.read_f32() else {
                fatal_throw!("Check failed: feature_line_stream >> value ");
            };
            check_ge!(value, 0.0f32);
            check_le!(value, 255.0f32);
            data.push(TruncateCast::<u8>::truncate_cast(value));
        }
    }

    let descriptors = FeatureDescriptors::new(
        FeatureExtractorType::Sift,
        RowMajorMatrix::from_vec(keypoints.len(), SIFT_DESCRIPTOR_DIM, data),
    );
    Ok((keypoints, descriptors))
}
