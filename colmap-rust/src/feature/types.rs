//! Port of `colmap/feature/types.h` / `types.cc`: the feature vocabulary every later stage
//! shares — [`FeatureExtractorType`] and [`FeatureMatcherType`], [`FeatureKeypoint`] (a
//! location with its affine shape), [`FeatureDescriptors`] / [`FeatureDescriptorsFloat`]
//! (descriptor rows tagged with the extractor that made them), [`FeatureMatch`], and the
//! keypoint / match matrix conversions. Built on [`crate::linalg::RowMajorMatrix`] for the
//! Eigen row-major typedefs. Its users are `scene::two_view_geometry` (inlier matches) and,
//! later, SIFT, the matchers and the database. Port of colmap-sharp's
//! `Feature/FeatureKeypoint.cs`, `FeatureDescriptors.cs` and `FeatureMatch.cs`.
//! Tests: `tests/feature/types.rs` (`types_test.cc` 1:1).
//!
//! Tiers:
//! - A (exact): the location/shape bookkeeping (the affine constructor, `Rescale`, `Rot90`,
//!   equality), the descriptor conversions and the matrix conversions of stored fields.
//! - B (tolerance): the trigonometric pieces (the scale/orientation constructor,
//!   `FromShapeParameters`, `ComputeOrientation`, `ComputeShear`, and so `KeypointsToMatrix`)
//!   go through `math::fns`'s `sinf` / `cosf` / `atan2f`, which may differ from the platform
//!   libm in the last ulp (docs/CPP_DIVERGENCES.md entry 1); `types_test.cc` compares them
//!   with a tolerance.
//!
//! Translation notes:
//! - COLMAP's overloaded constructors become named ones: `FeatureKeypoint(x, y)` is
//!   [`FeatureKeypoint::new`], `(x, y, scale, orientation)` is
//!   [`FeatureKeypoint::from_scale_orientation`], `(x, y, a11, a12, a21, a22)` is
//!   [`FeatureKeypoint::from_affine`]; `Rescale(scale_x, scale_y)` is
//!   [`FeatureKeypoint::rescale_xy`]. The default keypoint is `FeatureKeypoint(0, 0)`, the
//!   origin with the identity shape (a manual `Default`, never all-zero).
//! - The `THROW_CHECK`s of the constructors and `Rescale`, and `LOG(FATAL_THROW)` for an
//!   unsupported descriptor type, return `Err`.
//! - The descriptors' `type` field is `extractor_type` (`type` is a Rust keyword).
//! - Converting SIFT descriptors from float casts each value to `u8` with Rust's saturating
//!   `as`, where COLMAP's `static_cast` is undefined out of range (entry 180). The
//!   learned-feature types reinterpret float32 bytes in the platform's byte order, exactly
//!   like COLMAP's `memcpy`.
//! - `KeypointsFromMatrix` / `MatchesFromMatrix` take a [`RowMajorMatrix`] and check its
//!   column count at run time, where Eigen's `Ref<Matrix<.., Dynamic, 4>>` fixes it in the
//!   type (entry 181).

use std::fmt;

use crate::linalg::RowMajorMatrix;
use crate::math::fns;
use crate::util::check::{CheckOpValue, ColmapError, ErrorKind, Result};
use crate::util::types::{Point2DIdx, INVALID_POINT2D_IDX};
use crate::{check_eq, check_ge, check_gt, fatal_throw};

/// Port of `colmap::FeatureExtractorType` (`MAKE_ENUM_CLASS_OVERLOAD_STREAM(
/// FeatureExtractorType, -1, UNDEFINED, SIFT, ALIKED_N16ROT, ALIKED_N32, LOMA_B,
/// LOMA_B128)`), with COLMAP's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(i32)]
pub enum FeatureExtractorType {
    /// `UNDEFINED` (-1).
    #[default]
    Undefined = -1,
    /// `SIFT` (0).
    Sift = 0,
    /// `ALIKED_N16ROT` (1).
    AlikedN16Rot = 1,
    /// `ALIKED_N32` (2).
    AlikedN32 = 2,
    /// `LOMA_B` (3).
    LomaB = 3,
    /// `LOMA_B128` (4).
    LomaB128 = 4,
}

impl FeatureExtractorType {
    /// Every value, in COLMAP's declaration order.
    pub const ALL: [FeatureExtractorType; 6] = [
        FeatureExtractorType::Undefined,
        FeatureExtractorType::Sift,
        FeatureExtractorType::AlikedN16Rot,
        FeatureExtractorType::AlikedN32,
        FeatureExtractorType::LomaB,
        FeatureExtractorType::LomaB128,
    ];

    /// `FeatureExtractorTypeToString`: COLMAP's spelling, e.g. `"ALIKED_N16ROT"`.
    pub fn as_str(self) -> &'static str {
        match self {
            FeatureExtractorType::Undefined => "UNDEFINED",
            FeatureExtractorType::Sift => "SIFT",
            FeatureExtractorType::AlikedN16Rot => "ALIKED_N16ROT",
            FeatureExtractorType::AlikedN32 => "ALIKED_N32",
            FeatureExtractorType::LomaB => "LOMA_B",
            FeatureExtractorType::LomaB128 => "LOMA_B128",
        }
    }

    /// `FeatureExtractorTypeFromString`. Anything but COLMAP's spellings is its
    /// `std::runtime_error("Unknown string value: <v> for enum: FeatureExtractorType")`.
    pub fn from_colmap_str(value: &str) -> Result<FeatureExtractorType> {
        Self::ALL
            .into_iter()
            .find(|t| t.as_str() == value)
            .ok_or_else(|| unknown_enum_value(value, "FeatureExtractorType"))
    }
}

impl fmt::Display for FeatureExtractorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl CheckOpValue for FeatureExtractorType {
    /// `operator<<` (`MAKE_ENUM_CLASS_OVERLOAD_STREAM`): the enum's name, as in a failed
    /// `THROW_CHECK_EQ(descriptors.type, FeatureExtractorType::SIFT)`.
    fn check_op_string(&self) -> String {
        self.as_str().to_owned()
    }
}

/// Port of `colmap::FeatureMatcherType` (`MAKE_ENUM_CLASS_OVERLOAD_STREAM(
/// FeatureMatcherType, -1, UNDEFINED, SIFT_BRUTEFORCE, SIFT_LIGHTGLUE, ALIKED_BRUTEFORCE,
/// ALIKED_LIGHTGLUE, LOMA_BRUTEFORCE, LOMA_B, LOMA_B128, LOMA_R, LOMA_L, LOMA_G)`), with
/// COLMAP's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(i32)]
pub enum FeatureMatcherType {
    /// `UNDEFINED` (-1).
    #[default]
    Undefined = -1,
    /// `SIFT_BRUTEFORCE` (0).
    SiftBruteforce = 0,
    /// `SIFT_LIGHTGLUE` (1).
    SiftLightglue = 1,
    /// `ALIKED_BRUTEFORCE` (2).
    AlikedBruteforce = 2,
    /// `ALIKED_LIGHTGLUE` (3).
    AlikedLightglue = 3,
    /// `LOMA_BRUTEFORCE` (4).
    LomaBruteforce = 4,
    /// `LOMA_B` (5).
    LomaB = 5,
    /// `LOMA_B128` (6).
    LomaB128 = 6,
    /// `LOMA_R` (7).
    LomaR = 7,
    /// `LOMA_L` (8).
    LomaL = 8,
    /// `LOMA_G` (9).
    LomaG = 9,
}

impl FeatureMatcherType {
    /// Every value, in COLMAP's declaration order.
    pub const ALL: [FeatureMatcherType; 11] = [
        FeatureMatcherType::Undefined,
        FeatureMatcherType::SiftBruteforce,
        FeatureMatcherType::SiftLightglue,
        FeatureMatcherType::AlikedBruteforce,
        FeatureMatcherType::AlikedLightglue,
        FeatureMatcherType::LomaBruteforce,
        FeatureMatcherType::LomaB,
        FeatureMatcherType::LomaB128,
        FeatureMatcherType::LomaR,
        FeatureMatcherType::LomaL,
        FeatureMatcherType::LomaG,
    ];

    /// `FeatureMatcherTypeToString`: COLMAP's spelling, e.g. `"SIFT_BRUTEFORCE"`.
    pub fn as_str(self) -> &'static str {
        match self {
            FeatureMatcherType::Undefined => "UNDEFINED",
            FeatureMatcherType::SiftBruteforce => "SIFT_BRUTEFORCE",
            FeatureMatcherType::SiftLightglue => "SIFT_LIGHTGLUE",
            FeatureMatcherType::AlikedBruteforce => "ALIKED_BRUTEFORCE",
            FeatureMatcherType::AlikedLightglue => "ALIKED_LIGHTGLUE",
            FeatureMatcherType::LomaBruteforce => "LOMA_BRUTEFORCE",
            FeatureMatcherType::LomaB => "LOMA_B",
            FeatureMatcherType::LomaB128 => "LOMA_B128",
            FeatureMatcherType::LomaR => "LOMA_R",
            FeatureMatcherType::LomaL => "LOMA_L",
            FeatureMatcherType::LomaG => "LOMA_G",
        }
    }

    /// `FeatureMatcherTypeFromString`. Anything but COLMAP's spellings is its
    /// `std::runtime_error("Unknown string value: <v> for enum: FeatureMatcherType")`.
    pub fn from_colmap_str(value: &str) -> Result<FeatureMatcherType> {
        Self::ALL
            .into_iter()
            .find(|t| t.as_str() == value)
            .ok_or_else(|| unknown_enum_value(value, "FeatureMatcherType"))
    }
}

impl fmt::Display for FeatureMatcherType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `MAKE_ENUM_CLASS`'s `FromString` failure.
fn unknown_enum_value(value: &str, enum_name: &str) -> ColmapError {
    ColmapError::new(
        ErrorKind::RuntimeError,
        format!("Unknown string value: {value} for enum: {enum_name}"),
    )
}

/// Port of `colmap::FeatureKeypoint`: a feature location with its affine shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureKeypoint {
    /// Location x of the feature, with the origin at the upper left image corner, i.e. the
    /// upper left pixel has the coordinate (0.5, 0.5).
    pub x: f32,
    /// Location y of the feature (same origin as `x`).
    pub y: f32,
    /// Affine shape of the feature, element (1, 1).
    pub a11: f32,
    /// Affine shape, element (1, 2).
    pub a12: f32,
    /// Affine shape, element (2, 1).
    pub a21: f32,
    /// Affine shape, element (2, 2).
    pub a22: f32,
}

impl Default for FeatureKeypoint {
    /// `FeatureKeypoint()`: `FeatureKeypoint(0, 0)`, the origin with the identity shape.
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl FeatureKeypoint {
    /// `FeatureKeypoint(x, y)`: a keypoint at `(x, y)` with the identity shape.
    pub const fn new(x: f32, y: f32) -> Self {
        Self::from_affine(x, y, 1.0, 0.0, 0.0, 1.0)
    }

    /// `FeatureKeypoint(x, y, scale, orientation)`: an isotropic shape of the given scale,
    /// rotated by `orientation`. `Err` for a negative (or NaN) scale.
    pub fn from_scale_orientation(x: f32, y: f32, scale: f32, orientation: f32) -> Result<Self> {
        check_ge!(scale, 0.0);
        let scale_cos_orientation = scale * fns::cosf(orientation);
        let scale_sin_orientation = scale * fns::sinf(orientation);
        Ok(Self::from_affine(
            x,
            y,
            scale_cos_orientation,
            -scale_sin_orientation,
            scale_sin_orientation,
            scale_cos_orientation,
        ))
    }

    /// `FeatureKeypoint(x, y, a11, a12, a21, a22)`: a keypoint with the affine shape
    /// `[a11 a12; a21 a22]`.
    pub const fn from_affine(x: f32, y: f32, a11: f32, a12: f32, a21: f32, a22: f32) -> Self {
        Self {
            x,
            y,
            a11,
            a12,
            a21,
            a22,
        }
    }

    /// `FromShapeParameters`: a keypoint from per-axis scales, orientation and shear. `Err`
    /// for a negative (or NaN) scale.
    pub fn from_shape_parameters(
        x: f32,
        y: f32,
        scale_x: f32,
        scale_y: f32,
        orientation: f32,
        shear: f32,
    ) -> Result<Self> {
        check_ge!(scale_x, 0.0);
        check_ge!(scale_y, 0.0);
        Ok(Self::from_affine(
            x,
            y,
            scale_x * fns::cosf(orientation),
            -scale_y * fns::sinf(orientation + shear),
            scale_x * fns::sinf(orientation),
            scale_y * fns::cosf(orientation + shear),
        ))
    }

    /// `Rescale(scale)`: rescales the feature location and shape size by `scale`.
    pub fn rescale(&mut self, scale: f32) -> Result<()> {
        self.rescale_xy(scale, scale)
    }

    /// `Rescale(scale_x, scale_y)`: rescales the location and shape size per axis. `Err`
    /// unless both factors are positive.
    pub fn rescale_xy(&mut self, scale_x: f32, scale_y: f32) -> Result<()> {
        check_gt!(scale_x, 0.0);
        check_gt!(scale_y, 0.0);
        self.x *= scale_x;
        self.y *= scale_y;
        self.a11 *= scale_x;
        self.a12 *= scale_y;
        self.a21 *= scale_x;
        self.a22 *= scale_y;
        Ok(())
    }

    /// `Rot90`: rotates the feature location and shape by `k * 90` degrees
    /// counter-clockwise around the image center. `width` and `height` are the dimensions of
    /// the image the keypoint is currently defined on.
    pub fn rot90(&mut self, k: i32, width: i32, height: i32) {
        // C++'s `%` truncates toward zero like Rust's, so -1 % 4 == -1 here too.
        let mut k = k % 4;
        if k < 0 {
            k += 4;
        }
        if k == 0 {
            return;
        }
        let (mut new_x, mut new_y) = (self.x, self.y);
        let (mut new_a11, mut new_a12, mut new_a21, mut new_a22) =
            (self.a11, self.a12, self.a21, self.a22);
        let w = width as f32;
        let h = height as f32;

        if k == 1 {
            // 90 CCW
            new_x = self.y;
            new_y = w - self.x;
            new_a11 = self.a21;
            new_a12 = self.a22;
            new_a21 = -self.a11;
            new_a22 = -self.a12;
        } else if k == 2 {
            // 180 CCW
            new_x = w - self.x;
            new_y = h - self.y;
            new_a11 = -self.a11;
            new_a12 = -self.a12;
            new_a21 = -self.a21;
            new_a22 = -self.a22;
        } else if k == 3 {
            // 270 CCW
            new_x = h - self.y;
            new_y = self.x;
            new_a11 = -self.a21;
            new_a12 = -self.a22;
            new_a21 = self.a11;
            new_a22 = self.a12;
        }
        self.x = new_x;
        self.y = new_y;
        self.a11 = new_a11;
        self.a12 = new_a12;
        self.a21 = new_a21;
        self.a22 = new_a22;
    }

    /// `ComputeScale`: the mean of the two axis scales.
    pub fn compute_scale(&self) -> f32 {
        (self.compute_scale_x() + self.compute_scale_y()) / 2.0
    }

    /// `ComputeScaleX`: the scale along the first shape axis.
    pub fn compute_scale_x(&self) -> f32 {
        fns::sqrtf(self.a11 * self.a11 + self.a21 * self.a21)
    }

    /// `ComputeScaleY`: the scale along the second shape axis.
    pub fn compute_scale_y(&self) -> f32 {
        fns::sqrtf(self.a12 * self.a12 + self.a22 * self.a22)
    }

    /// `ComputeOrientation`: the orientation of the first shape axis.
    pub fn compute_orientation(&self) -> f32 {
        fns::atan2f(self.a21, self.a11)
    }

    /// `ComputeShear`: the shear angle between the shape axes.
    pub fn compute_shear(&self) -> f32 {
        fns::atan2f(-self.a12, self.a22) - self.compute_orientation()
    }
}

/// `FeatureKeypoints`: `std::vector<FeatureKeypoint>`.
pub type FeatureKeypoints = Vec<FeatureKeypoint>;

/// `FeatureDescriptor`: one descriptor, `Eigen::Matrix<uint8_t, 1, Dynamic, RowMajor>`.
pub type FeatureDescriptor = Vec<u8>;

/// `FeatureDescriptorsData`: one byte descriptor per row.
pub type FeatureDescriptorsData = RowMajorMatrix<u8>;

/// `FeatureDescriptorsFloatData`: one float descriptor per row.
pub type FeatureDescriptorsFloatData = RowMajorMatrix<f32>;

/// The learned-feature types, whose float32 descriptors are stored as raw bytes.
fn stores_float_bytes(extractor_type: FeatureExtractorType) -> bool {
    matches!(
        extractor_type,
        FeatureExtractorType::AlikedN16Rot
            | FeatureExtractorType::AlikedN32
            | FeatureExtractorType::LomaB
            | FeatureExtractorType::LomaB128
    )
}

/// Size of a float32 in bytes, C++'s `sizeof(float)`.
const SIZEOF_FLOAT: usize = std::mem::size_of::<f32>();

/// Port of `colmap::FeatureDescriptors`: byte descriptors, one per row, with the extractor
/// type that produced them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FeatureDescriptors {
    /// `type`: the extractor that produced the descriptors.
    pub extractor_type: FeatureExtractorType,
    /// The descriptors, one per row.
    pub data: FeatureDescriptorsData,
}

impl FeatureDescriptors {
    /// `FeatureDescriptors(type, data)`.
    pub fn new(extractor_type: FeatureExtractorType, data: FeatureDescriptorsData) -> Self {
        Self {
            extractor_type,
            data,
        }
    }

    /// `FromFloat`: SIFT values are cast to `u8` (saturating, entry 180); the learned-feature
    /// float32 values are reinterpreted as bytes. `Err("Unsupported feature type: <type>")`
    /// for any other type.
    pub fn from_float(float_desc: &FeatureDescriptorsFloat) -> Result<FeatureDescriptors> {
        let rows = float_desc.data.rows();
        let float_cols = float_desc.data.cols();
        let data = match float_desc.extractor_type {
            // Cast each float value to uint8. `as` truncates toward zero like static_cast and
            // saturates where static_cast is undefined (out of range, NaN).
            FeatureExtractorType::Sift => RowMajorMatrix::from_vec(
                rows,
                float_cols,
                float_desc
                    .data
                    .as_slice()
                    .iter()
                    .map(|&v| v as u8)
                    .collect(),
            ),
            // Reinterpret float32 data as uint8 bytes, in memory (native) byte order.
            t if stores_float_bytes(t) => RowMajorMatrix::from_vec(
                rows,
                float_cols * SIZEOF_FLOAT,
                float_desc
                    .data
                    .as_slice()
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect(),
            ),
            t => fatal_throw!("Unsupported feature type: {}", t),
        };
        Ok(FeatureDescriptors::new(float_desc.extractor_type, data))
    }

    /// `ToFloat`: [`FeatureDescriptorsFloat::from_bytes`] of these descriptors.
    pub fn to_float(&self) -> Result<FeatureDescriptorsFloat> {
        FeatureDescriptorsFloat::from_bytes(self)
    }
}

/// Port of `colmap::FeatureDescriptorsFloat`: float descriptors, one per row, with the
/// extractor type that produced them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FeatureDescriptorsFloat {
    /// `type`: the extractor that produced the descriptors.
    pub extractor_type: FeatureExtractorType,
    /// The descriptors, one per row.
    pub data: FeatureDescriptorsFloatData,
}

impl FeatureDescriptorsFloat {
    /// `FeatureDescriptorsFloat(type, data)`.
    pub fn new(extractor_type: FeatureExtractorType, data: FeatureDescriptorsFloatData) -> Self {
        Self {
            extractor_type,
            data,
        }
    }

    /// `FromBytes`: SIFT bytes are cast to float; the learned-feature bytes are
    /// reinterpreted as float32 values (`Err` unless the byte row length is a multiple of 4).
    /// `Err("Unsupported feature type: <type>")` for any other type.
    pub fn from_bytes(byte_desc: &FeatureDescriptors) -> Result<FeatureDescriptorsFloat> {
        let rows = byte_desc.data.rows();
        let uint8_cols = byte_desc.data.cols();
        let data = match byte_desc.extractor_type {
            // Cast each uint8 value to float.
            FeatureExtractorType::Sift => RowMajorMatrix::from_vec(
                rows,
                uint8_cols,
                byte_desc
                    .data
                    .as_slice()
                    .iter()
                    .map(|&v| f32::from(v))
                    .collect(),
            ),
            // Reinterpret uint8 bytes as float32 data, in memory (native) byte order.
            t if stores_float_bytes(t) => {
                check_eq!(uint8_cols % SIZEOF_FLOAT, 0);
                let float_cols = uint8_cols / SIZEOF_FLOAT;
                RowMajorMatrix::from_vec(
                    rows,
                    float_cols,
                    byte_desc
                        .data
                        .as_slice()
                        .chunks_exact(SIZEOF_FLOAT)
                        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                )
            }
            t => fatal_throw!("Unsupported feature type: {}", t),
        };
        Ok(FeatureDescriptorsFloat::new(byte_desc.extractor_type, data))
    }

    /// `ToBytes`: [`FeatureDescriptors::from_float`] of these descriptors.
    pub fn to_bytes(&self) -> Result<FeatureDescriptors> {
        FeatureDescriptors::from_float(self)
    }
}

/// Port of `colmap::FeatureMatch`: feature `point2d_idx1` in the first image matches feature
/// `point2d_idx2` in the second. The default is two invalid indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureMatch {
    /// Feature index in the first image.
    pub point2d_idx1: Point2DIdx,
    /// Feature index in the second image.
    pub point2d_idx2: Point2DIdx,
}

impl Default for FeatureMatch {
    /// `FeatureMatch()`: both indices [`INVALID_POINT2D_IDX`].
    fn default() -> Self {
        Self::new(INVALID_POINT2D_IDX, INVALID_POINT2D_IDX)
    }
}

impl FeatureMatch {
    /// `FeatureMatch(point2D_idx1, point2D_idx2)`.
    pub const fn new(point2d_idx1: Point2DIdx, point2d_idx2: Point2DIdx) -> Self {
        Self {
            point2d_idx1,
            point2d_idx2,
        }
    }
}

/// `FeatureMatches`: `std::vector<FeatureMatch>`.
pub type FeatureMatches = Vec<FeatureMatch>;

/// `kKeypointMatrixCols`: the columns `[x, y, scale, orientation]` of a keypoint matrix.
pub const KEYPOINT_MATRIX_COLS: usize = 4;

/// `FeatureKeypointsMatrix`: an N x [`KEYPOINT_MATRIX_COLS`] float matrix.
pub type FeatureKeypointsMatrix = RowMajorMatrix<f32>;

/// `FeatureMatchesMatrix`: an N x 2 matrix of point2D indices.
pub type FeatureMatchesMatrix = RowMajorMatrix<u32>;

/// Port of `colmap::KeypointsToMatrix`: an N x 4 matrix `[x, y, scale, orientation]`.
pub fn keypoints_to_matrix(feature_keypoints: &[FeatureKeypoint]) -> FeatureKeypointsMatrix {
    let num_features = feature_keypoints.len();
    let mut keypoints = FeatureKeypointsMatrix::new(num_features, KEYPOINT_MATRIX_COLS);
    for (i, keypoint) in feature_keypoints.iter().enumerate() {
        keypoints[(i, 0)] = keypoint.x;
        keypoints[(i, 1)] = keypoint.y;
        keypoints[(i, 2)] = keypoint.compute_scale();
        keypoints[(i, 3)] = keypoint.compute_orientation();
    }
    keypoints
}

/// Port of `colmap::KeypointsFromMatrix`: keypoints from an N x 4 matrix
/// `[x, y, scale, orientation]`. `Err` for another column count (entry 181) or a negative
/// scale.
pub fn keypoints_from_matrix(keypoints: &FeatureKeypointsMatrix) -> Result<FeatureKeypoints> {
    check_eq!(keypoints.cols(), KEYPOINT_MATRIX_COLS);
    (0..keypoints.rows())
        .map(|i| {
            FeatureKeypoint::from_scale_orientation(
                keypoints[(i, 0)],
                keypoints[(i, 1)],
                keypoints[(i, 2)],
                keypoints[(i, 3)],
            )
        })
        .collect()
}

/// Port of `colmap::MatchesToMatrix`: an N x 2 matrix of point2D indices.
pub fn matches_to_matrix(feature_matches: &[FeatureMatch]) -> FeatureMatchesMatrix {
    let mut matches = FeatureMatchesMatrix::new(feature_matches.len(), 2);
    for (i, feature_match) in feature_matches.iter().enumerate() {
        matches[(i, 0)] = feature_match.point2d_idx1;
        matches[(i, 1)] = feature_match.point2d_idx2;
    }
    matches
}

/// Port of `colmap::MatchesFromMatrix`: matches from an N x 2 matrix of point2D indices.
/// `Err` for another column count (entry 181).
pub fn matches_from_matrix(matches: &FeatureMatchesMatrix) -> Result<FeatureMatches> {
    check_eq!(matches.cols(), 2);
    Ok((0..matches.rows())
        .map(|i| FeatureMatch::new(matches[(i, 0)], matches[(i, 1)]))
        .collect())
}
