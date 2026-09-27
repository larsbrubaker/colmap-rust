//! The id vocabulary of COLMAP's `src/colmap/util/types.h`: rig, camera, image, frame,
//! image-pair, 2D-point, 3D-point, timestamp and pose-prior ids with their `kInvalid*`
//! sentinels, the `sensor_t` / `data_t` composite ids, the image-pair id packing
//! (`ImagePairToPairId`, `PairIdToImagePair`) and `PairHash` / `HashCombine`. Everything in
//! `scene/` builds on it. Port of colmap-sharp's `ColmapSharp/Util/Types.cs`.
//! Tests: `tests/util/types.rs` (`types_test.cc`).
//!
//! Tier A (exact): the pair-id packing and [`PairHash::pair_hash`] give COLMAP's numbers.
//!
//! Decisions (same as colmap-sharp):
//! - Ids are plain integer aliases, not newtypes. COLMAP's hot loops index containers by id
//!   and do arithmetic on them (pair packing, point2D index ranges); wrappers would add a
//!   conversion at each such site for little protection, since COLMAP itself relies only on
//!   the typedef names. `image_t` is [`ImageId`], `kInvalidImageId` is [`INVALID_IMAGE_ID`].
//! - COLMAP's `span<T>` and `filter_view` are not ported: Rust slices and
//!   `Iterator::filter` are the equivalents and callers use those directly.
//! - The `std::hash` specializations of `sensor_t` / `data_t` become derived `Hash`, since
//!   no result depends on the hash values (hash-container iteration order never reaches an
//!   output; CLAUDE.md "Hash iteration order").

use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};

use super::check::{ColmapError, ErrorKind, Result};

/// `rig_t`.
pub type RigId = u32;
/// `kInvalidRigId`.
pub const INVALID_RIG_ID: RigId = RigId::MAX;

/// `camera_t`.
pub type CameraId = u32;
/// `kInvalidCameraId`.
pub const INVALID_CAMERA_ID: CameraId = CameraId::MAX;

/// `image_t`.
pub type ImageId = u32;
/// `kInvalidImageId`.
pub const INVALID_IMAGE_ID: ImageId = ImageId::MAX;

/// `kMaxNumImages`: image ids stay below `i32::MAX` so that a pair of them packs into one
/// [`ImagePairId`].
pub const MAX_NUM_IMAGES: u64 = i32::MAX as u64;

/// `frame_t`.
pub type FrameId = u32;
/// `kInvalidFrameId`.
pub const INVALID_FRAME_ID: FrameId = FrameId::MAX;

/// `image_pair_t`: each image pair gets a unique id, see [`image_pair_to_pair_id`].
pub type ImagePairId = u64;
/// `kInvalidImagePairId`.
pub const INVALID_IMAGE_PAIR_ID: ImagePairId = ImagePairId::MAX;

/// `point2D_t`: index per image, i.e. determines the maximum number of 2D points per image.
pub type Point2DIdx = u32;
/// `kInvalidPoint2DIdx`.
pub const INVALID_POINT2D_IDX: Point2DIdx = Point2DIdx::MAX;

/// `point3D_t`: unique identifier per added 3D point. Since many 3D points are added,
/// deleted and possibly re-added, the maximum number of unique ids should be large.
pub type Point3DId = u64;
/// `kInvalidPoint3DId`.
pub const INVALID_POINT3D_ID: Point3DId = Point3DId::MAX;

/// `timestamp_t`: nanoseconds. An integer rather than a double avoids precision loss when
/// comparing or differencing large absolute timestamps and allows exact map keys.
pub type Timestamp = i64;
/// `kInvalidTimestamp`: `INT64_MIN`, so any realistic timestamp (including negative ones)
/// stays valid.
pub const INVALID_TIMESTAMP: Timestamp = Timestamp::MIN;

/// `pose_prior_t`.
pub type PosePriorId = u32;
/// `kInvalidPosePriorId`.
pub const INVALID_POSE_PRIOR_ID: PosePriorId = PosePriorId::MAX;

/// Port of `colmap::SensorType` (`MAKE_ENUM_CLASS_OVERLOAD_STREAM(SensorType, -1, INVALID,
/// CAMERA, IMU)`). Ordered by its integer value, as `std::tie` compares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum SensorType {
    /// `INVALID` (-1).
    Invalid = -1,
    /// `CAMERA` (0).
    Camera = 0,
    /// `IMU` (1).
    Imu = 1,
}

impl SensorType {
    /// `SensorTypeToString`: "INVALID", "CAMERA" or "IMU".
    pub fn as_str(self) -> &'static str {
        match self {
            SensorType::Invalid => "INVALID",
            SensorType::Camera => "CAMERA",
            SensorType::Imu => "IMU",
        }
    }

    /// `SensorTypeFromString`. Anything but the three spellings is COLMAP's
    /// `std::runtime_error("Unknown string value: <v> for enum: SensorType")`.
    pub fn from_colmap_str(value: &str) -> Result<SensorType> {
        match value {
            "INVALID" => Ok(SensorType::Invalid),
            "CAMERA" => Ok(SensorType::Camera),
            "IMU" => Ok(SensorType::Imu),
            _ => Err(ColmapError::new(
                ErrorKind::RuntimeError,
                format!("Unknown string value: {value} for enum: SensorType"),
            )),
        }
    }
}

impl fmt::Display for SensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Port of `colmap::sensor_t`: a sensor identified by its type and its id within that type
/// (a [`CameraId`] for cameras). Ordered by `(type, id)` like COLMAP's `operator<`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SensorId {
    /// Type of the sensor.
    pub sensor_type: SensorType,
    /// Unique identifier of the sensor within its type.
    pub id: u32,
}

impl SensorId {
    /// `sensor_t::kInvalidId`.
    pub const INVALID_ID: u32 = u32::MAX;

    /// `sensor_t(type, id)`.
    pub const fn new(sensor_type: SensorType, id: u32) -> Self {
        Self { sensor_type, id }
    }
}

impl Default for SensorId {
    /// The default `sensor_t`: `(INVALID, kInvalidId)`.
    fn default() -> Self {
        INVALID_SENSOR_ID
    }
}

/// `kInvalidSensorId`.
pub const INVALID_SENSOR_ID: SensorId = SensorId::new(SensorType::Invalid, SensorId::INVALID_ID);

/// Port of `colmap::data_t`: one measurement (an [`ImageId`] for cameras) of a sensor.
/// Ordered by `(sensor_id, id)` like COLMAP's `operator<`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataId {
    /// The sensor that produced the measurement.
    pub sensor_id: SensorId,
    /// The measurement's id within the sensor. `u64` as in COLMAP, although its
    /// constructor takes a `u32`.
    pub id: u64,
}

impl DataId {
    /// `data_t::kInvalidId`. COLMAP declares it `uint32_t` max although the field is
    /// `uint64_t`; both are kept.
    pub const INVALID_ID: u32 = u32::MAX;

    /// `data_t(sensor_id, id)`.
    pub const fn new(sensor_id: SensorId, id: u32) -> Self {
        Self {
            sensor_id,
            id: id as u64,
        }
    }
}

impl Default for DataId {
    /// The default `data_t`: `(kInvalidSensorId, kInvalidId)`.
    fn default() -> Self {
        INVALID_DATA_ID
    }
}

/// `kInvalidDataId`.
pub const INVALID_DATA_ID: DataId = DataId::new(INVALID_SENSOR_ID, DataId::INVALID_ID);

/// Port of `ShouldSwapImagePair`: a pair is stored with the smaller image id first, so the
/// pair id does not depend on the order the ids are given in.
pub fn should_swap_image_pair(image_id1: ImageId, image_id2: ImageId) -> bool {
    image_id1 > image_id2
}

/// Port of `ThrowIfGtMaxImages` (`std::runtime_error`).
fn throw_if_gt_max_images(image_id: ImageId) -> Result<()> {
    if u64::from(image_id) >= MAX_NUM_IMAGES {
        return Err(ColmapError::new(
            ErrorKind::RuntimeError,
            format!("image_id={image_id} >= kMaxNumImages."),
        ));
    }
    Ok(())
}

/// Port of `ImagePairToPairId`: packs an unordered image pair into one id, smaller id first.
/// Errors if either id is not below [`MAX_NUM_IMAGES`].
pub fn image_pair_to_pair_id(image_id1: ImageId, image_id2: ImageId) -> Result<ImagePairId> {
    throw_if_gt_max_images(image_id1)?;
    throw_if_gt_max_images(image_id2)?;
    // Both ids are below 2^31 - 1, so the product and sum stay far below 2^64.
    Ok(if should_swap_image_pair(image_id1, image_id2) {
        MAX_NUM_IMAGES * u64::from(image_id2) + u64::from(image_id1)
    } else {
        MAX_NUM_IMAGES * u64::from(image_id1) + u64::from(image_id2)
    })
}

/// Port of `PairIdToImagePair`: the inverse of [`image_pair_to_pair_id`], smaller id first.
pub fn pair_id_to_image_pair(pair_id: ImagePairId) -> Result<(ImageId, ImageId)> {
    // `static_cast<image_t>` truncates to the low 32 bits, as `as u32` does.
    let image_id2 = (pair_id % MAX_NUM_IMAGES) as ImageId;
    let image_id1 = ((pair_id - u64::from(image_id2)) / MAX_NUM_IMAGES) as ImageId;
    throw_if_gt_max_images(image_id1)?;
    throw_if_gt_max_images(image_id2)?;
    Ok((image_id1, image_id2))
}

/// Port of `HashCombine`: folds `value` into `seed` with boost's `hash_combine` spread, on
/// a 64-bit `size_t` (every platform COLMAP's oracle runs on).
pub fn hash_combine(seed: u64, value: u64) -> u64 {
    seed ^ (value
        .wrapping_add(0x9e37_79b9)
        .wrapping_add(seed << 6)
        .wrapping_add(seed >> 2))
}

/// Port of `colmap::PairHash` on the pair types COLMAP keys containers with. Pairs of
/// 32-bit integers pack into disjoint halves of the 64-bit hash (collision-free; signed
/// halves go through their unsigned counterpart, so negatives keep exactly their low 32
/// bits). Pairs of 64-bit integers go through [`hash_combine`] of their `std::hash` values,
/// which libc++ defines as the value itself.
pub trait PairHash {
    /// COLMAP's `PairHash{}(pair)`, bit-exact.
    fn pair_hash(&self) -> u64;
}

impl PairHash for (u32, u32) {
    fn pair_hash(&self) -> u64 {
        (u64::from(self.0) << 32) | u64::from(self.1)
    }
}

impl PairHash for (i32, i32) {
    fn pair_hash(&self) -> u64 {
        (self.0 as u32, self.1 as u32).pair_hash()
    }
}

impl PairHash for (u64, u64) {
    fn pair_hash(&self) -> u64 {
        hash_combine(self.0, self.1)
    }
}

/// A `std::hash::Hasher` for `HashMap`/`HashSet` keyed on a pair of integers (COLMAP's
/// `FlatHashSet<std::pair<..>, PairHash>`). It computes [`PairHash::pair_hash`] from the
/// two `write_*` calls a tuple's `Hash` makes, then mixes it (Fibonacci multiply and a
/// fold): hashbrown picks buckets from the low bits and tags from the top 7, and the raw
/// packed value of small ids has all-zero top bits. Only [`PairHash::pair_hash`] is
/// COLMAP-exact; this hasher's output is never observable in results, because hash-container
/// iteration order must not reach an output (CLAUDE.md).
#[derive(Debug, Default, Clone, Copy)]
pub struct PairHasher {
    first: Option<u64>,
    first_is_32bit: bool,
    hash: u64,
}

impl PairHasher {
    fn push(&mut self, value: u64, is_32bit: bool) {
        match self.first.take() {
            None => {
                self.first = Some(value);
                self.first_is_32bit = is_32bit;
            }
            Some(first) => {
                self.hash = if is_32bit && self.first_is_32bit {
                    (first << 32) | value
                } else {
                    hash_combine(first, value)
                };
            }
        }
    }
}

impl Hasher for PairHasher {
    fn write(&mut self, bytes: &[u8]) {
        // Only reached by key types other than integer pairs; fold the bytes in so the
        // hasher stays correct (equal keys, equal hashes) for them too.
        for &b in bytes {
            self.hash = hash_combine(self.hash, u64::from(b));
        }
    }

    fn write_u32(&mut self, i: u32) {
        self.push(u64::from(i), true);
    }

    fn write_i32(&mut self, i: i32) {
        self.push(u64::from(i as u32), true);
    }

    fn write_u64(&mut self, i: u64) {
        self.push(i, false);
    }

    fn write_i64(&mut self, i: i64) {
        self.push(i as u64, false);
    }

    fn write_usize(&mut self, i: usize) {
        self.push(i as u64, false);
    }

    fn finish(&self) -> u64 {
        let hash = match self.first {
            // A single integer was written (not a pair): use it as is.
            Some(first) => first,
            None => self.hash,
        };
        let mixed = hash.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        mixed ^ (mixed >> 32)
    }
}

/// `BuildHasher` for pair-keyed hash containers:
/// `HashSet<(Point2DIdx, Point2DIdx), PairHashBuilder>`.
pub type PairHashBuilder = BuildHasherDefault<PairHasher>;
