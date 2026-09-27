// Port of COLMAP's src/colmap/util/types_test.cc, 1:1 (Suite_Name -> suite_name), testing
// colmap_rust::util::types. `FlatHashSet<pair, PairHash>` is a std `HashSet` with
// `PairHashBuilder`; `h(pair)` is `PairHash::pair_hash`. Tier A (exact).
//
// Not ported, because the C++ code they test is not ported (src/util/types.rs header):
// Span.SizeAndEmpty (COLMAP's span<T>; Rust uses slices) and FilterView.Empty / All / None /
// Nominal / RangeExpression (COLMAP's filter_view; Rust uses Iterator::filter). Testing std
// in their place would test no colmap-rust code.

use std::collections::HashSet;
use std::hash::{BuildHasher, Hash};

use colmap_rust::util::types::*;

#[test]
fn should_swap_image_pair_nominal() {
    assert!(!should_swap_image_pair(0, 0));
    assert!(!should_swap_image_pair(0, 1));
    assert!(should_swap_image_pair(1, 0));
    assert!(!should_swap_image_pair(1, 1));
}

#[test]
fn image_pair_to_pair_id_nominal() {
    assert_eq!(image_pair_to_pair_id(0, 0).unwrap(), 0);
    assert_eq!(image_pair_to_pair_id(0, 1).unwrap(), 1);
    assert_eq!(image_pair_to_pair_id(0, 2).unwrap(), 2);
    assert_eq!(image_pair_to_pair_id(0, 3).unwrap(), 3);
    assert_eq!(image_pair_to_pair_id(1, 2).unwrap(), MAX_NUM_IMAGES + 2);
    for i in 0..20u32 {
        for j in 0..20u32 {
            let pair_id = image_pair_to_pair_id(i, j).unwrap();
            let (image_id1, image_id2) = pair_id_to_image_pair(pair_id).unwrap();
            if i < j {
                assert_eq!(i, image_id1);
                assert_eq!(j, image_id2);
            } else {
                assert_eq!(i, image_id2);
                assert_eq!(j, image_id1);
            }
        }
    }
}

fn pair_set<T: Eq + Hash>() -> HashSet<T, PairHashBuilder> {
    HashSet::with_hasher(PairHashBuilder::default())
}

#[test]
fn feature_match_hashing_nominal() {
    let mut set = pair_set::<(Point2DIdx, Point2DIdx)>();
    set.insert((1, 2));
    assert_eq!(set.len(), 1);
    set.insert((1, 2));
    assert_eq!(set.len(), 1);
    assert!(!set.contains(&(0, 0)));
    assert!(set.contains(&(1, 2)));
    assert!(!set.contains(&(2, 1)));
    set.insert((2, 1));
    assert_eq!(set.len(), 2);
    assert!(!set.contains(&(0, 0)));
    assert!(set.contains(&(1, 2)));
    assert!(set.contains(&(2, 1)));
}

#[test]
fn feature_match_hashing_large_values() {
    let hi = Point2DIdx::MAX;
    let lo: Point2DIdx = 1;
    let mut set = pair_set::<(Point2DIdx, Point2DIdx)>();
    set.insert((hi, lo));
    set.insert((lo, hi));
    set.insert((hi, hi));
    set.insert((lo, lo));
    assert_eq!(set.len(), 4);
    assert!(set.contains(&(hi, lo)));
    assert!(set.contains(&(lo, hi)));
    assert!(set.contains(&(hi, hi)));
    assert!(set.contains(&(lo, lo)));
}

#[test]
fn feature_match_hashing_deterministic() {
    let a: (Point2DIdx, Point2DIdx) = (42, 99);
    let b: (Point2DIdx, Point2DIdx) = (99, 42);
    assert_eq!(a.pair_hash(), a.pair_hash());
    assert_ne!(a.pair_hash(), b.pair_hash());
}

#[test]
fn signed_pair_hashing_large_values() {
    let hi = i32::MAX;
    let lo = i32::MIN;
    let mut set = pair_set::<(i32, i32)>();
    set.insert((hi, lo));
    set.insert((lo, hi));
    set.insert((hi, hi));
    set.insert((lo, lo));
    assert_eq!(set.len(), 4);
    assert!(set.contains(&(hi, lo)));
    assert!(set.contains(&(lo, hi)));
    assert!(set.contains(&(hi, hi)));
    assert!(set.contains(&(lo, lo)));
}

#[test]
fn signed_pair_hashing_deterministic() {
    assert_eq!((-42i32, 99i32).pair_hash(), (-42i32, 99i32).pair_hash());
    assert_ne!((-42i32, 99i32).pair_hash(), (99i32, -42i32).pair_hash());
    // Distinct negatives that share low bits with positives must not collide.
    assert_ne!((-1i32, 0i32).pair_hash(), (0i32, -1i32).pair_hash());
}

#[test]
fn point3_d_pair_hashing_nominal() {
    let mut set = pair_set::<(Point3DId, Point3DId)>();
    set.insert((1, 2));
    assert_eq!(set.len(), 1);
    set.insert((1, 2));
    assert_eq!(set.len(), 1);
    assert!(!set.contains(&(0, 0)));
    assert!(set.contains(&(1, 2)));
    assert!(!set.contains(&(2, 1)));
    set.insert((2, 1));
    assert_eq!(set.len(), 2);
    assert!(!set.contains(&(0, 0)));
    assert!(set.contains(&(1, 2)));
    assert!(set.contains(&(2, 1)));
}

#[test]
fn point3_d_pair_hashing_large_values() {
    let hi = Point3DId::MAX;
    let lo: Point3DId = 1;
    let mut set = pair_set::<(Point3DId, Point3DId)>();
    set.insert((hi, lo));
    set.insert((lo, hi));
    set.insert((hi, hi));
    set.insert((lo, lo));
    assert_eq!(set.len(), 4);
    assert!(set.contains(&(hi, lo)));
    assert!(set.contains(&(lo, hi)));
    assert!(set.contains(&(hi, hi)));
    assert!(set.contains(&(lo, lo)));
}

#[test]
fn point3_d_pair_hashing_deterministic() {
    let a: (Point3DId, Point3DId) = (42, 99);
    let b: (Point3DId, Point3DId) = (99, 42);
    assert_eq!(a.pair_hash(), a.pair_hash());
    assert_ne!(a.pair_hash(), b.pair_hash());
}

// Rust-only (port of colmap-sharp's PairHash_GetHashCodeMixesBothHalves): the container
// hasher must not collapse a pair to a ^ b, which would make swapped pairs and every pair
// with the same XOR collide.
#[test]
fn rust_only_pair_hasher_mixes_both_halves() {
    let build = PairHashBuilder::default();
    let h32 = |p: (u32, u32)| build.hash_one(p);
    assert_ne!(h32((1, 2)), h32((2, 1)));
    // 1 ^ 2 == 0 ^ 3 == 4 ^ 7.
    assert_ne!(h32((1, 2)), h32((0, 3)));
    assert_ne!(h32((0, 3)), h32((4, 7)));
    assert_ne!(build.hash_one((1i32, 2i32)), build.hash_one((2i32, 1i32)));
    assert_ne!(build.hash_one((1u64, 2u64)), build.hash_one((2u64, 1u64)));
}

// Rust-only: COLMAP's exact PairHash values (32-bit halves pack, 64-bit pairs go through
// boost's hash_combine), computed by hand from types.h.
#[test]
fn rust_only_pair_hash_exact_values() {
    assert_eq!((1u32, 2u32).pair_hash(), (1u64 << 32) | 2);
    assert_eq!((-1i32, 0i32).pair_hash(), 0xFFFF_FFFF_0000_0000);
    // HashCombine(1, 2) = 1 ^ (2 + 0x9e3779b9 + (1 << 6) + (1 >> 2)).
    assert_eq!((1u64, 2u64).pair_hash(), 1 ^ (2 + 0x9e37_79b9 + 64));
    assert_eq!(hash_combine(0, 0), 0x9e37_79b9);
}

// Rust-only: the error path of ThrowIfGtMaxImages (std::runtime_error in COLMAP).
#[test]
fn rust_only_image_pair_to_pair_id_rejects_too_large_ids() {
    let too_large = MAX_NUM_IMAGES as ImageId;
    let err = image_pair_to_pair_id(too_large, 0).unwrap_err();
    assert_eq!(err.kind(), colmap_rust::ErrorKind::RuntimeError);
    assert_eq!(err.message(), "image_id=2147483647 >= kMaxNumImages.");
    assert!(image_pair_to_pair_id(0, too_large - 1).is_ok());
    // image_id1 = kMaxNumImages, image_id2 = 0.
    assert!(pair_id_to_image_pair(MAX_NUM_IMAGES * MAX_NUM_IMAGES).is_err());
}

// Rust-only: the sensor_t / data_t defaults, ordering and SensorType spellings.
#[test]
fn rust_only_sensor_and_data_ids() {
    assert_eq!(SensorId::default(), INVALID_SENSOR_ID);
    assert_eq!(INVALID_SENSOR_ID.sensor_type, SensorType::Invalid);
    assert_eq!(INVALID_SENSOR_ID.id, u32::MAX);
    assert_eq!(DataId::default(), INVALID_DATA_ID);
    assert_eq!(INVALID_DATA_ID.id, u64::from(u32::MAX));
    // Ordered by (type, id) with INVALID (-1) first.
    assert!(SensorId::new(SensorType::Invalid, 5) < SensorId::new(SensorType::Camera, 0));
    assert!(SensorId::new(SensorType::Camera, 1) < SensorId::new(SensorType::Camera, 2));
    assert!(SensorId::new(SensorType::Camera, 9) < SensorId::new(SensorType::Imu, 0));
    let camera = SensorId::new(SensorType::Camera, 1);
    assert!(DataId::new(camera, 3) < DataId::new(camera, 4));
    assert_eq!(SensorType::Camera.to_string(), "CAMERA");
    for t in [SensorType::Invalid, SensorType::Camera, SensorType::Imu] {
        assert_eq!(SensorType::from_colmap_str(t.as_str()).unwrap(), t);
    }
    assert_eq!(
        SensorType::from_colmap_str("GPS").unwrap_err().message(),
        "Unknown string value: GPS for enum: SensorType"
    );
}
