// Port of COLMAP's src/colmap/math/math_test.cc (1:1, same test names in snake_case, same
// values and tolerances). `Median<int>({...})` takes an rvalue in C++; here the vector is a
// temporary passed as `&mut`.
#![allow(clippy::float_cmp)]

use colmap_rust::math::{
    clamp, deg_to_rad, deg_to_rad_f32, mean, median, median_absolute_deviation, n_choose_k,
    next_combination, percentile, rad_to_deg, rad_to_deg_f32, scale_sigmoid_default, sigmoid,
    sign_of_number, std_dev, truncate_cast, variance, M_PI,
};

#[test]
fn sign_of_number_nominal() {
    assert_eq!(sign_of_number(0), 1);
    assert_eq!(sign_of_number(-0.1), -1);
    assert_eq!(sign_of_number(0.1), 1);
    assert_eq!(sign_of_number(f32::INFINITY), 1);
    assert_eq!(sign_of_number(-f32::INFINITY), -1);
}

#[test]
fn clamp_nominal() {
    assert_eq!(clamp(0, -1, 1), 0);
    assert_eq!(clamp(0, 0, 1), 0);
    assert_eq!(clamp(0, -1, 0), 0);
    assert_eq!(clamp(0, -1, 1), 0);
    assert_eq!(clamp(0, 1, 2), 1);
    assert_eq!(clamp(0, -2, -1), -1);
    assert_eq!(clamp(0, 0, 0), 0);
}

#[test]
fn deg_to_rad_nominal() {
    assert_eq!(deg_to_rad_f32(0.0), 0.0f32);
    assert_eq!(deg_to_rad(0.0), 0.0);
    // C++ promotes the float to double for `- M_PI`, and compares against 1e-6f.
    assert!((f64::from(deg_to_rad_f32(180.0)) - M_PI).abs() < f64::from(1e-6f32));
    assert!((deg_to_rad(180.0) - M_PI).abs() < 1e-6);
}

#[test]
fn rad_to_deg_nominal() {
    assert_eq!(rad_to_deg_f32(0.0), 0.0f32);
    assert_eq!(rad_to_deg(0.0), 0.0);
    // RadToDeg(M_PI) is the double overload in both C++ lines.
    assert!((rad_to_deg(M_PI) - f64::from(180.0f32)).abs() < f64::from(1e-6f32));
    assert!((rad_to_deg(M_PI) - 180.0).abs() < 1e-6);
}

#[test]
fn rad_to_deg_roundtrip() {
    for i in 0..360 {
        let angle = f64::from(i);
        assert!((angle - rad_to_deg(deg_to_rad(angle))).abs() <= 1e-6);
    }
}

fn med<T: colmap_rust::math::AsF64>(mut v: Vec<T>) -> f64 {
    median(&mut v).unwrap()
}

#[test]
fn median_nominal() {
    assert_eq!(med::<i32>(vec![1, 2, 3, 4]), 2.5);
    assert_eq!(med::<i32>(vec![4, 1, 3, 2]), 2.5);
    assert_eq!(med::<i32>(vec![1, 2, 3, 100]), 2.5);
    assert_eq!(med::<i32>(vec![1, 2, 3, 4, 100]), 3.0);
    assert_eq!(med::<i32>(vec![4, 100, 1, 3, 2]), 3.0);
    assert_eq!(med::<i32>(vec![-100, 1, 2, 3, 4]), 2.0);
    assert_eq!(med::<i32>(vec![-1, -2, -3, -4]), -2.5);
    assert_eq!(med::<i32>(vec![-3, -1, -4, -2]), -2.5);
    assert_eq!(med::<i32>(vec![-1, -2, 3, 4]), 1.0);
    // Test integer overflow scenario.
    assert_eq!(med::<i8>(vec![100, 115, 119, 127]), 117.0);
}

fn mad(mut v: Vec<i32>) -> (f64, f64) {
    median_absolute_deviation(&mut v).unwrap()
}

#[test]
fn median_absolute_deviation_nominal() {
    // {1, 2, 3, 4, 5} -> median=3, deviations={2, 1, 0, 1, 2}, MAD=1
    let (median1, mad1) = mad(vec![1, 2, 3, 4, 5]);
    assert_eq!(median1, 3.0);
    assert_eq!(mad1, 1.0);

    // {1, 2, 3, 4} -> median=2.5, deviations={1.5, 0.5, 0.5, 1.5}, MAD=1
    let (median2, mad2) = mad(vec![1, 2, 3, 4]);
    assert_eq!(median2, 2.5);
    assert_eq!(mad2, 1.0);

    // Unsorted input: {5, 1, 3, 2, 4} -> same as {1, 2, 3, 4, 5}
    let (median3, mad3) = mad(vec![5, 1, 3, 2, 4]);
    assert_eq!(median3, 3.0);
    assert_eq!(mad3, 1.0);

    // Single element: {42} -> median=42, MAD=0
    let (median4, mad4) = mad(vec![42]);
    assert_eq!(median4, 42.0);
    assert_eq!(mad4, 0.0);

    // With outlier: {1, 2, 3, 4, 100} -> median=3, deviations={2, 1, 0, 1, 97}
    let (median5, mad5) = mad(vec![1, 2, 3, 4, 100]);
    assert_eq!(median5, 3.0);
    assert_eq!(mad5, 1.0);
}

fn pct(mut v: Vec<i32>, p: f64) -> f64 {
    percentile(&mut v, p).unwrap()
}

#[test]
fn percentile_nominal() {
    assert_eq!(pct(vec![0], 0.0), 0.0);
    assert_eq!(pct(vec![0], 50.0), 0.0);
    assert_eq!(pct(vec![0], 100.0), 0.0);
    assert_eq!(pct(vec![0, 1], 0.0), 0.0);
    assert_eq!(pct(vec![1, 0], 0.0), 0.0);
    assert_eq!(pct(vec![0, 1], 50.0), 0.5);
    assert_eq!(pct(vec![1, 0], 50.0), 0.5);
    assert_eq!(pct(vec![0, 1], 100.0), 1.0);
    assert_eq!(pct(vec![1, 0], 100.0), 1.0);
    assert_eq!(pct(vec![0, 1, 2], 0.0), 0.0);
    assert_eq!(pct(vec![0, 1, 2], 50.0), 1.0);
    assert_eq!(pct(vec![0, 1, 2], 100.0), 2.0);
    assert_eq!(pct(vec![0, 1, 1, 2], 0.0), 0.0);
    assert_eq!(pct(vec![0, 1, 1, 2], 100. / 3.), 1.0);
    assert_eq!(pct(vec![0, 1, 1, 2], 50.0), 1.0);
    assert_eq!(pct(vec![0, 1, 1, 2], 100. / 3. * 2.), 1.0);
    assert_eq!(pct(vec![1, 2, 0, 1], 100. / 3. * 2.), 1.0);
    assert_eq!(pct(vec![0, 1, 1, 2], 100.0), 2.0);
    assert_eq!(pct(vec![1, 2, 0, 1], 100.0), 2.0);
    assert_eq!(pct(vec![0, 100], 1.0), 1.0);
    assert_eq!(pct(vec![0, 100], 50.0), 50.0);
    assert_eq!(pct(vec![0, 100], 50.1), 50.1);
    assert_eq!(pct(vec![0, 100], 99.0), 99.0);
    assert_eq!(pct(vec![0, 1, 2, 3], 1.0), 0.03);
    assert_eq!(pct(vec![0, 1, 2, 3], 2.0), 0.06);
    assert_eq!(pct(vec![0, 1, 2, 3], 33.0), 0.99);
    assert_eq!(pct(vec![0, 1, 2, 3], 34.0), 1.02);
    assert_eq!(pct(vec![3, 0, 1, 2], 34.0), 1.02);
}

#[test]
fn mean_nominal() {
    assert_eq!(mean::<i32>(&[1, 2, 3, 4]).unwrap(), 2.5);
    assert_eq!(mean::<i32>(&[1, 2, 3, 100]).unwrap(), 26.5);
    assert_eq!(mean::<i32>(&[1, 2, 3, 4, 100]).unwrap(), 22.0);
    assert_eq!(mean::<i32>(&[-100, 1, 2, 3, 4]).unwrap(), -18.0);
    assert_eq!(mean::<i32>(&[-1, -2, -3, -4]).unwrap(), -2.5);
    assert_eq!(mean::<i32>(&[-1, -2, 3, 4]).unwrap(), 1.0);
}

#[test]
fn variance_nominal() {
    let v = |x: &[i32]| variance(x).unwrap();
    assert!((v(&[1, 2, 3, 4]) - 1.66666666).abs() <= 1e-6);
    assert!((v(&[1, 2, 3, 100]) - 2401.66666666).abs() <= 1e-6);
    assert!((v(&[1, 2, 3, 4, 100]) - 1902.5).abs() <= 1e-6);
    assert!((v(&[-100, 1, 2, 3, 4]) - 2102.5).abs() <= 1e-6);
    assert!((v(&[-1, -2, -3, -4]) - 1.66666666).abs() <= 1e-6);
    assert!((v(&[-1, -2, 3, 4]) - 8.66666666).abs() <= 1e-6);
}

#[test]
fn std_dev_nominal() {
    let a = [1, 2, 3, 4];
    assert!((variance(&a).unwrap().sqrt() - std_dev(&a).unwrap()).abs() <= 1e-6);
    let b = [1, 2, 3, 100];
    assert!((variance(&b).unwrap().sqrt() - std_dev(&b).unwrap()).abs() <= 1e-6);
}

#[test]
fn next_combination_nominal() {
    let mut list = vec![0];
    assert!(!next_combination(&mut list, 1));
    list = vec![0, 1];
    assert!(!next_combination(&mut list, 2));
    assert_eq!(list[0], 0);
    assert!(next_combination(&mut list, 1));
    assert_eq!(list[0], 1);
    assert!(!next_combination(&mut list, 1));
    assert_eq!(list[0], 0);
    list = vec![0, 1, 2];
    assert_eq!(list[0], 0);
    assert_eq!(list[1], 1);
    assert_eq!(list[2], 2);
    assert!(next_combination(&mut list, 2));
    assert_eq!(list[0], 0);
    assert_eq!(list[1], 2);
    assert_eq!(list[2], 1);
    assert!(next_combination(&mut list, 2));
    assert_eq!(list[0], 1);
    assert_eq!(list[1], 2);
    assert_eq!(list[2], 0);
    assert!(!next_combination(&mut list, 2));
    assert_eq!(list[0], 0);
    assert_eq!(list[1], 1);
    assert_eq!(list[2], 2);
}

#[test]
fn sigmoid_nominal() {
    assert_eq!(sigmoid(0.0, 1.0), 0.5);
    assert!((sigmoid(100.0, 1.0) - 1.0f64).abs() <= 1e-10);
    assert!((sigmoid(-100.0, 1.0) - 0.0f64).abs() <= 1e-10);
}

#[test]
fn scale_sigmoid_nominal() {
    assert!((scale_sigmoid_default(0.5f64) - 0.5).abs() <= 1e-10);
    assert!((scale_sigmoid_default(1.0f64) - 1.0).abs() <= 1e-10);
    assert!((scale_sigmoid_default(-1.0f64) - 0.0).abs() <= 1e-4);
}

#[test]
fn n_choose_k_nominal() {
    assert_eq!(n_choose_k(0, 0), 0);

    assert_eq!(n_choose_k(1, 0), 1);
    assert_eq!(n_choose_k(2, 0), 1);
    assert_eq!(n_choose_k(3, 0), 1);

    assert_eq!(n_choose_k(1, 1), 1);
    assert_eq!(n_choose_k(2, 1), 2);
    assert_eq!(n_choose_k(3, 1), 3);

    assert_eq!(n_choose_k(2, 2), 1);
    assert_eq!(n_choose_k(2, 3), 0);

    assert_eq!(n_choose_k(3, 2), 3);
    assert_eq!(n_choose_k(4, 2), 6);
    assert_eq!(n_choose_k(5, 2), 10);

    assert_eq!(n_choose_k(500, 3), 20708500);
    assert_eq!(n_choose_k(500, 7), 1486071034734000);
    assert_eq!(n_choose_k(10000, 5), 832500291625002000);
}

#[test]
fn truncate_cast_nominal() {
    assert_eq!(truncate_cast::<i32, i8>(-129), -128);
    assert_eq!(truncate_cast::<i32, i8>(128), 127);
    assert_eq!(truncate_cast::<i32, u8>(-1), 0);
    assert_eq!(truncate_cast::<i32, u8>(256), 255);
    assert_eq!(truncate_cast::<i32, u16>(-1), 0);
    assert_eq!(truncate_cast::<i32, u16>(65536), 65535);
}
