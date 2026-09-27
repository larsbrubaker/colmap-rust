// Port of COLMAP's src/colmap/geometry/gps_test.cc (all cases), plus Rust-only checks of
// the UTM argument checks (including longitude 180, docs/CPP_DIVERGENCES.md entry 82).

// gps_test.cc's reference literals are kept digit for digit.
#![allow(clippy::excessive_precision)]

use colmap_rust::geometry::{Ellipsoid, GpsTransform};
use colmap_rust::linalg::Vector3d;

use super::eigen_matrix_near;

// The two Munich points of gps_test.cc, (lat, lon) in degrees from d/m/s.
fn ell() -> Vec<Vector3d> {
    vec![
        Vector3d::new(
            48.0 + 8. / 60.0 + 51.70361 / 3600.0,
            11.0 + 34. / 60.0 + 10.51777 / 3600.0,
            561.1851,
        ),
        Vector3d::new(
            48.0 + 8. / 60.0 + 52.40575 / 3600.0,
            11.0 + 34. / 60.0 + 11.77179 / 3600.0,
            561.1509,
        ),
    ]
}

// A third point one degree east, in UTM zone 33.
fn ell3() -> Vec<Vector3d> {
    let mut points = ell();
    points.push(Vector3d::new(
        48.0 + 8. / 60.0 + 52.40575 / 3600.0,
        12.0 + 34. / 60.0 + 11.77179 / 3600.0,
        561.1509,
    ));
    points
}

fn grs80_xyz() -> Vec<Vector3d> {
    vec![
        Vector3d::new(
            4.1772397090808507e6,
            0.85515377993121441e6,
            4.7282674046563692e6,
        ),
        Vector3d::new(
            4.1772186604902023e6,
            0.8551759313518483e6,
            4.7282818502697079e6,
        ),
    ]
}

fn wgs84_xyz() -> Vec<Vector3d> {
    vec![
        Vector3d::new(
            4.177239709042750e6,
            0.855153779923415e6,
            4.728267404769168e6,
        ),
        Vector3d::new(
            4.177218660452103e6,
            0.855175931344048e6,
            4.728281850382507e6,
        ),
    ]
}

fn roundtrip_xyz() -> Vec<Vector3d> {
    vec![
        Vector3d::new(
            4.177239709080851e6,
            0.855153779931214e6,
            4.728267404656370e6,
        ),
        Vector3d::new(
            4.177218660490202e6,
            0.855175931351848e6,
            4.728281850269709e6,
        ),
    ]
}

fn assert_all_near(actual: &[Vector3d], expected: &[Vector3d], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(eigen_matrix_near(a, e, tol), "{a:?} vs {e:?}");
    }
}

#[test]
fn gps_ellipsoid_to_ecef_grs80() {
    let gps_tform = GpsTransform::new(Ellipsoid::Grs80);
    assert_all_near(&gps_tform.ellipsoid_to_ecef(&ell()), &grs80_xyz(), 1e-8);
}

#[test]
fn gps_ellipsoid_to_ecef_wgs84() {
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    assert_all_near(&gps_tform.ellipsoid_to_ecef(&ell()), &wgs84_xyz(), 1e-8);
}

#[test]
fn gps_ecef_to_ellipsoid_grs80() {
    let gps_tform = GpsTransform::new(Ellipsoid::Grs80);
    assert_all_near(&gps_tform.ecef_to_ellipsoid(&grs80_xyz()), &ell(), 1e-5);
}

#[test]
fn gps_ecef_to_ellipsoid_wgs84() {
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    assert_all_near(&gps_tform.ecef_to_ellipsoid(&wgs84_xyz()), &ell(), 1e-5);
}

#[test]
fn gps_ecef_to_ellipsoidipsoid_to_ecef_grs80() {
    let xyz = roundtrip_xyz();
    let gps_tform = GpsTransform::new(Ellipsoid::Grs80);
    let ell = gps_tform.ecef_to_ellipsoid(&xyz);
    let xyz2 = gps_tform.ellipsoid_to_ecef(&ell);
    assert_all_near(&xyz, &xyz2, 1e-5);
}

#[test]
fn gps_ecef_to_ellipsoidipsoid_to_ecef_wgs84() {
    let xyz = roundtrip_xyz();
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    let ell = gps_tform.ecef_to_ellipsoid(&xyz);
    let xyz2 = gps_tform.ellipsoid_to_ecef(&ell);
    assert_all_near(&xyz, &xyz2, 1e-5);
}

#[test]
fn gps_ellipsoid_to_enu_wgs84() {
    let ref_xyz = wgs84_xyz();
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    // Get lat0, lon0 origin from ref
    let ori_ell = gps_tform.ecef_to_ellipsoid(&[ref_xyz[0]])[0];
    // Get ENU ref from ECEF ref
    let ref_enu = gps_tform.ecef_to_enu(&ref_xyz, ref_xyz[0]);
    // Get ENU from Ell
    let enu = gps_tform.ellipsoid_to_enu(&ell(), ori_ell.x, ori_ell.y, ori_ell.z);
    assert_all_near(&enu, &ref_enu, 1e-8);
}

#[test]
fn gps_ecef_to_enu() {
    let ref_xyz = wgs84_xyz();
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    let xyz = gps_tform.ellipsoid_to_ecef(&ell());
    // Get ENU from ECEF ref
    let ref_enu = gps_tform.ecef_to_enu(&ref_xyz, ref_xyz[0]);
    // Get ENU from ECEF
    let enu = gps_tform.ecef_to_enu(&xyz, xyz[0]);
    assert_all_near(&enu, &ref_enu, 1e-8);
}

#[test]
fn gps_enu_to_ellipsoid_wgs84() {
    let ref_ell = ell();
    let xyz = wgs84_xyz();
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    // Get lat0, lon0 origin from ref
    let ori_ell = gps_tform.ecef_to_ellipsoid(&xyz);
    let (lat0, lon0, alt0) = (ori_ell[0].x, ori_ell[0].y, ori_ell[0].z);
    // Get ENU from ECEF
    let enu = gps_tform.ecef_to_enu(&xyz, xyz[0]);
    let _xyz_enu = gps_tform.enu_to_ecef(&enu, lat0, lon0, alt0);
    // Get Ell from ENU
    let ell = gps_tform.enu_to_ellipsoid(&enu, lat0, lon0, alt0);
    assert_all_near(&ell, &ref_ell, 1e-5);
}

#[test]
fn gps_enu_to_ecef() {
    let ell = ell();
    let ref_xyz = wgs84_xyz();
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    // Get lat0, lon0 origin from Ell
    let (lat0, lon0, alt0) = (ell[0].x, ell[0].y, ell[0].z);
    // Get ENU from Ell
    let enu = gps_tform.ellipsoid_to_enu(&ell, lat0, lon0, alt0);
    // Get XYZ from ENU
    let xyz = gps_tform.enu_to_ecef(&enu, lat0, lon0, alt0);
    assert_all_near(&xyz, &ref_xyz, 1e-8);
}

const EAST_OFFSET: f64 = 5.0e5;

// Calculated by COLMAP from GeographicLib (TransverseMercatorProj -l 9 -p 9).
fn wgs84_utm() -> Vec<Vector3d> {
    vec![
        Vector3d::new(
            1.91125018424899e5 + EAST_OFFSET,
            5.335909515367108e6,
            561.1851,
        ),
        Vector3d::new(
            1.91150201163177e5 + EAST_OFFSET,
            5.335932057413140e6,
            561.1509,
        ),
        Vector3d::new(
            2.65520501819149e5 + EAST_OFFSET,
            5.338903134602814e6,
            561.1509,
        ),
    ]
}

// The same with `-e 6378137.0 1.0/298.257222100882711243162837`.
fn grs80_utm() -> Vec<Vector3d> {
    vec![
        Vector3d::new(
            1.91125018426643e5 + EAST_OFFSET,
            5.335909515244992e6,
            561.1851,
        ),
        Vector3d::new(
            1.91150201164921e5 + EAST_OFFSET,
            5.335932057291023e6,
            561.1509,
        ),
        Vector3d::new(
            2.65520501821572e5 + EAST_OFFSET,
            5.338903134480723e6,
            561.1509,
        ),
    ]
}

#[test]
fn gps_ellipsoid_to_utm_wgs84() {
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    let (utm, zone) = gps_tform.ellipsoid_to_utm(&ell3()).unwrap();
    let tolerance = 1e-8; // 10nm
    assert_all_near(&utm, &wgs84_utm(), tolerance);
    assert_eq!(zone, 32);
}

#[test]
fn gps_ellipsoid_to_utm_grs80() {
    let gps_tform = GpsTransform::new(Ellipsoid::Grs80);
    let (utm, zone) = gps_tform.ellipsoid_to_utm(&ell3()).unwrap();
    let tolerance = 1e-8; // 10nm
    assert_all_near(&utm, &grs80_utm(), tolerance);
    assert_eq!(zone, 32);
}

#[test]
fn gps_utm_to_ellipsoid_wgs84() {
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    let ell = gps_tform.utm_to_ellipsoid(&wgs84_utm(), 32, true).unwrap();
    assert_all_near(&ell, &ell3(), 1e-8);
}

#[test]
fn gps_utm_to_ellipsoid_grs80() {
    // As in COLMAP, the GRS80 UTM inputs are converted back with the WGS84 ellipsoid.
    let gps_tform = GpsTransform::new(Ellipsoid::Wgs84);
    let ell = gps_tform.utm_to_ellipsoid(&grs80_utm(), 32, true).unwrap();
    assert_all_near(&ell, &ell3(), 1e-8);
}

#[test]
fn rust_only_gps_utm_rejects_out_of_range_input() {
    let gps_tform = GpsTransform::default();
    assert!(gps_tform
        .ellipsoid_to_utm(&[Vector3d::new(91.0, 0.0, 0.0)])
        .is_err());
    assert!(gps_tform
        .ellipsoid_to_utm(&[Vector3d::new(0.0, -181.0, 0.0)])
        .is_err());
    // Longitude 180 maps to zone 61: an out-of-bounds write in COLMAP, an error here.
    assert!(gps_tform
        .ellipsoid_to_utm(&[Vector3d::new(0.0, 180.0, 0.0)])
        .is_err());
    assert!(gps_tform
        .ellipsoid_to_utm(&[Vector3d::new(0.0, -180.0, 0.0)])
        .is_ok());
    assert!(gps_tform.utm_to_ellipsoid(&[], 0, true).is_err());
    assert!(gps_tform.utm_to_ellipsoid(&[], 61, true).is_err());
}
