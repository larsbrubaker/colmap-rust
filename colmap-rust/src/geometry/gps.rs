//! Port of COLMAP's `colmap/geometry/gps.h` and `gps.cc`: [`GpsTransform`], conversions
//! between ellipsoidal GPS coordinates (lat/lon in degrees, altitude in meters) and ECEF, ENU
//! and UTM. Port of colmap-sharp's `Geometry/GPSTransform.cs`. Tests: `tests/geometry/gps.rs`
//! (gps_test.cc) and `tests/geometry/rust_only_transforms_oracle.rs`.
//!
//! Tier B against the pycolmap oracle (relative 1e-14, or 1e-8 m for coordinates in meters):
//! the transcendentals go through [`crate::math::fns`] (docs/CPP_DIVERGENCES.md entry 1), the
//! macOS wheel fuses some multiply-adds this port does not (entry 80) and `utm_to_ellipsoid`'s
//! latitude can be 1 ulp off (entry 83). The UTM zone is Tier A.

use crate::linalg::{Matrix3d, Vector3d};
use crate::math::{deg_to_rad, fns, rad_to_deg};
use crate::{check_ge, check_le, check_lt, Result};

/// `GPSTransform::Ellipsoid` (`MAKE_ENUM_CLASS(Ellipsoid, 0, GRS80, WGS84)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Ellipsoid {
    /// `GRS80` (0), COLMAP's default.
    #[default]
    Grs80 = 0,
    /// `WGS84` (1).
    Wgs84 = 1,
}

/// Transforms ellipsoidal GPS coordinates to Cartesian coordinate systems and back. Port of
/// `colmap::GPSTransform`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpsTransform {
    /// Semimajor axis.
    a: f64,
    /// Semiminor axis. Kept for parity with COLMAP, which computes but never reads it.
    #[allow(dead_code)]
    b: f64,
    /// Flattening.
    f: f64,
    /// Numerical eccentricity squared.
    e2: f64,
}

impl Default for GpsTransform {
    /// `GPSTransform()`: the GRS80 ellipsoid.
    fn default() -> Self {
        Self::new(Ellipsoid::Grs80)
    }
}

/// UTM series-expansion parameters (COLMAP's anonymous `UTMParams`), in kilometers.
/// Notation from <https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system>.
struct UtmParams {
    /// Powers of n, where `n[i] = n^i`.
    n: [f64; UTM_ORDER + 1],
    alpha: [f64; UTM_ORDER],
    beta: [f64; UTM_ORDER],
    delta: [f64; UTM_ORDER],
    /// Multiplicative factor.
    a: f64,
}

/// Order of the series expansion, determining the precision.
const UTM_ORDER: usize = 4;
/// UTM scale factor at the central meridian.
const UTM_K0: f64 = 0.9996;
/// Easting of the origin in km.
const UTM_E0: f64 = 500.0;

impl UtmParams {
    fn new(a: f64, f: f64) -> Self {
        let mut n = [0.0; UTM_ORDER + 1];
        n[0] = 1.0;
        n[1] = f / (2.0 - f);
        for i in 2..UTM_ORDER + 1 {
            n[i] = n[1] * n[i - 1];
        }
        let alpha = [
            1.0 / 2.0 * n[1] - 2.0 / 3.0 * n[2] + 5.0 / 16.0 * n[3] + 41.0 / 180.0 * n[4],
            13.0 / 48.0 * n[2] - 3.0 / 5.0 * n[3] + 557.0 / 1440.0 * n[4],
            61.0 / 240.0 * n[3] - 103.0 / 140.0 * n[4],
            49561.0 / 161280.0 * n[4],
        ];
        let beta = [
            1.0 / 2.0 * n[1] - 2.0 / 3.0 * n[2] + 37.0 / 96.0 * n[3] - 1.0 / 360.0 * n[4],
            1.0 / 48.0 * n[2] + 1.0 / 15.0 * n[3] - 437.0 / 1440.0 * n[4],
            17.0 / 480.0 * n[3] - 37.0 / 840.0 * n[4],
            4397.0 / 161280.0 * n[4],
        ];
        let delta = [
            2.0 * n[1] - 2.0 / 3.0 * n[2] - 2.0 * n[3] - 116.0 / 45.0 * n[4],
            7.0 / 3.0 * n[2] - 8.0 / 5.0 * n[3] - 227.0 / 45.0 * n[4],
            56.0 / 15.0 * n[3] - 136.0 / 35.0 * n[4],
            4279.0 / 630.0 * n[4],
        ];
        let a = a / (1.0 + n[1]) * (1.0 + n[2] / 4.0 + n[4] / 64.0);
        Self {
            n,
            alpha,
            beta,
            delta,
            a,
        }
    }

    /// Northing of the origin in km: 0 in the northern hemisphere, 10,000 in the southern.
    fn n0(lat_or_hemi: f64) -> f64 {
        if lat_or_hemi > 0.0 {
            0.0
        } else {
            1e4
        }
    }

    fn zone_to_central_meridian(zone: i32) -> f64 {
        f64::from(6 * zone - 183)
    }

    fn meridian_to_zone(meridian: f64) -> i32 {
        ((meridian + 180.0) / 6.0).floor() as i32 + 1
    }
}

/// The ECEF-to-ENU rotation at a reference latitude/longitude (degrees).
fn ecef_to_enu_rotation(ref_lat: f64, ref_lon: f64) -> Matrix3d {
    let cos_lat = fns::cos(deg_to_rad(ref_lat));
    let sin_lat = fns::sin(deg_to_rad(ref_lat));
    let cos_lon = fns::cos(deg_to_rad(ref_lon));
    let sin_lon = fns::sin(deg_to_rad(ref_lon));
    Matrix3d::new(
        -sin_lon,
        cos_lon,
        0.0,
        -sin_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lat,
        cos_lat * cos_lon,
        cos_lat * sin_lon,
        sin_lat,
    )
}

impl GpsTransform {
    /// `GPSTransform(ellipsoid)`.
    pub fn new(ellipsoid: Ellipsoid) -> Self {
        let a = 6378137.0;
        let f = match ellipsoid {
            // COLMAP's literal, kept digit for digit.
            #[allow(clippy::excessive_precision)]
            Ellipsoid::Grs80 => 1.0 / 298.257222100882711243162837,
            Ellipsoid::Wgs84 => 1.0 / 298.257223563,
        };
        let b = (1.0 - f) * a;
        Self {
            a,
            b,
            f,
            e2: f * (2.0 - f),
        }
    }

    /// `EllipsoidToECEF`: (lat, lon, alt) to ECEF.
    pub fn ellipsoid_to_ecef(&self, lat_lon_alt: &[Vector3d]) -> Vec<Vector3d> {
        lat_lon_alt
            .iter()
            .map(|lla| {
                let lat = deg_to_rad(lla.x);
                let lon = deg_to_rad(lla.y);
                let alt = lla.z;
                let sin_lat = fns::sin(lat);
                let sin_lon = fns::sin(lon);
                let cos_lat = fns::cos(lat);
                let cos_lon = fns::cos(lon);
                // Prime vertical radius of curvature.
                let n = self.a / fns::sqrt(1.0 - self.e2 * sin_lat * sin_lat);
                Vector3d::new(
                    (n + alt) * cos_lat * cos_lon,
                    (n + alt) * cos_lat * sin_lon,
                    (n * (1.0 - self.e2) + alt) * sin_lat,
                )
            })
            .collect()
    }

    /// `ECEFToEllipsoid`: ECEF to (lat, lon, alt), iterating latitude and altitude to 1e-12.
    pub fn ecef_to_ellipsoid(&self, xyz_in_ecef: &[Vector3d]) -> Vec<Vector3d> {
        const EPS: f64 = 1e-12;
        xyz_in_ecef
            .iter()
            .map(|p| {
                let (x, y, z) = (p.x, p.y, p.z);
                let radius_xy = fns::sqrt(x * x + y * y);
                // Iteratively solve for latitude and altitude.
                let mut lat = fns::atan2(z, radius_xy);
                let mut alt = 0.0;
                for _ in 0..100 {
                    let sin_lat = fns::sin(lat);
                    let n = self.a / fns::sqrt(1.0 - self.e2 * sin_lat * sin_lat);
                    let prev_alt = alt;
                    alt = radius_xy / fns::cos(lat) - n;
                    let prev_lat = lat;
                    lat = fns::atan((z / radius_xy) * 1.0 / (1.0 - self.e2 * n / (n + alt)));
                    if (prev_lat - lat).abs() < EPS && (prev_alt - alt).abs() < EPS {
                        break;
                    }
                }
                Vector3d::new(rad_to_deg(lat), rad_to_deg(fns::atan2(y, x)), alt)
            })
            .collect()
    }

    /// `EllipsoidToENU`: (lat, lon, alt) to ENU with origin at the reference point.
    pub fn ellipsoid_to_enu(
        &self,
        lat_lon_alt: &[Vector3d],
        ref_lat: f64,
        ref_lon: f64,
        ref_alt: f64,
    ) -> Vec<Vector3d> {
        let xyz_in_ecef = self.ellipsoid_to_ecef(lat_lon_alt);
        let ref_ecef = self.ellipsoid_to_ecef(&[Vector3d::new(ref_lat, ref_lon, ref_alt)])[0];
        self.ecef_to_enu(&xyz_in_ecef, ref_ecef)
    }

    /// `ECEFToENU`: ECEF to ENU with origin at `ref_ecef`.
    /// Reference: <https://en.wikipedia.org/wiki/Geographic_coordinate_conversion>.
    pub fn ecef_to_enu(&self, xyz_in_ecef: &[Vector3d], ref_ecef: Vector3d) -> Vec<Vector3d> {
        // Compute lat/lon of reference point for rotation matrix.
        let ref_ell = self.ecef_to_ellipsoid(&[ref_ecef])[0];
        let r = ecef_to_enu_rotation(ref_ell.x, ref_ell.y);
        xyz_in_ecef.iter().map(|&p| r * (p - ref_ecef)).collect()
    }

    /// `ENUToEllipsoid`: ENU (origin at the reference point) to (lat, lon, alt).
    pub fn enu_to_ellipsoid(
        &self,
        xyz_in_enu: &[Vector3d],
        ref_lat: f64,
        ref_lon: f64,
        ref_alt: f64,
    ) -> Vec<Vector3d> {
        self.ecef_to_ellipsoid(&self.enu_to_ecef(xyz_in_enu, ref_lat, ref_lon, ref_alt))
    }

    /// `ENUToECEF`: ENU (origin at the reference point) to ECEF.
    pub fn enu_to_ecef(
        &self,
        xyz_in_enu: &[Vector3d],
        ref_lat: f64,
        ref_lon: f64,
        ref_alt: f64,
    ) -> Vec<Vector3d> {
        // Compute ECEF coordinates of the reference point.
        let ref_xyz_in_ecef =
            self.ellipsoid_to_ecef(&[Vector3d::new(ref_lat, ref_lon, ref_alt)])[0];
        // ENU to ECEF is the transpose of ECEF to ENU.
        let r = ecef_to_enu_rotation(ref_lat, ref_lon).transpose();
        xyz_in_enu
            .iter()
            .map(|&p| (r * p) + ref_xyz_in_ecef)
            .collect()
    }

    /// `EllipsoidToUTM`: (lat, lon, alt) to UTM (easting, northing, alt) in meters, and the
    /// zone. When the points span several zones, the zone with the most points is used (the
    /// first such zone on ties, as `std::max_element`). 4th-order series expansion.
    ///
    /// Fails on a latitude outside [-90, 90] or a longitude outside [-180, 180], and on a
    /// longitude of exactly 180, which maps to zone 61 (an out-of-bounds write in COLMAP;
    /// docs/CPP_DIVERGENCES.md entry 82).
    pub fn ellipsoid_to_utm(&self, lat_lon_alt: &[Vector3d]) -> Result<(Vec<Vector3d>, i32)> {
        // Reference:
        // https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
        let params = UtmParams::new(self.a / 1000.0, self.f); // Convert to kilometers.

        // Select the predominant zone when points span multiple zones.
        let mut zone_counts = [0usize; 60];
        for lla in lat_lon_alt {
            check_ge!(lla.x, -90.0);
            check_le!(lla.x, 90.0);
            check_ge!(lla.y, -180.0);
            check_le!(lla.y, 180.0);
            let z_index = (UtmParams::meridian_to_zone(lla.y) - 1) as usize;
            check_lt!(
                z_index,
                zone_counts.len(),
                "(longitude 180 maps to zone 61)"
            );
            zone_counts[z_index] += 1;
        }
        let mut best = 0;
        for (i, &count) in zone_counts.iter().enumerate().skip(1) {
            if zone_counts[best] < count {
                best = i;
            }
        }
        let zone = best as i32 + 1;
        let lambda0 = deg_to_rad(UtmParams::zone_to_central_meridian(zone));

        let two_sqrt_n = 2.0 * fns::sqrt(params.n[1]) / (1.0 + params.n[1]);
        let xyz_in_utm = lat_lon_alt
            .iter()
            .map(|lla| {
                let phi = deg_to_rad(lla.x);
                let lambda = deg_to_rad(lla.y);

                let t = fns::sinh(
                    fns::atanh(fns::sin(phi)) - two_sqrt_n * fns::atanh(two_sqrt_n * fns::sin(phi)),
                );
                let xi = fns::atan(t / fns::cos(lambda - lambda0));
                let eta = fns::atanh(fns::sin(lambda - lambda0) / fns::sqrt(1.0 + t * t));

                let mut e = eta;
                let mut n = xi;
                for i in 0..UTM_ORDER {
                    let doubled_index = 2.0 * (i + 1) as f64;
                    e += params.alpha[i]
                        * fns::cos(doubled_index * xi)
                        * fns::sinh(doubled_index * eta);
                    n += params.alpha[i]
                        * fns::sin(doubled_index * xi)
                        * fns::cosh(doubled_index * eta);
                }
                let e = UTM_E0 + UTM_K0 * params.a * e;
                let n = UtmParams::n0(lla.x) + UTM_K0 * params.a * n;
                // Convert to meters.
                Vector3d::new(e * 1000.0, n * 1000.0, lla.z)
            })
            .collect();
        Ok((xyz_in_utm, zone))
    }

    /// `UTMToEllipsoid`: UTM (easting, northing, alt) in meters of the given zone (1..=60)
    /// and hemisphere to (lat, lon, alt).
    pub fn utm_to_ellipsoid(
        &self,
        xyz_in_utm: &[Vector3d],
        zone: i32,
        is_north: bool,
    ) -> Result<Vec<Vector3d>> {
        check_ge!(zone, 1);
        check_le!(zone, 60);

        let params = UtmParams::new(self.a / 1000.0, self.f); // Convert to kilometers.
        let hemisphere = if is_north { 1.0 } else { 0.0 };

        Ok(xyz_in_utm
            .iter()
            .map(|ena| {
                let xi = (ena.y / 1000.0 - UtmParams::n0(hemisphere)) / (UTM_K0 * params.a);
                let eta = (ena.x / 1000.0 - UTM_E0) / (UTM_K0 * params.a);

                let mut xi_prime = 0.0;
                let mut eta_prime = 0.0;
                for i in 0..UTM_ORDER {
                    let doubled_index = 2.0 * (i + 1) as f64;
                    xi_prime += params.beta[i]
                        * fns::sin(doubled_index * xi)
                        * fns::cosh(doubled_index * eta);
                    eta_prime += params.beta[i]
                        * fns::cos(doubled_index * xi)
                        * fns::sinh(doubled_index * eta);
                }
                let xi_prime = xi - xi_prime;
                let eta_prime = eta - eta_prime;
                let chi = fns::asin(fns::sin(xi_prime) / fns::cosh(eta_prime));

                let mut phi = chi;
                for i in 0..UTM_ORDER {
                    let doubled_index = 2.0 * (i + 1) as f64;
                    phi += params.delta[i] * fns::sin(doubled_index * chi);
                }

                let lat = rad_to_deg(phi);
                let lon = UtmParams::zone_to_central_meridian(zone)
                    + rad_to_deg(fns::atan(fns::sinh(eta_prime) / fns::cos(xi_prime)));
                Vector3d::new(lat, lon, ena.z)
            })
            .collect())
    }
}
