//! Port of `colmap/scene/synthetic.h` / `.cc`'s `SynthesizeNoise` and `SynthesizeImages`
//! (colmap-sharp's `Scene/Synthetic.Noise.cs`): Gaussian noise on a synthetic dataset's frame
//! poses, 2D and 3D points and pose priors, and images that render a feature patch at every
//! 2D point. Neighbors: [`super::synthetic`] (the dataset generator) and
//! [`super::synthetic_matches`] (the pose priors' WGS84 conversion). Tests:
//! `tests/scene/synthetic.rs`.
//!
//! Tier A given the PRNG stream; images and 3D points are visited in ascending id order where
//! COLMAP walks hash maps (docs/CPP_DIVERGENCES.md entry 240), so the same draws land on
//! different points.
//!
//! [`synthesize_images`] hands each rendered bitmap to a sink instead of writing
//! `image_path / image.Name()`: the core has no image file I/O (entry 268).

use crate::linalg::{AngleAxisd, Matrix3d, Quaterniond, Vector2d, Vector3d};
use crate::math::fns;
use crate::math::random::{self, libcxx, Mt19937};
use crate::math::random_eigen::random_eigen_vector3d;
use crate::math::utils::deg_to_rad;
use crate::scene::database::Database;
use crate::scene::reconstruction::Reconstruction;
use crate::sensor::bitmap::color::BitmapColor;
use crate::sensor::bitmap::Bitmap;
use crate::Result;

use super::synthetic_matches::{
    pose_prior_position_cartesian_to_wgs84, pose_prior_position_wgs84_to_cartesian,
};

/// Port of `colmap::SyntheticNoiseOptions`, with COLMAP's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticNoiseOptions {
    pub rig_from_world_translation_stddev: f64,
    /// Random rotation in degrees around the z-axis of the rig.
    pub rig_from_world_rotation_stddev: f64,
    pub point3d_stddev: f64,
    pub point2d_stddev: f64,

    /// Translational standard deviation of the prior position in meters.
    pub prior_position_stddev: f64,
    /// Rotational standard deviation of the prior gravity in degrees.
    pub prior_gravity_stddev: f64,
}

impl Default for SyntheticNoiseOptions {
    fn default() -> Self {
        Self {
            rig_from_world_translation_stddev: 0.0,
            rig_from_world_rotation_stddev: 0.0,
            point3d_stddev: 0.0,
            point2d_stddev: 0.0,
            prior_position_stddev: 1.5,
            prior_gravity_stddev: 1.0,
        }
    }
}

/// Port of `colmap::SynthesizeNoise(options, reconstruction)` (no database): perturbs the
/// registered frames' poses, the 2D points and the 3D points, then updates the 3D point
/// errors.
pub fn synthesize_noise(
    options: &SyntheticNoiseOptions,
    reconstruction: &mut Reconstruction,
) -> Result<()> {
    synthesize_noise_impl(options, reconstruction, None)
}

/// Port of `colmap::SynthesizeNoise(options, reconstruction, database)`: as
/// [`synthesize_noise`], and also moves the database's keypoints with the 2D points and
/// perturbs every pose prior's position (adding the variance to its covariance) and gravity.
pub fn synthesize_noise_with_database(
    options: &SyntheticNoiseOptions,
    reconstruction: &mut Reconstruction,
    database: &Database,
) -> Result<()> {
    synthesize_noise_impl(options, reconstruction, Some(database))
}

fn synthesize_noise_impl(
    options: &SyntheticNoiseOptions,
    reconstruction: &mut Reconstruction,
    database: Option<&Database>,
) -> Result<()> {
    crate::check_ge!(options.rig_from_world_translation_stddev, 0.0);
    crate::check_ge!(options.rig_from_world_rotation_stddev, 0.0);
    crate::check_ge!(options.point3d_stddev, 0.0);
    crate::check_ge!(options.point2d_stddev, 0.0);
    crate::check_ge!(options.prior_position_stddev, 0.0);
    crate::check_ge!(options.prior_gravity_stddev, 0.0);

    for frame_id in reconstruction.reg_frame_ids().to_vec() {
        let rig_from_world = reconstruction.frame_mut(frame_id)?.rig_from_world_mut()?;

        if options.rig_from_world_rotation_stddev > 0.0 {
            let angle = random::random_gaussian(0.0, options.rig_from_world_rotation_stddev)
                .clamp(-180.0, 180.0);
            rig_from_world.rotation = rig_from_world.rotation
                * Quaterniond::from_angle_axis(AngleAxisd::new(
                    deg_to_rad(angle),
                    Vector3d::new(0.0, 0.0, 1.0),
                ));
        }

        if options.rig_from_world_translation_stddev > 0.0 {
            let x = random::random_gaussian(0.0, options.rig_from_world_translation_stddev);
            let y = random::random_gaussian(0.0, options.rig_from_world_translation_stddev);
            let z = random::random_gaussian(0.0, options.rig_from_world_translation_stddev);
            rig_from_world.translation += Vector3d::new(x, y, z);
        }
    }

    if options.point2d_stddev > 0.0 {
        let image_ids: Vec<_> = reconstruction.images().keys().copied().collect();
        for image_id in image_ids {
            let image = reconstruction.image_mut(image_id)?;
            for point2d in image.points2d_mut().iter_mut() {
                let x = random::random_gaussian(0.0, options.point2d_stddev);
                let y = random::random_gaussian(0.0, options.point2d_stddev);
                point2d.xy += Vector2d::new(x, y);
            }
            if let Some(database) = database {
                let mut keypoints = database.read_keypoints(image_id)?;
                for (point2d_idx, keypoint) in keypoints.iter_mut().enumerate() {
                    // C++ indexes `image.Point2D(point2D_idx)` unchecked; a database with more
                    // keypoints than the image has 2D points fails here instead.
                    let point2d = image.points2d().get(point2d_idx).ok_or_else(|| {
                        crate::util::check::ColmapError::new(
                            crate::util::check::ErrorKind::InvalidArgument,
                            format!(
                                "SynthesizeNoise: image {image_id} has fewer 2D points than \
                                 keypoints"
                            ),
                        )
                    })?;
                    keypoint.x = point2d.xy.x as f32;
                    keypoint.y = point2d.xy.y as f32;
                }
                database.update_keypoints(image.image_id(), &keypoints)?;
            }
        }
    }

    if options.point3d_stddev > 0.0 {
        for point3d_id in reconstruction.point3d_ids() {
            let x = random::random_gaussian(0.0, options.point3d_stddev);
            let y = random::random_gaussian(0.0, options.point3d_stddev);
            let z = random::random_gaussian(0.0, options.point3d_stddev);
            let point3d = reconstruction.point3d_mut(point3d_id)?;
            point3d.xyz += Vector3d::new(x, y, z);
        }
    }

    if let Some(database) = database {
        if options.prior_position_stddev > 0.0 || options.prior_gravity_stddev > 0.0 {
            add_pose_prior_noise(options, database)?;
        }
    }

    reconstruction.update_point3d_errors()
}

/// The pose prior half of `SynthesizeNoise`.
fn add_pose_prior_noise(options: &SyntheticNoiseOptions, database: &Database) -> Result<()> {
    for mut pose_prior in database.read_all_pose_priors()? {
        if options.prior_position_stddev > 0.0 {
            let prior_in_wgs84 = pose_prior.coordinate_system
                == crate::geometry::pose_prior::CoordinateSystem::Wgs84;
            if prior_in_wgs84 {
                pose_prior_position_wgs84_to_cartesian(&mut pose_prior);
            }
            let x = random::random_gaussian(0.0, options.prior_position_stddev);
            let y = random::random_gaussian(0.0, options.prior_position_stddev);
            let z = random::random_gaussian(0.0, options.prior_position_stddev);
            pose_prior.position += Vector3d::new(x, y, z);
            if !pose_prior.has_position_cov() {
                pose_prior.position_covariance = Matrix3d::zeros();
            }
            let variance = options.prior_position_stddev * options.prior_position_stddev;
            pose_prior.position_covariance += Matrix3d::identity() * variance;
            if prior_in_wgs84 {
                pose_prior_position_cartesian_to_wgs84(&mut pose_prior);
            }
        }
        if options.prior_gravity_stddev > 0.0 {
            let angle = random::random_gaussian(0.0, deg_to_rad(options.prior_gravity_stddev));
            let axis = pose_prior
                .gravity
                .cross(random_eigen_vector3d())
                .normalized();
            // Eigen's `AngleAxisd * Vector3d` applies `toRotationMatrix()`.
            pose_prior.gravity = (AngleAxisd::new(angle, axis).to_rotation_matrix()
                * pose_prior.gravity)
                .normalized();
        }
        database.update_pose_prior(&pose_prior)?;
    }
    Ok(())
}

/// Port of `colmap::SyntheticImageOptions`, with COLMAP's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticImageOptions {
    pub feature_peak_radius: i32,
    pub feature_patch_radius: i32,
    pub feature_patch_max_brightness: i32,
}

impl Default for SyntheticImageOptions {
    fn default() -> Self {
        Self {
            feature_peak_radius: 2,
            feature_patch_radius: 15,
            feature_patch_max_brightness: 128,
        }
    }
}

/// Port of `colmap::SynthesizeImages`: generates patches with a dark background and a bright
/// feature peak for each 2D point in an image. The color of the peak and the pattern of the
/// background is unique per 3D point. Notice that this approach does not result in perfect
/// feature detections and matches due to overlapping patches, etc.
///
/// Each image's RGB bitmap goes to `write_image(image_name, bitmap)`, in ascending image id
/// order, where COLMAP writes it to `image_path / image.Name()` (entry 268).
pub fn synthesize_images(
    options: &SyntheticImageOptions,
    reconstruction: &Reconstruction,
    mut write_image: impl FnMut(&str, Bitmap),
) -> Result<()> {
    crate::check_gt!(options.feature_patch_radius, 0);
    crate::check_lt!(options.feature_peak_radius, options.feature_patch_radius);
    crate::check_gt!(options.feature_patch_max_brightness, 0);
    crate::check_lt!(options.feature_patch_max_brightness, 255);

    // `std::sqrt(2 * r * r)`: the product is an `int`.
    let patch_radius = fns::sqrt(f64::from(
        2 * options.feature_patch_radius * options.feature_patch_radius,
    ));

    let mut total_num_descriptors: usize = 0;
    for image in reconstruction.images().values() {
        let camera = reconstruction.camera(image.camera_id())?;
        // `static_cast<int>(camera.width)`; widths beyond `i32::MAX` saturate (UB in C++).
        let width = camera.width as i32;
        let height = camera.height as i32;

        let mut bitmap = Bitmap::new(width, height, true);
        bitmap.fill(BitmapColor::new(0, 0, 0))?;

        for point2d in image.points2d() {
            // `static_cast<int>(std::round(x))`: round half away from zero, then truncate.
            let x = point2d.xy.x.round() as i32;
            let y = point2d.xy.y.round() as i32;
            if x < 0 || y < 0 || x >= width || y >= height {
                continue;
            }

            // `std::mt19937`'s seed is 32 bits on libc++; C++ converts the id the same way.
            let seed = if point2d.has_point3d() {
                point2d.point3d_id as u32
            } else {
                total_num_descriptors += 1;
                (reconstruction.num_points3d() + total_num_descriptors) as u32
            };
            let mut feature_generator = Mt19937::new(seed);

            draw_patch(
                options,
                &mut bitmap,
                x,
                y,
                patch_radius,
                &mut feature_generator,
            );
            draw_peak(options, &mut bitmap, x, y, &mut feature_generator);
        }

        write_image(image.name(), bitmap);
    }
    Ok(())
}

/// Draws a circular patch around the feature with a unique pattern with the aim of producing
/// a unique feature descriptor. The pattern is a bit darker than the peak, so the keypoint is
/// detected at the center.
fn draw_patch(
    options: &SyntheticImageOptions,
    bitmap: &mut Bitmap,
    x: i32,
    y: i32,
    patch_radius: f64,
    feature_generator: &mut Mt19937,
) {
    let width = bitmap.width();
    let height = bitmap.height();
    let patch_minx = (x - options.feature_patch_radius).max(0);
    let patch_maxx = (x + options.feature_patch_radius).min(width);
    let patch_miny = (y - options.feature_patch_radius).max(0);
    let patch_maxy = (y + options.feature_patch_radius).min(height);
    for py in patch_miny..patch_maxy {
        for px in patch_minx..patch_maxx {
            let radius = fns::sqrt(f64::from((px - x) * (px - x) + (py - y) * (py - y)));
            if radius > f64::from(options.feature_patch_radius) {
                continue;
            }
            // Adjust the brightness so it fades out to the edge of the patch. The `int`
            // distribution's upper bound is the `double` expression truncated.
            let max_brightness = ((1.0 - radius / patch_radius)
                * f64::from(options.feature_patch_max_brightness))
                as i32;
            let brightness = libcxx::uniform_int(feature_generator, 0i32, max_brightness);
            bitmap.set_pixel(px, py, BitmapColor::gray(brightness as u8));
        }
    }
}

/// Draws a small, bright peak around the feature for keypoint detection.
fn draw_peak(
    options: &SyntheticImageOptions,
    bitmap: &mut Bitmap,
    x: i32,
    y: i32,
    feature_generator: &mut Mt19937,
) {
    let width = bitmap.width();
    let height = bitmap.height();
    let peak_minx = (x - options.feature_peak_radius).max(0);
    let peak_maxx = (x + options.feature_peak_radius).min(width);
    let peak_miny = (y - options.feature_peak_radius).max(0);
    let peak_maxy = (y + options.feature_peak_radius).min(height);
    // `BitmapColor<uint8_t>(dist(g), dist(g), dist(g))`: drawn left to right, as clang
    // evaluates constructor arguments.
    let min = options.feature_patch_max_brightness;
    let r = libcxx::uniform_int(feature_generator, min, 255i32) as u8;
    let g = libcxx::uniform_int(feature_generator, min, 255i32) as u8;
    let b = libcxx::uniform_int(feature_generator, min, 255i32) as u8;
    let peak_color = BitmapColor::new(r, g, b);
    for py in peak_miny..peak_maxy {
        for px in peak_minx..peak_maxx {
            bitmap.set_pixel(px, py, peak_color);
        }
    }
}
