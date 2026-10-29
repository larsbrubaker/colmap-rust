//! Port of `colmap/scene/reconstruction.cc`, part 4 (and colmap-sharp's
//! `Scene/Reconstruction.Ply.cs`): the [`Reconstruction`] methods that move 3D points to and
//! from PLY point clouds (`ConvertToPLY`, `ImportPLY`) and color them from the images
//! (`ExtractColorsForImage`, `ExtractColorsForAllImages`), plus `CreateImageDirs`. The PLY
//! files themselves are [`crate::util::ply`]. Tests: `tests/scene/reconstruction_ply.rs`
//! (the matching `reconstruction_test.cc` cases).
//!
//! Tier A. Translation notes:
//! - Points are visited in ascending id order (the `scene/reconstruction.rs` header), so
//!   `ConvertToPLY` lists them by id.
//! - The color extractors take the images from the host instead of reading
//!   `path / image.Name()` from disk: `load_image` receives the image's name and returns its
//!   bitmap, or `None` where COLMAP's `Bitmap::Read` fails. COLMAP reads with `as_rgb = true`;
//!   a grey bitmap gives the same bilinear color in all three channels either way.
//! - `ExtractColorsForAllImages` runs on one thread, so its per-point sums are accumulated in
//!   registered-image order: COLMAP's result with `num_threads = 1` (with more threads COLMAP
//!   sums per-thread partials in a scheduling-dependent order). Its `LOG(WARNING)` for an
//!   unreadable image is not emitted (the core has no logger).
//!
//! Both extractor notes are deliberate divergences: image file decoding is not ported, so the
//! host owns image I/O, natively and in the browser (as for `synthesize_images`, whose doc in
//! `scene/synthetic_noise.rs` holds the same reasoning), and the core is single-threaded
//! (CLAUDE.md, Threading). With more threads COLMAP adds the per-thread partial sums in a
//! scheduling-dependent order, which can move a mean that lands near a rounding tie. COLMAP's
//! `ExtractColorsForImage` returns `false` for an unreadable file and
//! `ExtractColorsForAllImages` skips it with a log warning; `None` from `load_image` does the
//! same. Evidence: `tests/scene/reconstruction_ply.rs`
//! `reconstruction_extract_colors_for_all_images` (COLMAP's test, with the deleted image file
//! as a missing map entry) and `tests/scene/rust_only_reconstruction_ply.rs`
//! `rust_only_extract_colors_for_image_colors_black_points_only`.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::linalg::{Vector3d, Vector3ub};
use crate::scene::reconstruction::Reconstruction;
use crate::scene::track::Track;
use crate::sensor::bitmap::Bitmap;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::ply::{read_ply, read_ply_path, PlyPoint};
use crate::util::string::string_split;
use crate::util::types::{ImageId, Point3DId};
use crate::Result;

/// The per-point color sum of `ExtractColorsForAllImages`.
#[derive(Default)]
struct ColorData {
    sum: [f64; 3],
    count: i32,
}

impl Reconstruction {
    /// Port of `ConvertToPLY`: every 3D point's position (narrowed to `f32`) and color.
    pub fn convert_to_ply(&self) -> Vec<PlyPoint> {
        self.points3d
            .values()
            .map(|point3d| PlyPoint {
                x: point3d.xyz.x as f32,
                y: point3d.xyz.y as f32,
                z: point3d.xyz.z as f32,
                r: point3d.color.x,
                g: point3d.color.y,
                b: point3d.color.z,
                ..PlyPoint::default()
            })
            .collect()
    }

    /// Port of `ImportPLY(ply_points)`: replaces the 3D points with the given ones, each with
    /// an empty track. Images keep their point references, as in COLMAP.
    pub fn import_ply(&mut self, ply_points: &[PlyPoint]) -> Result<()> {
        self.points3d.clear();
        for ply_point in ply_points {
            self.add_point3d(
                Vector3d::new(
                    f64::from(ply_point.x),
                    f64::from(ply_point.y),
                    f64::from(ply_point.z),
                ),
                Track::default(),
                Vector3ub::new(ply_point.r, ply_point.g, ply_point.b),
            )?;
        }
        Ok(())
    }

    /// Port of `ImportPLY(path)` on a stream: [`read_ply`], then [`Self::import_ply`]. The
    /// points are cleared first, so a file that fails to read leaves none, as in COLMAP.
    pub fn import_ply_stream<R: std::io::Read + ?Sized>(&mut self, stream: &mut R) -> Result<()> {
        self.points3d.clear();
        let ply_points = read_ply(stream)?;
        self.import_ply(&ply_points)
    }

    /// Port of `ImportPLY(path)`.
    pub fn import_ply_path(&mut self, path: &Path) -> Result<()> {
        self.points3d.clear();
        let ply_points = read_ply_path(path)?;
        self.import_ply(&ply_points)
    }

    /// Port of `ExtractColorsForImage`: colors the still-black 3D points the image observes
    /// with the bilinear color at each observation. Returns `false` if `load_image` has no
    /// bitmap for the image (COLMAP: the file could not be read).
    pub fn extract_colors_for_image(
        &mut self,
        image_id: ImageId,
        load_image: impl FnOnce(&str) -> Option<Bitmap>,
    ) -> Result<bool> {
        let image = self.image(image_id)?;
        let Some(bitmap) = load_image(image.name()) else {
            return Ok(false);
        };
        let observations: Vec<(Point3DId, f64, f64)> = image
            .points2d()
            .iter()
            .filter(|p| p.has_point3d())
            .map(|p| (p.point3d_id, p.xy.x, p.xy.y))
            .collect();

        let black = Vector3ub::zeros();
        for (point3d_id, x, y) in observations {
            let point3d = self.point3d_mut(point3d_id)?;
            if point3d.color == black {
                // COLMAP assumes that the upper left pixel center is (0.5, 0.5).
                if let Some(color) = bitmap.interpolate_bilinear(x - 0.5, y - 0.5) {
                    let color_ub = color.cast::<u8>();
                    point3d.color = Vector3ub::new(color_ub.r, color_ub.g, color_ub.b);
                }
            }
        }
        Ok(true)
    }

    /// Port of `ExtractColorsForAllImages`: each 3D point's color becomes the rounded mean of
    /// its observations' bilinear colors over the registered images `load_image` provides;
    /// a point with no such observation becomes black.
    pub fn extract_colors_for_all_images(
        &mut self,
        mut load_image: impl FnMut(&str) -> Option<Bitmap>,
    ) -> Result<()> {
        // Keyed lookups only; the map is never iterated, so its order cannot reach the output.
        let mut data: HashMap<Point3DId, ColorData> = HashMap::new();
        for image_id in self.reg_image_ids()? {
            let image = self.image(image_id)?;
            let Some(bitmap) = load_image(image.name()) else {
                continue;
            };
            for point2d in image.points2d() {
                if point2d.has_point3d() {
                    // COLMAP assumes that the upper left pixel center is (0.5, 0.5).
                    if let Some(color) =
                        bitmap.interpolate_bilinear(point2d.xy.x - 0.5, point2d.xy.y - 0.5)
                    {
                        let color_data = data.entry(point2d.point3d_id).or_default();
                        color_data.sum[0] += f64::from(color.r);
                        color_data.sum[1] += f64::from(color.g);
                        color_data.sum[2] += f64::from(color.b);
                        color_data.count += 1;
                    }
                }
            }
        }

        for (point3d_id, point3d) in self.points3d.iter_mut() {
            point3d.color = match data.get(point3d_id) {
                Some(color_data) => {
                    let count = f64::from(color_data.count);
                    // `std::round` (half away from zero), then `cast<uint8_t>` of a value
                    // already in [0, 255].
                    let c = |i: usize| (color_data.sum[i] / count).round() as u8;
                    Vector3ub::new(c(0), c(1), c(2))
                }
                None => Vector3ub::zeros(),
            };
        }
        Ok(())
    }

    /// Port of `CreateImageDirs`: creates under `path` every directory an image name has
    /// ("a/b/c.jpg" creates "a" and "a/b").
    pub fn create_image_dirs(&self, path: &Path) -> Result<()> {
        let mut image_dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for image in self.images.values() {
            let name_split = string_split(image.name(), "/");
            if name_split.len() > 1 {
                let mut dir = path.to_path_buf();
                for part in &name_split[..name_split.len() - 1] {
                    dir = dir.join(part);
                    image_dirs.insert(dir.clone());
                }
            }
        }
        for dir in image_dirs {
            // `CreateDirIfNotExists(dir, /*recursive=*/true)`.
            std::fs::create_dir_all(&dir).map_err(|error| {
                ColmapError::new(
                    ErrorKind::RuntimeError,
                    format!("Could not create directory {}: {error}", dir.display()),
                )
            })?;
        }
        Ok(())
    }
}
