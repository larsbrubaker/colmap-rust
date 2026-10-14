//! Port of `colmap/scene/reconstruction.cc`, part 4: the [`Reconstruction`] methods that read
//! and write a model folder — `Read` (format auto-detected), `Write`, `ReadText`,
//! `ReadBinary`, `WriteText` and `WriteBinary` — on top of the per-file readers and writers in
//! [`super::reconstruction_io_binary`], [`super::reconstruction_io_text`] and
//! [`super::reconstruction_io_text_write`]. Port of colmap-sharp's `Scene/Reconstruction.IO.cs`.
//! Tests: `tests/scene/reconstruction_synthetic.rs` (`reconstruction_test.cc` 1:1) and
//! `tests/scene/rust_only_reconstruction_io.rs`.

use std::path::Path;

use crate::scene::reconstruction::Reconstruction;
use crate::scene::reconstruction_io_binary::{
    read_cameras_binary_path, read_frames_binary_path, read_images_binary_path,
    read_points3d_binary_path, read_rigs_binary_path, write_cameras_binary_path,
    write_frames_binary_path, write_images_binary_path, write_points3d_binary_path,
    write_rigs_binary_path,
};
use crate::scene::reconstruction_io_text::{
    read_cameras_text_path, read_frames_text_path, read_images_text_path, read_points3d_text_path,
    read_rigs_text_path,
};
use crate::scene::reconstruction_io_text_write::{
    write_cameras_text_path, write_frames_text_path, write_images_text_path,
    write_points3d_text_path, write_rigs_text_path,
};
use crate::scene::reconstruction_io_utils::check_dir_exists;
use crate::util::check::{ColmapError, ErrorKind};
use crate::Result;

fn exists_all(path: &Path, ext: &str) -> bool {
    ["cameras", "images", "points3D"]
        .iter()
        .all(|stem| path.join(format!("{stem}.{ext}")).is_file())
}

impl Reconstruction {
    /// `Read(path)`: the binary model in `path` if its `cameras.bin`, `images.bin` and
    /// `points3D.bin` exist, else the text model if the `.txt` ones do.
    pub fn read(&mut self, path: &Path) -> Result<()> {
        if exists_all(path, "bin") {
            self.read_binary(path)
        } else if exists_all(path, "txt") {
            self.read_text(path)
        } else {
            Err(ColmapError::new(
                ErrorKind::InvalidArgument,
                format!(
                    "rigs, cameras, frames, images, points3D files do not exist at \"{}\"",
                    path.display()
                ),
            ))
        }
    }

    /// `Write(path)`: the binary model.
    pub fn write(&self, path: &Path) -> Result<()> {
        self.write_binary(path)
    }

    /// Empties the model before a read. COLMAP's `ReadText` / `ReadBinary` clear the maps but
    /// keep the registered-frame list and counts, so reading into a used reconstruction left
    /// them stale; here everything is reset (entry 242).
    fn clear_for_read(&mut self) {
        *self = Reconstruction::new();
    }

    /// `ReadText(path)`: `cameras.txt`, `rigs.txt` and `frames.txt` if present (a model from
    /// before rigs and frames gets trivial ones), `images.txt` and `points3D.txt`.
    pub fn read_text(&mut self, path: &Path) -> Result<()> {
        self.clear_for_read();
        read_cameras_text_path(self, &path.join("cameras.txt"))?;
        let rigs_path = path.join("rigs.txt");
        if rigs_path.is_file() {
            read_rigs_text_path(self, &rigs_path)?;
        }
        let frames_path = path.join("frames.txt");
        if frames_path.is_file() {
            read_frames_text_path(self, &frames_path)?;
        }
        read_images_text_path(self, &path.join("images.txt"))?;
        read_points3d_text_path(self, &path.join("points3D.txt"))
    }

    /// `ReadBinary(path)`: as [`Reconstruction::read_text`] for the `.bin` files.
    pub fn read_binary(&mut self, path: &Path) -> Result<()> {
        self.clear_for_read();
        read_cameras_binary_path(self, &path.join("cameras.bin"))?;
        let rigs_path = path.join("rigs.bin");
        if rigs_path.is_file() {
            read_rigs_binary_path(self, &rigs_path)?;
        }
        let frames_path = path.join("frames.bin");
        if frames_path.is_file() {
            read_frames_binary_path(self, &frames_path)?;
        }
        read_images_binary_path(self, &path.join("images.bin"))?;
        read_points3d_binary_path(self, &path.join("points3D.bin"))
    }

    /// `WriteText(path)`: the five `.txt` files into the existing directory `path`.
    pub fn write_text(&self, path: &Path) -> Result<()> {
        check_dir_exists(path)?;
        write_rigs_text_path(self, &path.join("rigs.txt"))?;
        write_cameras_text_path(self, &path.join("cameras.txt"))?;
        write_frames_text_path(self, &path.join("frames.txt"))?;
        write_images_text_path(self, &path.join("images.txt"))?;
        write_points3d_text_path(self, &path.join("points3D.txt"))
    }

    /// `WriteBinary(path)`: the five `.bin` files into the existing directory `path`.
    pub fn write_binary(&self, path: &Path) -> Result<()> {
        check_dir_exists(path)?;
        write_rigs_binary_path(self, &path.join("rigs.bin"))?;
        write_cameras_binary_path(self, &path.join("cameras.bin"))?;
        write_frames_binary_path(self, &path.join("frames.bin"))?;
        write_images_binary_path(self, &path.join("images.bin"))?;
        write_points3d_binary_path(self, &path.join("points3D.bin"))
    }
}
