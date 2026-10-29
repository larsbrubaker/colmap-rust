//! Port of `colmap/scene/reconstruction_manager.h` / `.cc` (and colmap-sharp's
//! `Scene/ReconstructionManager.cs`): the list of [`Reconstruction`]s the incremental mapper
//! produces, one per connected model, read from and written to model folders through
//! [`super::reconstruction_files`]. Tests: `tests/scene/reconstruction_manager.rs`
//! (`reconstruction_manager_test.cc` 1:1) and `tests/scene/rust_only_reconstruction_manager.rs`.
//!
//! Translation notes:
//! - COLMAP holds each model in a `shared_ptr<Reconstruction>` so a mapper can keep using it
//!   while the manager owns it, and its non-const `Get` returns the `shared_ptr&` so a caller
//!   can swap the model. Here the manager owns the models by value: [`ReconstructionManager::get`]
//!   and [`ReconstructionManager::get_mut`] borrow one, and [`ReconstructionManager::set`]
//!   swaps it.
//! - `Get`'s `std::out_of_range` (`vector::at`) is an [`ErrorKind::OutOfRange`] error.
//! - `Write` orders the models by point count with an index tie-break, a deliberate
//!   divergence. COLMAP's `Write` orders them with `std::sort` by descending `NumPoints3D` and
//!   writes them to `0/`, `1/`, ..., leaving the order of equal counts unspecified; here the
//!   sort is stable, so equal-sized models keep their manager order. The tie order decides
//!   which folder a model lands in (CLAUDE.md's stable-sort rule). libc++'s `std::sort` uses
//!   insertion sort for short ranges, which also keeps equal elements in order, so for the
//!   handful of models a mapper produces the result is COLMAP's; only a long run of equal
//!   counts could differ. Evidence: `tests/scene/rust_only_reconstruction_manager.rs`
//!   `rust_only_write_orders_by_point_count_then_index` writes models with point counts 1, 2,
//!   1, 2 and reads back sources 1, 3, 0, 2. Same as colmap-sharp entry 34.

use std::path::Path;

use crate::scene::reconstruction::Reconstruction;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::file::create_dir_if_not_exists;
use crate::Result;

/// Port of `colmap::ReconstructionManager`.
#[derive(Clone, Debug, Default)]
pub struct ReconstructionManager {
    reconstructions: Vec<Reconstruction>,
}

impl ReconstructionManager {
    /// An empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// `Size()`: the number of reconstructions managed.
    pub fn size(&self) -> usize {
        self.reconstructions.len()
    }

    fn out_of_range(&self, idx: usize) -> ColmapError {
        ColmapError::new(
            ErrorKind::OutOfRange,
            format!(
                "Reconstruction index {idx} out of range (size {})",
                self.reconstructions.len()
            ),
        )
    }

    /// `Get(idx)` (const): the reconstruction at `idx`; fails if out of range.
    pub fn get(&self, idx: usize) -> Result<&Reconstruction> {
        self.reconstructions
            .get(idx)
            .ok_or_else(|| self.out_of_range(idx))
    }

    /// `Get(idx)` (non-const): the reconstruction at `idx`, mutable; fails if out of range.
    pub fn get_mut(&mut self, idx: usize) -> Result<&mut Reconstruction> {
        let error = self.out_of_range(idx);
        self.reconstructions.get_mut(idx).ok_or(error)
    }

    /// Assignment through the non-const `Get(idx)`: replaces the reconstruction at `idx` and
    /// returns the old one; fails if out of range.
    pub fn set(&mut self, idx: usize, reconstruction: Reconstruction) -> Result<Reconstruction> {
        let slot = self.get_mut(idx)?;
        Ok(std::mem::replace(slot, reconstruction))
    }

    /// `Add()`: adds a new empty reconstruction and returns its index.
    pub fn add(&mut self) -> usize {
        let idx = self.size();
        self.reconstructions.push(Reconstruction::new());
        idx
    }

    /// `Delete(idx)`: deletes a specific reconstruction.
    pub fn delete(&mut self, idx: usize) -> Result<()> {
        crate::check_lt!(idx, self.reconstructions.len());
        self.reconstructions.remove(idx);
        Ok(())
    }

    /// `Clear()`: deletes all reconstructions.
    pub fn clear(&mut self) {
        self.reconstructions.clear();
    }

    /// `Read(path)`: reads the model folder `path` ([`Reconstruction::read`]) into a new
    /// reconstruction and returns its index. As in COLMAP, the new (empty) reconstruction
    /// stays added when the read fails.
    pub fn read(&mut self, path: &Path) -> Result<usize> {
        let idx = self.add();
        self.reconstructions[idx].read(path)?;
        Ok(idx)
    }

    /// `Write(path)`: writes every reconstruction (binary) into the sub-folders "0", "1",
    /// ... of `path`, the one with the most 3D points first; equal counts keep their order
    /// (see the module header). Each sub-folder is created if needed; `path` must exist.
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut order: Vec<usize> = (0..self.reconstructions.len()).collect();
        // std::sort by descending point count; the stable sort keeps equal counts in index
        // order (see the module header).
        order.sort_by(|&a, &b| {
            self.reconstructions[b]
                .num_points3d()
                .cmp(&self.reconstructions[a].num_points3d())
        });
        for (i, &idx) in order.iter().enumerate() {
            let reconstruction_path = path.join(i.to_string());
            create_dir_if_not_exists(&reconstruction_path)?;
            self.reconstructions[idx].write(&reconstruction_path)?;
        }
        Ok(())
    }
}
