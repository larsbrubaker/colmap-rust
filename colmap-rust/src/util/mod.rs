//! Port of COLMAP's `src/colmap/util/`: the parts the library needs. Where Rust's standard
//! library replaces a COLMAP helper cleanly, the module header says so (e.g. `StringPrintf`
//! is `format!`, `span`/`filter_view` are slices and `Iterator::filter`, `ThreadPool` is
//! rayon at the call site).
//!
//! - [`check`]: the `THROW_CHECK` family of `logging.h` and [`check::ColmapError`].
//! - [`stream_format`]: C++ `std::ostream` formatting of doubles (`%g`).
//! - [`types`]: `types.h` — id types, `kInvalid*` sentinels, image-pair ids, `PairHash`.
//! - [`string`]: `string.h` — split, trim, replace, case, `StringToDouble`.
//! - [`misc`]: `misc.h` — vector helpers and the CSV list conversions.
//! - [`endian`]: `endian.h` — byte order and little-endian binary stream I/O.
//! - [`line_tokens`]: `std::getline` and `istream >>` token parsing for the text readers.
//! - [`file`](mod@file): the path-string helpers of `file.h` (extensions).
//! - [`timer`]: `timer.h` — the pausable stopwatch, with a host clock on wasm.
//! - [`threading`]: `GetEffectiveNumThreads` of `threading.h`.
//! - [`json`]: a small JSON reader (written here) standing in for `boost::property_tree`'s
//!   `read_json`, with [`ptree_translate`] ptree's `bool`/`double` value translators.
//! - [`cancellation`]: `cancellation.h` — [`cancellation::CancelToken`] and the
//!   [`cancellation::Progress`] callback type.
//! - [`ply`]: `ply.h` — PLY point clouds and meshes, text and binary (readers in `ply` and
//!   `ply_mesh_read`, writers in `ply_write`).

pub mod cancellation;
pub mod check;
pub mod endian;
pub mod file;
pub mod json;
pub mod line_tokens;
pub mod misc;
pub mod ply;
mod ply_mesh_read;
mod ply_write;
pub mod ptree_translate;
pub mod stream_format;
pub mod string;
pub mod threading;
pub mod timer;
pub mod types;
