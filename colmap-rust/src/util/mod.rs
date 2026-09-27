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
//! - [`file`]: the path-string helpers of `file.h` (extensions).
//! - [`timer`]: `timer.h` — the pausable stopwatch, with a host clock on wasm.
//! - [`threading`]: `GetEffectiveNumThreads` of `threading.h`.
//! - [`cancellation`]: `cancellation.h` — [`cancellation::CancelToken`] and the
//!   [`cancellation::Progress`] callback type.

pub mod cancellation;
pub mod check;
pub mod endian;
pub mod file;
pub mod misc;
pub mod stream_format;
pub mod string;
pub mod threading;
pub mod timer;
pub mod types;
