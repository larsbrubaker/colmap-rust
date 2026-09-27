//! Port of COLMAP's `src/colmap/util/`: so far the error/check machinery of `logging.h`
//! ([`check`]) and the C++ `std::ostream` formatting of doubles ([`stream_format`]) that its
//! messages (and later COLMAP's text writers) depend on.

pub mod check;
pub mod stream_format;
