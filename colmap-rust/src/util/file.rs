//! The path-string helpers of COLMAP's `src/colmap/util/file.h/.cc` that need no file
//! system: `HasFileExtension`, `AddFileExtension` and `SplitFileExtension` (the undistorters
//! and image readers name their outputs with them). The file-system functions (`ExistsFile`,
//! `CreateDirIfNotExists`, `GetRecursiveFileList`, downloads, ...) are not ported here: the
//! core crate reads and writes through host-supplied streams so that it runs in the browser.
//! Port of colmap-sharp's `ColmapSharp/Util/FileUtils.cs`, plus `SplitFileExtension`.
//! Tests: `tests/util/file.rs` (the matching `file_test.cc` cases).
//!
//! Tier A (exact): `std::filesystem::path::extension()` is reproduced by hand: the
//! extension starts at the file name's last '.', except that a name whose only '.' is its
//! first character (a dot file), "." and ".." have none. Only '/' separates path
//! components, as for `std::filesystem::path` on POSIX (COLMAP on Windows also splits at
//! '\\'); the core crate's paths are the host's portable, '/'-separated names.

use super::check::Result;
use super::string::string_split;

/// `path.filename()` for a POSIX path string: everything after the last '/'.
fn file_name(path: &str) -> &str {
    match path.rfind('/') {
        Some(slash) => &path[slash + 1..],
        None => path,
    }
}

/// `path.extension()`: from the file name's last '.', or "" (see the header).
fn extension(path: &str) -> &str {
    let name = file_name(path);
    if name == ".." {
        return "";
    }
    match name.rfind('.') {
        Some(dot) if dot > 0 => &name[dot..],
        _ => "",
    }
}

/// Port of `HasFileExtension`: whether `file_name`'s extension equals `ext` lower-cased.
/// Only `ext` is lower-cased, so "a.JPG" does not have ".jpg", as in COLMAP. `ext` must be
/// non-empty and start with '.'.
pub fn has_file_extension(file_name: &str, ext: &str) -> Result<bool> {
    crate::check!(!ext.is_empty());
    crate::check_eq!(ext.as_bytes()[0] as char, '.');
    Ok(extension(file_name) == ext.to_ascii_lowercase())
}

/// Port of `AddFileExtension`: appends `ext` to the path as is.
pub fn add_file_extension(path: &str, ext: &str) -> String {
    format!("{path}{ext}")
}

/// Port of `SplitFileExtension`: `(root, ext)` split at the last '.' of the whole path
/// (not just the file name, as COLMAP does it with `StringSplit(path, ".")`). `ext`
/// includes the '.', and is "" when the path ends in '.' or has no '.'.
pub fn split_file_extension(path: &str) -> Result<(String, String)> {
    let parts = string_split(path, ".");
    crate::check_gt!(parts.len(), 0);
    if parts.len() == 1 {
        return Ok((parts[0].clone(), String::new()));
    }
    let mut root = String::new();
    for part in &parts[..parts.len() - 1] {
        root.push_str(part);
        root.push('.');
    }
    root.pop();
    let last = &parts[parts.len() - 1];
    let ext = if last.is_empty() {
        String::new()
    } else {
        format!(".{last}")
    };
    Ok((root, ext))
}
