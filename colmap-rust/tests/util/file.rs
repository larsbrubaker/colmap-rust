// Port of the HasFileExtension, AddFileExtension and SplitFileExtension cases of COLMAP's
// src/colmap/util/file_test.cc, 1:1 (Suite_Name -> suite_name), testing
// colmap_rust::util::file. Tier A (exact). The other file_test.cc cases exercise file-system
// functions that are not ported (src/util/file.rs header).

use colmap_rust::util::file::*;

#[test]
fn has_file_extension_nominal() {
    assert!(!has_file_extension("", ".jpg").unwrap());
    assert!(!has_file_extension("testjpg", ".jpg").unwrap());
    assert!(has_file_extension("test.jpg", ".jpg").unwrap());
    assert!(has_file_extension("test.jpg", ".Jpg").unwrap());
    assert!(has_file_extension("test.jpg", ".JPG").unwrap());
    assert!(has_file_extension("test.", ".").unwrap());
}

#[test]
fn add_file_extension_nominal() {
    assert_eq!(add_file_extension("test", ".txt"), "test.txt");
    assert_eq!(add_file_extension("test.jpg", ".txt"), "test.jpg.txt");
}

#[test]
fn split_file_extension_nominal() {
    let split = |path: &str| split_file_extension(path).unwrap();
    assert_eq!(split(""), ("".into(), "".into()));
    assert_eq!(split("."), ("".into(), "".into()));
    assert_eq!(split("file"), ("file".into(), "".into()));
    assert_eq!(split("file."), ("file".into(), "".into()));
    assert_eq!(split("file.jpg"), ("file".into(), ".jpg".into()));
    assert_eq!(split("dir/file.jpg"), ("dir/file".into(), ".jpg".into()));
    assert_eq!(split("/dir/file.jpg"), ("/dir/file".into(), ".jpg".into()));
    assert_eq!(
        split("dir/file.suffix.jpg"),
        ("dir/file.suffix".into(), ".jpg".into())
    );
    assert_eq!(
        split("dir.suffix/file.suffix.jpg"),
        ("dir.suffix/file.suffix".into(), ".jpg".into())
    );
    assert_eq!(
        split("dir.suffix/file."),
        ("dir.suffix/file".into(), "".into())
    );
    assert_eq!(
        split("./dir.suffix/file."),
        ("./dir.suffix/file".into(), "".into())
    );
}

// Rust-only: HasFileExtension's THROW_CHECKs on `ext` and the std::filesystem extension
// rules for dot files, "." and "..".
#[test]
fn rust_only_has_file_extension_edge_cases() {
    assert!(has_file_extension("test.jpg", "").is_err());
    assert!(has_file_extension("test.jpg", "jpg").is_err());
    assert!(!has_file_extension(".jpg", ".jpg").unwrap());
    assert!(has_file_extension("dir/.hidden.jpg", ".jpg").unwrap());
    assert!(!has_file_extension("dir.jpg/file", ".jpg").unwrap());
    assert!(!has_file_extension("..", ".").unwrap());
    assert!(!has_file_extension(".", ".").unwrap());
    // Only `ext` is lower-cased, as in COLMAP.
    assert!(!has_file_extension("test.JPG", ".jpg").unwrap());
}
