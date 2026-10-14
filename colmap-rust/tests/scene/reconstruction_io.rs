// Port of colmap/scene/reconstruction_io_test.cc, 1:1 (same values and checks) for the binary
// and text model readers and writers (colmap_rust::scene::reconstruction_io_binary,
// reconstruction_io_text and reconstruction_io_text_write). The TEST_P suite
// ParameterizedReaderWriterTests, instantiated over the four ReaderWriters (text and binary,
// on an in-memory stream and on files), becomes one test per instantiation calling a shared
// body: `reader_writer_tests_<test>_<reader_writer>`. A `std::stringstream` is a `Vec<u8>`
// written to and then read from the start. TextIO.LocaleIndependentRoundtrip sets a global
// C++ locale with a comma decimal point first; Rust formatting and parsing have no locale,
// so the port keeps the test's checks (dots, no commas, the roundtrip) without that setup.
// The Export* tests (reconstruction_io.cc's NVM, Cam, Recon3D, Bundler, PLY and VRML
// exporters) land with those exporters (PORTING_PLAN.md, Phase 4).

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::reconstruction_io_binary as bin;
use colmap_rust::scene::reconstruction_io_text as txt;
use colmap_rust::scene::reconstruction_io_text_write as txtw;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};

/// COLMAP's `CreateTestDir()`: a fresh, empty directory per call.
pub(crate) fn create_test_dir() -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "colmap_rust_scene_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The five files' bytes, by stem: rigs, cameras, frames, images, points3D.
const STEMS: [&str; 5] = ["rigs", "cameras", "frames", "images", "points3D"];

/// The test's `ReaderWriter` interface; `idx` is the file's index in [`STEMS`].
trait ReaderWriter {
    fn read(&mut self, idx: usize, reconstruction: &mut Reconstruction);
    fn write(&mut self, idx: usize, reconstruction: &Reconstruction);
    fn bytes(&self, idx: usize) -> Vec<u8>;
}

type ReadFn = fn(&mut Reconstruction, &mut &[u8]) -> colmap_rust::Result<()>;
type WriteFn = fn(&Reconstruction, &mut Vec<u8>) -> colmap_rust::Result<()>;

const TEXT_READERS: [ReadFn; 5] = [
    |r, s| txt::read_rigs_text(r, s),
    |r, s| txt::read_cameras_text(r, s),
    |r, s| txt::read_frames_text(r, s),
    |r, s| txt::read_images_text(r, s),
    |r, s| txt::read_points3d_text(r, s),
];
const TEXT_WRITERS: [WriteFn; 5] = [
    |r, s| txtw::write_rigs_text(r, s),
    |r, s| txtw::write_cameras_text(r, s),
    |r, s| txtw::write_frames_text(r, s),
    |r, s| txtw::write_images_text(r, s),
    |r, s| txtw::write_points3d_text(r, s),
];
const BINARY_READERS: [ReadFn; 5] = [
    |r, s| bin::read_rigs_binary(r, s),
    |r, s| bin::read_cameras_binary(r, s),
    |r, s| bin::read_frames_binary(r, s),
    |r, s| bin::read_images_binary(r, s),
    |r, s| bin::read_points3d_binary(r, s),
];
const BINARY_WRITERS: [WriteFn; 5] = [
    |r, s| bin::write_rigs_binary(r, s),
    |r, s| bin::write_cameras_binary(r, s),
    |r, s| bin::write_frames_binary(r, s),
    |r, s| bin::write_images_binary(r, s),
    |r, s| bin::write_points3d_binary(r, s),
];

type ReadPathFn = fn(&mut Reconstruction, &std::path::Path) -> colmap_rust::Result<()>;
type WritePathFn = fn(&Reconstruction, &std::path::Path) -> colmap_rust::Result<()>;

const TEXT_PATH_READERS: [ReadPathFn; 5] = [
    txt::read_rigs_text_path,
    txt::read_cameras_text_path,
    txt::read_frames_text_path,
    txt::read_images_text_path,
    txt::read_points3d_text_path,
];
const TEXT_PATH_WRITERS: [WritePathFn; 5] = [
    txtw::write_rigs_text_path,
    txtw::write_cameras_text_path,
    txtw::write_frames_text_path,
    txtw::write_images_text_path,
    txtw::write_points3d_text_path,
];
const BINARY_PATH_READERS: [ReadPathFn; 5] = [
    bin::read_rigs_binary_path,
    bin::read_cameras_binary_path,
    bin::read_frames_binary_path,
    bin::read_images_binary_path,
    bin::read_points3d_binary_path,
];
const BINARY_PATH_WRITERS: [WritePathFn; 5] = [
    bin::write_rigs_binary_path,
    bin::write_cameras_binary_path,
    bin::write_frames_binary_path,
    bin::write_images_binary_path,
    bin::write_points3d_binary_path,
];

/// `ReaderWriter{Text,Binary}StringStream`.
struct StringStream {
    readers: [ReadFn; 5],
    writers: [WriteFn; 5],
    streams: [Vec<u8>; 5],
}

impl ReaderWriter for StringStream {
    fn read(&mut self, idx: usize, reconstruction: &mut Reconstruction) {
        (self.readers[idx])(reconstruction, &mut self.streams[idx].as_slice()).unwrap();
    }
    fn write(&mut self, idx: usize, reconstruction: &Reconstruction) {
        (self.writers[idx])(reconstruction, &mut self.streams[idx]).unwrap();
    }
    fn bytes(&self, idx: usize) -> Vec<u8> {
        self.streams[idx].clone()
    }
}

/// `ReaderWriter{Text,Binary}FileStream`.
struct FileStream {
    readers: [ReadPathFn; 5],
    writers: [WritePathFn; 5],
    paths: Vec<PathBuf>,
}

impl FileStream {
    fn new(ext: &str, readers: [ReadPathFn; 5], writers: [WritePathFn; 5]) -> Self {
        let test_dir = create_test_dir();
        let paths = STEMS
            .iter()
            .map(|stem| test_dir.join(format!("{stem}.{ext}")))
            .collect();
        Self {
            readers,
            writers,
            paths,
        }
    }
}

impl ReaderWriter for FileStream {
    fn read(&mut self, idx: usize, reconstruction: &mut Reconstruction) {
        (self.readers[idx])(reconstruction, &self.paths[idx]).unwrap();
    }
    fn write(&mut self, idx: usize, reconstruction: &Reconstruction) {
        (self.writers[idx])(reconstruction, &self.paths[idx]).unwrap();
    }
    fn bytes(&self, idx: usize) -> Vec<u8> {
        std::fs::read(&self.paths[idx]).unwrap()
    }
}

fn text_string_stream() -> Box<dyn ReaderWriter> {
    Box::new(StringStream {
        readers: TEXT_READERS,
        writers: TEXT_WRITERS,
        streams: Default::default(),
    })
}

fn binary_string_stream() -> Box<dyn ReaderWriter> {
    Box::new(StringStream {
        readers: BINARY_READERS,
        writers: BINARY_WRITERS,
        streams: Default::default(),
    })
}

fn text_file_stream() -> Box<dyn ReaderWriter> {
    Box::new(FileStream::new("txt", TEXT_PATH_READERS, TEXT_PATH_WRITERS))
}

fn binary_file_stream() -> Box<dyn ReaderWriter> {
    Box::new(FileStream::new(
        "bin",
        BINARY_PATH_READERS,
        BINARY_PATH_WRITERS,
    ))
}

const RIGS: usize = 0;
const CAMERAS: usize = 1;
const FRAMES: usize = 2;
const IMAGES: usize = 3;
const POINTS3D: usize = 4;

fn roundtrip(mut reader_writer: Box<dyn ReaderWriter>) {
    let mut orig = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 3,
        num_cameras_per_rig: 4,
        num_frames_per_rig: 5,
        num_points3d: 321,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut orig).unwrap();

    for idx in [RIGS, CAMERAS, FRAMES, IMAGES, POINTS3D] {
        reader_writer.write(idx, &orig);
        assert!(!reader_writer.bytes(idx).is_empty());
    }

    let mut test = Reconstruction::new();
    reader_writer.read(CAMERAS, &mut test);
    assert_eq!(orig.cameras(), test.cameras());
    reader_writer.read(RIGS, &mut test);
    assert_eq!(orig.rigs(), test.rigs());
    reader_writer.read(FRAMES, &mut test);
    assert_eq!(orig.frames(), test.frames());
    reader_writer.read(IMAGES, &mut test);
    assert_eq!(orig.images(), test.images());
    reader_writer.read(POINTS3D, &mut test);
    assert_eq!(orig.points3d(), test.points3d());
}

fn legacy_without_rigs_and_frames(mut reader_writer: Box<dyn ReaderWriter>) {
    let mut orig = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 3,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 5,
        num_points3d: 321,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut orig).unwrap();

    for idx in [CAMERAS, IMAGES, POINTS3D] {
        reader_writer.write(idx, &orig);
        assert!(!reader_writer.bytes(idx).is_empty());
    }

    let mut test = Reconstruction::new();
    reader_writer.read(CAMERAS, &mut test);
    assert_eq!(orig.cameras(), test.cameras());
    reader_writer.read(IMAGES, &mut test);
    assert_eq!(orig.rigs(), test.rigs());
    assert_eq!(orig.frames(), test.frames());
    assert_eq!(orig.images(), test.images());
    reader_writer.read(POINTS3D, &mut test);
    assert_eq!(orig.points3d(), test.points3d());
}

#[test]
fn reader_writer_tests_roundtrip_text_string_stream() {
    roundtrip(text_string_stream());
}

#[test]
fn reader_writer_tests_roundtrip_binary_string_stream() {
    roundtrip(binary_string_stream());
}

#[test]
fn reader_writer_tests_roundtrip_text_file_stream() {
    roundtrip(text_file_stream());
}

#[test]
fn reader_writer_tests_roundtrip_binary_file_stream() {
    roundtrip(binary_file_stream());
}

#[test]
fn reader_writer_tests_legacy_without_rigs_and_frames_text_string_stream() {
    legacy_without_rigs_and_frames(text_string_stream());
}

#[test]
fn reader_writer_tests_legacy_without_rigs_and_frames_binary_string_stream() {
    legacy_without_rigs_and_frames(binary_string_stream());
}

#[test]
fn reader_writer_tests_legacy_without_rigs_and_frames_text_file_stream() {
    legacy_without_rigs_and_frames(text_file_stream());
}

#[test]
fn reader_writer_tests_legacy_without_rigs_and_frames_binary_file_stream() {
    legacy_without_rigs_and_frames(binary_file_stream());
}

#[test]
fn text_io_locale_independent_roundtrip() {
    let mut orig = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 2,
        num_frames_per_rig: 3,
        num_points3d: 50,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut orig).unwrap();

    let mut rw = text_string_stream();
    for idx in [RIGS, CAMERAS, FRAMES, IMAGES, POINTS3D] {
        rw.write(idx, &orig);
    }

    // Verify written float data uses dot, not comma, as decimal separator.
    // Check a data line (skip comment lines starting with '#').
    let cameras = String::from_utf8(rw.bytes(CAMERAS)).unwrap();
    for line in cameras.lines() {
        if !line.is_empty() && !line.starts_with('#') {
            assert!(line.contains('.'), "{line}");
            assert!(!line.contains(','), "{line}");
        }
    }

    let mut test = Reconstruction::new();
    rw.read(CAMERAS, &mut test);
    assert_eq!(orig.cameras(), test.cameras());
    rw.read(RIGS, &mut test);
    assert_eq!(orig.rigs(), test.rigs());
    rw.read(FRAMES, &mut test);
    assert_eq!(orig.frames(), test.frames());
    rw.read(IMAGES, &mut test);
    assert_eq!(orig.images(), test.images());
    rw.read(POINTS3D, &mut test);
    assert_eq!(orig.points3d(), test.points3d());
}
