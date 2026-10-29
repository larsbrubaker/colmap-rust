// The repository walk the file gates share: `file_compliance.rs` and `markdown_placement.rs` both
// pull this file in with `#[path]`, so the two can never disagree about what the repository is.
//
// The walk starts at the workspace root and reads the file system, not git's index, so a file an
// agent has just written counts before it is committed. Skipped trees are build output, VCS data
// and third-party material that is not ours: target/ (only next to a Cargo.toml, i.e. Cargo's
// build output at a crate or workspace root), .git/, node_modules/, cpp-reference/ and
// ceres-reference/ (COLMAP's and Ceres' sources, fetched by `scripts/fetch-reference.sh`),
// oracle/.venv/, web/pkg/, web/dist/, web/test-results/ and web/playwright-report/ (the browser
// tests' output) and .claude/worktrees/. A `target` directory elsewhere is source and is walked.
// Symbolic links are neither followed nor listed.

use std::fs;
use std::path::{Path, PathBuf};

/// The workspace root: this crate (`colmap-rust`) sits directly inside it.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("colmap-rust sits inside the workspace root")
        .to_path_buf()
}

/// True for a directory (relative to the root, `/`-separated) that the walk skips.
/// `beside_cargo_toml` says whether its parent directory holds a `Cargo.toml`: only then is a
/// `target` directory Cargo's build output rather than a source directory that happens to be
/// called `target`.
pub fn is_skipped_dir(relative: &str, beside_cargo_toml: bool) -> bool {
    let name = relative.rsplit('/').next().unwrap_or(relative);
    (name == "target" && beside_cargo_toml)
        || matches!(name, ".git" | "node_modules")
        || matches!(
            relative,
            "cpp-reference"
                | "ceres-reference"
                | "oracle/.venv"
                | "web/pkg"
                | "web/dist"
                | "web/test-results"
                | "web/playwright-report"
                | ".claude/worktrees"
        )
}

/// Every file the gates check, as (path relative to the root with `/` separators, full path),
/// sorted by path.
pub fn repo_files() -> Vec<(String, PathBuf)> {
    let root = workspace_root();
    let mut paths = Vec::new();
    collect_files(&root, &root, &mut paths);
    let mut files: Vec<(String, PathBuf)> = paths
        .into_iter()
        .map(|path| (relative(&root, &path), path))
        .collect();
    files.sort();
    files
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        let file_type = fs::symlink_metadata(&path).expect("metadata").file_type();
        if file_type.is_dir() {
            let beside_cargo_toml = dir.join("Cargo.toml").is_file();
            if !is_skipped_dir(&relative(root, &path), beside_cargo_toml) {
                collect_files(root, &path, out);
            }
        } else if file_type.is_file() {
            out.push(path);
        }
    }
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("under the root")
        .to_string_lossy()
        .replace('\\', "/")
}
