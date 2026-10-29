// Markdown placement gate (CLAUDE.md, "Documentation"). Documentation lives in the code, in the
// header or doc comment of the code it describes, where whoever works on that code reads it; a
// deliberate divergence from COLMAP is a comment at the code that differs. A Markdown file is
// allowed only where it is not documentation of code:
//
//   - `README.md`, `CLAUDE.md`, `PORTING_PLAN.md` (the plan for open work) and
//     `THIRD_PARTY_NOTICES.md` at the repository root;
//   - `docs/LICENSE_AUDIT.md`, the legal record of which upstream code may be ported;
//   - a license file anywhere, one whose name is `LICENSE`, `LICENCE` or `COPYING`, or ends in
//     `-LICENSE` / `_LICENSE` (e.g. `colmap-app/assets/fonts/font-awesome-LICENSE.md`);
//   - anything under `.claude/` (agent instructions and skills).
//
// Every other `.md` or `.markdown` file (any case) fails the gate, named with what to do about
// it. The files are those `support/repo_walk.rs` walks, the same as `file_compliance.rs`'s, and
// untracked files count, so a new doc fails before it is committed.

#[path = "support/repo_walk.rs"]
mod repo_walk;

use repo_walk::{is_skipped_dir, repo_files};

/// The Markdown files allowed at fixed paths (relative to the root).
const ALLOWED_PATHS: [&str; 5] = [
    "README.md",
    "CLAUDE.md",
    "PORTING_PLAN.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/LICENSE_AUDIT.md",
];

const WHAT_TO_DO: &str = "Documentation lives in the code: move what each file says into the \
     header or doc comment of the code it describes (a rule that spans several files goes in \
     the owning crate's root or the module's `mod.rs`, and a deliberate divergence from COLMAP \
     goes beside the code that differs), then delete the file. Markdown is allowed only for the \
     root README.md, CLAUDE.md, PORTING_PLAN.md and THIRD_PARTY_NOTICES.md, \
     docs/LICENSE_AUDIT.md, license files and .claude/ (CLAUDE.md, \"Documentation\").";

/// True for a Markdown file by its extension, whatever its case.
fn is_markdown(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => {
            ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("markdown")
        }
        _ => false,
    }
}

/// True for a license file: its name without the extension is `LICENSE`, `LICENCE` or
/// `COPYING`, or ends in one of those after a `-` or `_` (any case).
fn is_license_file(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    let stem = stem.to_ascii_uppercase();
    ["LICENSE", "LICENCE", "COPYING"].iter().any(|word| {
        stem == *word
            || stem
                .strip_suffix(word)
                .is_some_and(|rest| rest.ends_with('-') || rest.ends_with('_'))
    })
}

/// Whether the gate refuses `path` (relative to the root, `/`-separated): a Markdown file
/// outside the allowed places.
fn is_refused(path: &str) -> bool {
    is_markdown(path)
        && !ALLOWED_PATHS.contains(&path)
        && !path.starts_with(".claude/")
        && !is_license_file(path)
}

/// The failure text for the refused paths, or `None` when there are none.
fn failure_message(refused: &[&str]) -> Option<String> {
    if refused.is_empty() {
        return None;
    }
    let list: Vec<String> = refused.iter().map(|path| format!("  {path}")).collect();
    Some(format!(
        "{} Markdown file(s) outside the allowed places:\n{}\n{WHAT_TO_DO}",
        refused.len(),
        list.join("\n")
    ))
}

#[test]
fn rust_only_markdown_lives_only_in_the_allowed_places() {
    let files = repo_files();
    let refused: Vec<&str> = files
        .iter()
        .map(|(path, _)| path.as_str())
        .filter(|path| is_refused(path))
        .collect();
    if let Some(message) = failure_message(&refused) {
        panic!("{message}");
    }
}

#[test]
fn rust_only_markdown_scan_sees_the_allowed_files() {
    // Guards against a walk that silently checks nothing (wrong root, over-broad skips).
    let files = repo_files();
    let paths: Vec<&str> = files.iter().map(|(path, _)| path.as_str()).collect();
    for expected in ["README.md", "CLAUDE.md", "docs/LICENSE_AUDIT.md"] {
        assert!(paths.contains(&expected), "scan missed {expected}");
    }
    assert!(paths.iter().any(|path| path.starts_with(".claude/")));
}

#[test]
fn rust_only_markdown_placement_classifies_examples() {
    for allowed in [
        "README.md",
        "CLAUDE.md",
        "PORTING_PLAN.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/LICENSE_AUDIT.md",
        "colmap-app/assets/fonts/font-awesome-LICENSE.md",
        "third_party/x/LICENSE.md",
        "third_party/x/copying.markdown",
        ".claude/agents/implementer.md",
        ".claude/skills/file-size-refactoring/SKILL.md",
        "colmap-rust/src/lib.rs",
        "docs/notes.txt",
    ] {
        assert!(!is_refused(allowed), "{allowed} should pass");
    }
    for refused in [
        "docs/x.md",
        "docs/DIVERGENCES.md",
        "colmap-rust/README.md",
        "colmap-rust/src/scene/NOTES.MD",
        "web/tests/plan.markdown",
        "docs/LICENSE_HANDLING.md",
        "LICENSE_AUDIT.md",
        "docs/readme.md",
    ] {
        assert!(is_refused(refused), "{refused} should fail");
    }
    // A dot file or a name without an extension is not Markdown.
    assert!(!is_markdown(".md"));
    assert!(!is_markdown("docs/md"));
    // Third-party trees and build output are not walked, so their Markdown never counts.
    assert!(is_skipped_dir("cpp-reference", false));
    assert!(is_skipped_dir("ceres-reference", false));
    assert!(is_skipped_dir("web/node_modules", false));
    assert!(is_skipped_dir("web/playwright-report", false));
    assert!(is_skipped_dir("colmap-app/target", true));
}

#[test]
fn rust_only_markdown_failure_names_the_files_and_what_to_do() {
    assert_eq!(failure_message(&[]), None);
    let message = failure_message(&["docs/x.md"]).expect("a refused file fails");
    assert!(message.starts_with("1 Markdown file(s) outside the allowed places:\n  docs/x.md\n"));
    assert!(message.contains("move what each file says into the header or doc comment"));
    assert!(message.contains("then delete the file"));
}
