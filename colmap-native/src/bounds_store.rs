// Window-bounds persistence for the desktop shell: agg-gui-shell's `WindowBoundsStore` backed
// by a one-line text file ("<width> <height> <maximized 0|1>", physical pixels) in the user's
// config directory. The shell sanitizes whatever is loaded, so a corrupt or stale file only
// ever costs the saved size, never startup.

use std::path::PathBuf;

use agg_gui_shell::{SavedBounds, WindowBoundsStore};

/// `<config dir>/colmap-rust/window.txt`, or `None` when no config directory is known.
pub fn default_bounds_path() -> Option<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join("Library").join("Application Support"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    };
    base.map(|b| b.join("colmap-rust").join("window.txt"))
}

/// Parse the file's single line.
pub fn parse_bounds(text: &str) -> Option<SavedBounds> {
    let mut parts = text.split_whitespace();
    let width = parts.next()?.parse().ok()?;
    let height = parts.next()?.parse().ok()?;
    let maximized = parts.next()? == "1";
    Some(SavedBounds {
        width,
        height,
        maximized,
    })
}

/// Format bounds as the file's single line.
pub fn format_bounds(bounds: SavedBounds) -> String {
    format!(
        "{} {} {}\n",
        bounds.width,
        bounds.height,
        u8::from(bounds.maximized)
    )
}

/// File-backed bounds store.
pub struct FileBoundsStore {
    path: PathBuf,
}

impl FileBoundsStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl WindowBoundsStore for FileBoundsStore {
    fn load(&self) -> Option<SavedBounds> {
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|text| parse_bounds(&text))
    }

    fn save(&self, bounds: SavedBounds) {
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // Losing the saved window size is harmless; never interrupt the event loop for it.
        let _ = std::fs::write(&self.path, format_bounds(bounds));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_round_trip() {
        let b = SavedBounds {
            width: 1600,
            height: 900,
            maximized: true,
        };
        assert_eq!(parse_bounds(&format_bounds(b)), Some(b));
        assert_eq!(parse_bounds("garbage"), None);
        assert_eq!(parse_bounds(""), None);
    }
}
