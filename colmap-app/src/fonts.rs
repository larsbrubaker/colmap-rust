// Theme, fonts and text-quality settings, installed once at startup by every shell (and by the
// headless harness) so native, web and tests render with the same setup. Font Awesome 4.7 is a
// fallback face on Noto Sans, so an icon is just its code point drawn as ordinary text.
// Mirrors AtomArtist's `shell_init.rs` / `fa.rs`. Font licenses: SIL OFL 1.1, see
// `assets/fonts/` and docs/LICENSE_AUDIT.md.

use std::sync::Arc;

use agg_gui::font_settings;
use agg_gui::text::Font;
use agg_gui::theme::{set_visuals, Visuals};

const UI_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/NotoSans-Regular.ttf");
const ICON_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/font-awesome.ttf");

/// Font Awesome 4.7 code points used by the app.
pub mod fa {
    /// fa-info-circle — opens the About panel.
    pub const INFO_CIRCLE: char = '\u{f05a}';
    /// fa-times — closes a panel.
    pub const TIMES: char = '\u{f00d}';
}

/// Install the dark theme, the UI font (with the icon fallback) and the text recipe.
///
/// `device_scale` is physical pixels per logical pixel. LCD subpixel text and hinting are only
/// enabled at standard DPI, where they help; above 1.25x they cause color fringes. Returns the
/// UI font for widgets that take one explicitly. Idempotent.
pub fn install_theme_and_fonts(device_scale: f64) -> Result<Arc<Font>, &'static str> {
    set_visuals(Visuals::dark());
    let icons = Arc::new(Font::from_bytes(ICON_FONT_BYTES.to_vec())?);
    let font = Arc::new(Font::from_bytes(UI_FONT_BYTES.to_vec())?.with_fallback(icons));
    font_settings::set_system_font(Some(Arc::clone(&font)));

    let standard_dpi = device_scale <= 1.25;
    font_settings::set_font_size_scale(1.0);
    font_settings::set_lcd_enabled(standard_dpi);
    font_settings::set_hinting_enabled(standard_dpi);
    font_settings::set_gamma(1.0);
    font_settings::set_width(1.0);
    font_settings::set_interval(0.0);
    font_settings::set_faux_weight(0.0);
    font_settings::set_faux_italic(0.0);
    font_settings::set_primary_weight(1.0 / 3.0);
    Ok(font)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_fonts_load_and_cover_the_icons() {
        let font = install_theme_and_fonts(1.0).expect("bundled fonts parse");
        let icons = Font::from_bytes(ICON_FONT_BYTES.to_vec()).expect("icon font parses");
        for glyph in [fa::INFO_CIRCLE, fa::TIMES] {
            assert!(
                icons.glyph_visual_bounds(glyph, 16.0).is_some(),
                "missing icon glyph U+{:04X}",
                glyph as u32
            );
        }
        assert!(font_settings::lcd_enabled());
        install_theme_and_fonts(2.0).expect("reinstall");
        assert!(!font_settings::lcd_enabled());
        drop(font);
    }
}
