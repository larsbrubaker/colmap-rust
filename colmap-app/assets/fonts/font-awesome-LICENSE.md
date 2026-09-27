# Font Awesome 4.7.0

`font-awesome.ttf` is the Font Awesome 4.7.0 icon font (family "FontAwesome", by Dave Gandy /
Fort Awesome). colmap-rust embeds it in `colmap-app` for the UI's icons: `colmap-app/src/fonts.rs`
installs it as a fallback face and the `fa` module there names each icon's Unicode Private Use
Area code point (e.g. the top bar's info button and the About sheet's Close button), which
widgets render as ordinary text.

## License

- **Font file** (`font-awesome.ttf`, including the glyph designs inside it): SIL Open Font
  License 1.1 — <https://opensource.org/licenses/OFL-1.1>
- Font Awesome's CSS/LESS/SASS code is MIT licensed; none of it is used or vendored here.

Font Awesome by Dave Gandy / Fort Awesome — <https://fontawesome.com/v4/license/>
