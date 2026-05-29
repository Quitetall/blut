//! Theme — BLUT cockpit colors and reusable styles.
//!
//! Replicates the lamquant-lossless TUI theme convention
//! (`lamquant-lossless/src/tui/theme.rs`) so the BLUT cockpit matches
//! the rest of the project visually. BLUT is a standalone crate and
//! cannot depend on `lamquant-lossless`, so the convention is mirrored
//! here verbatim rather than imported.
//!
//! Honours `NO_COLOR`, `TERM=dumb`, and an explicit color/charset
//! preference. When color is disabled every style getter returns
//! `Style::default()` so the renderer emits no ANSI escapes; when the
//! charset is `ascii` the box/divider glyphs fall back to ASCII.

use ratatui::style::{Color, Modifier, Style};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

pub const CYAN: Color = Color::Cyan;
pub const GREEN: Color = Color::Green;
pub const RED: Color = Color::Red;
pub const YELLOW: Color = Color::Yellow;
pub const DIM: Color = Color::DarkGray;
pub const WHITE: Color = Color::White;

// ── Runtime detection ────────────────────────────────────────────────────
// Detected once at App::new() (or first style call). Atomics so the draw
// functions can read without a Mutex.

/// 0 = enabled, 1 = disabled.
static COLOR_DISABLED: AtomicBool = AtomicBool::new(false);
/// 0 = unicode, 1 = ascii.
static CHARSET: AtomicU8 = AtomicU8::new(0);

/// Re-detect from environment (and optionally an explicit cfg pref).
/// Should be called once at startup. `cfg_color`: "auto" | "always" |
/// "never". `cfg_charset`: "auto" | "unicode" | "ascii".
pub fn detect(cfg_color: &str, cfg_charset: &str) {
    let no_color = std::env::var_os("NO_COLOR").is_some();
    let term_dumb = std::env::var("TERM").map(|t| t == "dumb").unwrap_or(false);
    let color_off = match cfg_color {
        "never" => true,
        "always" => false,
        _ => no_color || term_dumb,
    };
    COLOR_DISABLED.store(color_off, Ordering::Relaxed);

    let lang = std::env::var("LANG").unwrap_or_default();
    let lc_ctype = std::env::var("LC_CTYPE").unwrap_or_default();
    let utf8_locale = lang.to_uppercase().contains("UTF-8")
        || lang.to_uppercase().contains("UTF8")
        || lc_ctype.to_uppercase().contains("UTF-8");
    let charset_ascii = match cfg_charset {
        "ascii" => true,
        "unicode" => false,
        _ => term_dumb || !utf8_locale,
    };
    CHARSET.store(if charset_ascii { 1 } else { 0 }, Ordering::Relaxed);
}

#[inline]
fn color_enabled() -> bool {
    !COLOR_DISABLED.load(Ordering::Relaxed)
}

/// True when the terminal cannot reliably render Unicode (e.g.
/// `TERM=dumb` or a non-UTF-8 locale).
pub fn ascii_only() -> bool {
    CHARSET.load(Ordering::Relaxed) == 1
}

/// Color-disable passthrough: returns `s` when color is enabled,
/// `Style::default()` (no ANSI) otherwise.
#[inline]
fn maybe(s: Style) -> Style {
    if color_enabled() { s } else { Style::default() }
}

pub fn title() -> Style {
    maybe(Style::default().fg(CYAN).add_modifier(Modifier::BOLD))
}
pub fn heading() -> Style {
    maybe(Style::default().fg(WHITE).add_modifier(Modifier::BOLD))
}
pub fn normal() -> Style {
    maybe(Style::default().fg(WHITE))
}
pub fn dim() -> Style {
    maybe(Style::default().fg(DIM))
}
pub fn highlight() -> Style {
    maybe(Style::default().fg(CYAN).add_modifier(Modifier::BOLD))
}
pub fn selected() -> Style {
    // In monochrome mode, selection is signalled with REVERSED.
    if color_enabled() {
        Style::default()
            .bg(Color::DarkGray)
            .fg(WHITE)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().add_modifier(Modifier::REVERSED)
    }
}
pub fn success() -> Style {
    maybe(Style::default().fg(GREEN))
}
pub fn error() -> Style {
    maybe(Style::default().fg(RED))
}
pub fn warning() -> Style {
    maybe(Style::default().fg(YELLOW))
}
pub fn key_hint() -> Style {
    maybe(Style::default().fg(CYAN))
}
pub fn key_label() -> Style {
    maybe(Style::default().fg(DIM))
}
pub fn status_bar() -> Style {
    maybe(Style::default().bg(Color::DarkGray).fg(WHITE))
}
pub fn status_msg() -> Style {
    maybe(Style::default().bg(Color::DarkGray).fg(GREEN))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    /// Process-wide lock for theme tests. `cargo test` runs cases in
    /// parallel; without this lock two cases that mutate the same env
    /// vars + global atomics race and observe each other's state.
    fn env_lock() -> &'static Mutex<()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
    }

    // Rust 2024 makes `set_var`/`remove_var` unsafe (they can race other
    // threads reading the environment). All callers below hold `env_lock`
    // for their whole body, so within a single test there is no concurrent
    // env reader — the `unsafe` precondition is satisfied by the lock.
    fn set_env(key: &str, val: &str) {
        unsafe { std::env::set_var(key, val) }
    }
    fn unset_env(key: &str) {
        unsafe { std::env::remove_var(key) }
    }

    #[test]
    fn detect_no_color_disables() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        unset_env("LANG");
        set_env("NO_COLOR", "1");
        detect("auto", "auto");
        assert!(!color_enabled());
        assert_eq!(title(), Style::default(), "color off → no ANSI style");
        unset_env("NO_COLOR");
    }

    #[test]
    fn explicit_always_overrides_no_color() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        set_env("NO_COLOR", "1");
        detect("always", "auto");
        assert!(color_enabled());
        unset_env("NO_COLOR");
    }

    #[test]
    fn explicit_never_disables_even_without_no_color() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        unset_env("NO_COLOR");
        detect("never", "auto");
        assert!(!color_enabled());
    }

    #[test]
    fn term_dumb_forces_ascii() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        set_env("TERM", "dumb");
        detect("auto", "auto");
        assert!(ascii_only());
        unset_env("TERM");
    }

    #[test]
    fn explicit_ascii_charset() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        detect("auto", "ascii");
        assert!(ascii_only());
    }

    #[test]
    fn explicit_unicode_charset_even_on_dumb_term() {
        let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        set_env("TERM", "dumb");
        detect("auto", "unicode");
        assert!(!ascii_only());
        // Restore default so later cases in this file see unicode.
        unset_env("TERM");
        detect("auto", "auto");
    }
}
