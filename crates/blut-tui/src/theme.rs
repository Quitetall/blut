// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Theme — BLUT cockpit colors and reusable styles.
//!
//! A small, self-contained palette of named style getters (titles,
//! headings, success/error/warning, key hints, status bar) so the views
//! render consistently without each drawer hand-rolling colors.
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
#[cfg(not(test))]
static COLOR_DISABLED: AtomicBool = AtomicBool::new(false);
/// 0 = unicode, 1 = ascii.
#[cfg(not(test))]
static CHARSET: AtomicU8 = AtomicU8::new(0);

// Under test, each test thread gets its own theme state. The state is
// process-global in the binary, where one `detect` runs at startup; in the
// test harness dozens of tests call `detect` concurrently, and with shared
// atomics any of them could flip another's colors between its `detect` and
// its assertion — `detect_no_color_disables` and
// `not_selected_uses_neutral_status_style` both failed intermittently in CI.
// libtest runs every test on its own thread, so thread-local state makes each
// test see exactly what it set.
#[cfg(test)]
thread_local! {
    static COLOR_DISABLED: AtomicBool = const { AtomicBool::new(false) };
    static CHARSET: AtomicU8 = const { AtomicU8::new(0) };
}

#[cfg(not(test))]
fn store_color_disabled(disabled: bool) {
    COLOR_DISABLED.store(disabled, Ordering::Relaxed);
}
#[cfg(test)]
fn store_color_disabled(disabled: bool) {
    COLOR_DISABLED.with(|c| c.store(disabled, Ordering::Relaxed));
}

#[cfg(not(test))]
fn load_color_disabled() -> bool {
    COLOR_DISABLED.load(Ordering::Relaxed)
}
#[cfg(test)]
fn load_color_disabled() -> bool {
    COLOR_DISABLED.with(|c| c.load(Ordering::Relaxed))
}

#[cfg(not(test))]
fn store_charset(charset: u8) {
    CHARSET.store(charset, Ordering::Relaxed);
}
#[cfg(test)]
fn store_charset(charset: u8) {
    CHARSET.with(|c| c.store(charset, Ordering::Relaxed));
}

#[cfg(not(test))]
fn load_charset() -> u8 {
    CHARSET.load(Ordering::Relaxed)
}
#[cfg(test)]
fn load_charset() -> u8 {
    CHARSET.with(|c| c.load(Ordering::Relaxed))
}

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
    store_color_disabled(color_off);

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
    store_charset(if charset_ascii { 1 } else { 0 });
}

/// Serializes tests around the environment variables `detect` reads.
///
/// Theme state is per-thread under test (see above), but the environment is
/// not: the theme tests set `NO_COLOR`, `TERM` and `LANG`, and every `detect`
/// call reads them. Reading the environment while another thread writes it
/// is exactly what Rust 2024's `unsafe` on `set_var` warns about. Every test
/// that calls `detect` holds this lock around the call.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[inline]
fn color_enabled() -> bool {
    !load_color_disabled()
}

/// True when the terminal cannot reliably render Unicode (e.g.
/// `TERM=dumb` or a non-UTF-8 locale).
pub fn ascii_only() -> bool {
    load_charset() == 1
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

// ── BLUT console palette (the engine-console identity) ────────────────────
// Truecolor named tokens; `maybe()` still zeroes them under NO_COLOR/dumb, so
// the accessibility discipline is preserved. Teal = verification / interactive
// (content-addressed integrity), green = cache-hit / healthy, amber = warning,
// red = the fail-closed hard-block.
pub const SIGNAL: Color = Color::Rgb(0x37, 0xC2, 0xB0);
/// Emphasis teal — reserved for the phase-2 drill-down tab bar.
#[allow(dead_code)]
pub const SIGNAL_BRIGHT: Color = Color::Rgb(0x5F, 0xE0, 0xCE);
pub const VERIFIED: Color = Color::Rgb(0x6B, 0xD0, 0x8A);
pub const AMBER: Color = Color::Rgb(0xE8, 0xA2, 0x3C);
pub const HALT: Color = Color::Rgb(0xE5, 0x54, 0x4B);
pub const FOG: Color = Color::Rgb(0x7B, 0x87, 0x94);
pub const LINE: Color = Color::Rgb(0x2A, 0x33, 0x3E);
pub const BRIGHT_C: Color = Color::Rgb(0xE7, 0xEC, 0xF1);

/// Uppercase, dimmed section labels ("MESH", "BROKER", …).
pub fn label() -> Style {
    maybe(Style::default().fg(FOG))
}
/// The accent — active dispatch, interactive, verified integrity.
pub fn signal() -> Style {
    maybe(Style::default().fg(SIGNAL))
}
pub fn signal_bold() -> Style {
    maybe(Style::default().fg(SIGNAL).add_modifier(Modifier::BOLD))
}
/// A big key metric that matters.
pub fn metric() -> Style {
    maybe(Style::default().fg(BRIGHT_C).add_modifier(Modifier::BOLD))
}
/// Cache-hit / healthy / verified state.
pub fn verified() -> Style {
    maybe(Style::default().fg(VERIFIED))
}
/// Warning / degraded state.
pub fn amber() -> Style {
    maybe(Style::default().fg(AMBER))
}
/// Blocked / failed / the clinical hard-block red line.
pub fn halt() -> Style {
    maybe(Style::default().fg(HALT).add_modifier(Modifier::BOLD))
}
/// Hairline panel borders.
pub fn panel_border() -> Style {
    maybe(Style::default().fg(LINE))
}
/// Active tab in the console tab bar (phase-2 drill-down navigation).
#[allow(dead_code)]
pub fn tab_active() -> Style {
    maybe(
        Style::default()
            .fg(SIGNAL_BRIGHT)
            .add_modifier(Modifier::BOLD),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rust 2024 makes `set_var`/`remove_var` unsafe (they can race other
    // threads reading the environment). All callers below hold `test_lock`
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
        let _g = test_lock();
        unset_env("LANG");
        set_env("NO_COLOR", "1");
        detect("auto", "auto");
        assert!(!color_enabled());
        assert_eq!(title(), Style::default(), "color off → no ANSI style");
        unset_env("NO_COLOR");
    }

    #[test]
    fn explicit_always_overrides_no_color() {
        let _g = test_lock();
        set_env("NO_COLOR", "1");
        detect("always", "auto");
        assert!(color_enabled());
        unset_env("NO_COLOR");
    }

    #[test]
    fn explicit_never_disables_even_without_no_color() {
        let _g = test_lock();
        unset_env("NO_COLOR");
        detect("never", "auto");
        assert!(!color_enabled());
    }

    #[test]
    fn term_dumb_forces_ascii() {
        let _g = test_lock();
        set_env("TERM", "dumb");
        detect("auto", "auto");
        assert!(ascii_only());
        unset_env("TERM");
    }

    #[test]
    fn explicit_ascii_charset() {
        let _g = test_lock();
        detect("auto", "ascii");
        assert!(ascii_only());
    }

    #[test]
    fn explicit_unicode_charset_even_on_dumb_term() {
        let _g = test_lock();
        set_env("TERM", "dumb");
        detect("auto", "unicode");
        assert!(!ascii_only());
        // Restore default so later cases in this file see unicode.
        unset_env("TERM");
        detect("auto", "auto");
    }
}
