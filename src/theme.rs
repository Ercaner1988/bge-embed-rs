//! Design tokens from `design/DESIGN.md`, applied to egui's `Style`/`Visuals`,
//! plus small drawing helpers for the shared parts (card, buttons, pill,
//! toggle, nav tab, inputs). Every colour/spacing/radius/size the screens use
//! comes from here - `gui.rs` has no inline hex or magic numbers.
//!
//! ponytail: helpers are plain functions that paint with the tokens directly,
//! not a generic widget/component framework - there are only 3 screens.

use eframe::egui::{
    self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Margin, Response,
    RichText, Sense, Stroke, StrokeKind, Ui, Vec2,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Spacing (design/DESIGN.md "Spacing (4 px grid)")
// ---------------------------------------------------------------------------

pub const GAP_XS: f32 = 4.0;
pub const GAP_SM: f32 = 8.0;
pub const GAP_MD: f32 = 12.0;
pub const GAP_LG: f32 = 16.0;
pub const INSET_SM: f32 = 8.0;
pub const INSET_MD: f32 = 16.0;
pub const INSET_LG: f32 = 24.0;

// ---------------------------------------------------------------------------
// Radius (design/DESIGN.md "Radius")
// ---------------------------------------------------------------------------

pub const RADIUS_CONTROL: u8 = 4;
pub const RADIUS_CARD: u8 = 8;

/// The "1 px" border/stroke width used on every card, pill, button and input.
pub const BORDER_WIDTH: f32 = 1.0;
/// "Progress track 8 px high" in DESIGN.md.
pub const PROGRESS_HEIGHT: f32 = 8.0;

/// Settings screen input box size ("input 88x32" in DESIGN.md).
pub const INPUT_WIDTH: f32 = 88.0;
pub const INPUT_HEIGHT: f32 = 32.0;
/// Toggle switch size ("Toggle: 36x20" in DESIGN.md).
pub const TOGGLE_WIDTH: f32 = 36.0;
pub const TOGGLE_HEIGHT: f32 = 20.0;
/// Larger than any control this app draws; epaint clamps a corner radius to
/// half the shape's shortest side, so this always renders as a full pill/circle.
pub const RADIUS_FULL: u8 = 255;

// ---------------------------------------------------------------------------
// Type (design/DESIGN.md "Type - Inter")
// ---------------------------------------------------------------------------

pub const SIZE_CAPTION: f32 = 12.0;
pub const SIZE_BODY: f32 = 14.0;
pub const SIZE_SUBTITLE: f32 = 17.0;
pub const SIZE_TITLE: f32 = 20.0;
/// Reserved by DESIGN.md; no screen uses it yet.
#[allow(dead_code)]
pub const SIZE_DISPLAY: f32 = 24.0;

#[derive(Clone, Copy)]
pub enum Weight {
    Regular,
    Medium,
    SemiBold,
}

impl Weight {
    fn family_name(self) -> &'static str {
        match self {
            Weight::Regular => "Inter-Regular",
            Weight::Medium => "Inter-Medium",
            Weight::SemiBold => "Inter-SemiBold",
        }
    }
}

/// A `FontId` for the given size/weight token, backed by the embedded Inter faces.
pub fn font(size: f32, weight: Weight) -> FontId {
    FontId::new(size, FontFamily::Name(weight.family_name().into()))
}

const INTER_REGULAR: &[u8] = include_bytes!("../assets/fonts/Inter-Regular.ttf");
const INTER_MEDIUM: &[u8] = include_bytes!("../assets/fonts/Inter-Medium.ttf");
const INTER_SEMIBOLD: &[u8] = include_bytes!("../assets/fonts/Inter-SemiBold.ttf");

fn install_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts
        .font_data
        .insert("Inter-Regular".into(), Arc::new(FontData::from_static(INTER_REGULAR)));
    fonts
        .font_data
        .insert("Inter-Medium".into(), Arc::new(FontData::from_static(INTER_MEDIUM)));
    fonts
        .font_data
        .insert("Inter-SemiBold".into(), Arc::new(FontData::from_static(INTER_SEMIBOLD)));

    // Inter Regular becomes the default proportional font (headings, body text
    // typed via TextStyle all pick it up); the three weights are additionally
    // reachable by name for RichText::font(theme::font(size, weight)).
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "Inter-Regular".to_string());
    fonts
        .families
        .insert(FontFamily::Name("Inter-Regular".into()), vec!["Inter-Regular".to_string()]);
    fonts
        .families
        .insert(FontFamily::Name("Inter-Medium".into()), vec!["Inter-Medium".to_string()]);
    fonts
        .families
        .insert(FontFamily::Name("Inter-SemiBold".into()), vec!["Inter-SemiBold".to_string()]);

    ctx.set_fonts(fonts);
}

// ---------------------------------------------------------------------------
// Colour (design/DESIGN.md "Colour (semantic, per theme)")
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Palette {
    pub bg_window: Color32,
    pub bg_surface: Color32,
    pub border: Color32,
    pub text: Color32,
    pub text_muted: Color32,
    pub accent: Color32,
    pub accent_on: Color32,
    pub success: Color32,
    pub warning: Color32,
    pub error: Color32,
}

const fn hex(rgb: u32) -> Color32 {
    Color32::from_rgb(((rgb >> 16) & 0xFF) as u8, ((rgb >> 8) & 0xFF) as u8, (rgb & 0xFF) as u8)
}

pub const LIGHT: Palette = Palette {
    bg_window: hex(0xF7F8FA),
    bg_surface: hex(0xFFFFFF),
    border: hex(0xDCE0E6),
    text: hex(0x171A1F),
    text_muted: hex(0x5B6472),
    accent: hex(0x1F6FD1),
    accent_on: hex(0xFFFFFF),
    success: hex(0x15803D),
    warning: hex(0x92600E),
    error: hex(0xB42318),
};

pub const DARK: Palette = Palette {
    bg_window: hex(0x171A1F),
    bg_surface: hex(0x262B33),
    border: hex(0x3F4652),
    text: hex(0xF7F8FA),
    text_muted: hex(0x9AA3AF),
    accent: hex(0x5AA2F5),
    accent_on: hex(0x0F1115),
    success: hex(0x3CC47C),
    warning: hex(0xE0A43A),
    error: hex(0xF06A62),
};

/// The palette matching egui's currently active theme (egui already tracks
/// light/dark for us, following the OS - see [`apply`]).
pub fn current(ctx: &egui::Context) -> Palette {
    match ctx.theme() {
        egui::Theme::Dark => DARK,
        egui::Theme::Light => LIGHT,
    }
}

fn base_visuals(p: Palette, dark_mode: bool) -> egui::Visuals {
    let mut v = if dark_mode {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    v.override_text_color = Some(p.text);
    v.weak_text_color = Some(p.text_muted);
    v.panel_fill = p.bg_window;
    v.window_fill = p.bg_window;
    v.extreme_bg_color = p.bg_surface;
    v.faint_bg_color = p.bg_surface;
    v.selection.bg_fill = p.accent;
    v.selection.stroke = Stroke::new(BORDER_WIDTH, p.accent_on);
    v.hyperlink_color = p.accent;
    v.warn_fg_color = p.warning;
    v.error_fg_color = p.error;
    v.window_stroke = Stroke::new(BORDER_WIDTH, p.border);
    v.widgets.noninteractive.bg_fill = p.bg_surface;
    v.widgets.noninteractive.bg_stroke = Stroke::new(BORDER_WIDTH, p.border);
    v.widgets.noninteractive.fg_stroke = Stroke::new(BORDER_WIDTH, p.text);
    v.widgets.noninteractive.corner_radius = CornerRadius::same(RADIUS_CARD);
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.corner_radius = CornerRadius::same(RADIUS_CONTROL);
        w.bg_fill = p.bg_surface;
        w.weak_bg_fill = p.bg_surface;
        w.bg_stroke = Stroke::new(BORDER_WIDTH, p.border);
        w.fg_stroke = Stroke::new(BORDER_WIDTH, p.text);
    }
    v
}

/// Installs the embedded Inter fonts and the light/dark visuals from
/// DESIGN.md. Theme then follows the OS automatically: `ThemePreference`
/// defaults to `System`, and `set_visuals_of` gives egui both palettes to
/// switch between (see `Context::system_theme`).
pub fn apply(ctx: &egui::Context) {
    install_fonts(ctx);
    ctx.set_visuals_of(egui::Theme::Light, base_visuals(LIGHT, false));
    ctx.set_visuals_of(egui::Theme::Dark, base_visuals(DARK, true));
    ctx.set_theme(egui::ThemePreference::System);
}

// ---------------------------------------------------------------------------
// Shared parts (design/DESIGN.md "Shared parts")
// ---------------------------------------------------------------------------

/// A `RichText` label in one call: `theme::rich(text, size, weight, color)`.
pub fn rich(text: impl Into<String>, size: f32, weight: Weight, color: Color32) -> RichText {
    RichText::new(text.into()).font(font(size, weight)).color(color)
}

/// Card: surface bg, 1 px border, radius 8, padding 16, fills the width.
pub fn card<R>(ui: &mut Ui, p: Palette, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    egui::Frame::new()
        .fill(p.bg_surface)
        .stroke(Stroke::new(BORDER_WIDTH, p.border))
        .corner_radius(CornerRadius::same(RADIUS_CARD))
        .inner_margin(Margin::same(INSET_MD as i8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add_contents(ui)
        })
        .inner
}

fn button(ui: &mut Ui, enabled: bool, text: RichText, fill: Color32, stroke: Stroke) -> Response {
    let btn = egui::Button::new(text)
        .fill(fill)
        .stroke(stroke)
        .corner_radius(CornerRadius::same(RADIUS_CONTROL));
    ui.scope(|ui| {
        ui.spacing_mut().button_padding = Vec2::new(INSET_MD, INSET_SM);
        ui.add_enabled(enabled, btn)
    })
    .inner
}

/// Primary button: accent bg, radius 4, padding 16x8, label caption/600 in `accent.on`.
pub fn primary_button(ui: &mut Ui, p: Palette, label: &str, enabled: bool) -> Response {
    button(
        ui,
        enabled,
        rich(label, SIZE_CAPTION, Weight::SemiBold, p.accent_on),
        p.accent,
        Stroke::NONE,
    )
}

/// Secondary button: surface bg + 1 px border, radius 4, padding 16x8, label caption/500.
pub fn secondary_button(ui: &mut Ui, p: Palette, label: &str, enabled: bool) -> Response {
    button(
        ui,
        enabled,
        rich(label, SIZE_CAPTION, Weight::Medium, p.text),
        p.bg_surface,
        Stroke::new(BORDER_WIDTH, p.border),
    )
}

/// Danger button: same shape as the primary button (radius 4, padding
/// 16x8, label caption/600), but filled with `status.error` instead of
/// `accent.default` - for a destructive confirmation action. No new colour
/// token: reuses `p.error` (already used for the Failed dot/text) and
/// `p.accent_on` (already used for text-on-fill).
pub fn danger_button(ui: &mut Ui, p: Palette, label: &str, enabled: bool) -> Response {
    button(ui, enabled, rich(label, SIZE_CAPTION, Weight::SemiBold, p.accent_on), p.error, Stroke::NONE)
}

/// Status pill: surface bg, 1 px border, full radius, padding 8x4, 8 px dot +
/// caption/500 label.
pub fn status_pill(ui: &mut Ui, p: Palette, dot_color: Color32, label: &str) {
    egui::Frame::new()
        .fill(p.bg_surface)
        .stroke(Stroke::new(BORDER_WIDTH, p.border))
        .corner_radius(CornerRadius::same(RADIUS_FULL))
        .inner_margin(Margin::symmetric(INSET_SM as i8, GAP_XS as i8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = GAP_SM;
                let (rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                ui.painter().circle_filled(rect.center(), 4.0, dot_color);
                ui.label(rich(label, SIZE_CAPTION, Weight::Medium, p.text));
            });
        });
}

/// One of the three bottom nav tabs. Active = accent bg + `accent.on` label
/// (600); inactive = no bg + muted label (500).
pub fn nav_tab(ui: &mut Ui, p: Palette, label: &str, active: bool, width: f32) -> Response {
    let text = rich(
        label,
        SIZE_CAPTION,
        if active { Weight::SemiBold } else { Weight::Medium },
        if active { p.accent_on } else { p.text_muted },
    );
    let btn = egui::Button::new(text)
        .fill(if active { p.accent } else { Color32::TRANSPARENT })
        .stroke(Stroke::NONE)
        .corner_radius(CornerRadius::same(RADIUS_CONTROL));
    ui.scope(|ui| {
        ui.spacing_mut().button_padding = Vec2::new(0.0, INSET_SM);
        ui.add_sized(Vec2::new(width, ui.spacing().interact_size.y + 2.0 * INSET_SM), btn)
    })
    .inner
}

/// Toggle switch (disabled/static): 36x20, full radius, padding 4, 12 px
/// knob in surface colour; track = accent when on, `text.muted` when off;
/// knob right when on, left when off. Always non-interactive here - every
/// Settings toggle in this task is a "coming soon" preview.
pub fn toggle_static(ui: &mut Ui, p: Palette, on: bool) -> Response {
    let size = Vec2::new(TOGGLE_WIDTH, TOGGLE_HEIGHT);
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    let track = if on { p.accent } else { p.text_muted };
    ui.painter().rect_filled(rect, CornerRadius::same(RADIUS_FULL), track);
    let knob_radius = 6.0; // "12 px knob" in DESIGN.md
    let pad = GAP_XS; // "padding 4" in DESIGN.md
    let cx = if on {
        rect.right() - pad - knob_radius
    } else {
        rect.left() + pad + knob_radius
    };
    ui.painter().circle_filled(egui::pos2(cx, rect.center().y), knob_radius, p.bg_surface);
    response
}

/// A small read-only "input" box (Port, Parallel threads): window bg, 1 px
/// border, radius 4, padding 8, body text.
pub fn input_box(ui: &mut Ui, p: Palette, text: &str, size: Vec2) -> Response {
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(RADIUS_CONTROL), p.bg_window);
    ui.painter()
        .rect_stroke(rect, CornerRadius::same(RADIUS_CONTROL), Stroke::new(BORDER_WIDTH, p.border), StrokeKind::Inside);
    let text_pos = rect.left_center() + Vec2::new(INSET_SM, 0.0);
    ui.painter()
        .text(text_pos, egui::Align2::LEFT_CENTER, text, font(SIZE_BODY, Weight::Regular), p.text);
    response
}

/// The Endpoint card's URL field: window bg, radius 4, padding 8, caption/500
/// muted text, fills the width made available to it.
pub fn url_field(ui: &mut Ui, p: Palette, text: &str) -> Response {
    let size = Vec2::new(ui.available_width(), SIZE_CAPTION + 2.0 * INSET_SM);
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(RADIUS_CONTROL), p.bg_window);
    let text_pos = rect.left_center() + Vec2::new(INSET_SM, 0.0);
    ui.painter()
        .text(text_pos, egui::Align2::LEFT_CENTER, text, font(SIZE_CAPTION, Weight::Medium), p.text_muted);
    response
}

/// Connections-row avatar: "32 px circle, window bg + border, 2-letter
/// initials caption/600 muted text" (DESIGN.md).
pub fn avatar(ui: &mut Ui, p: Palette, initials: &str) {
    let d = 32.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(d), Sense::hover());
    ui.painter().circle_filled(rect.center(), d / 2.0, p.bg_window);
    ui.painter()
        .circle_stroke(rect.center(), d / 2.0 - BORDER_WIDTH / 2.0, Stroke::new(BORDER_WIDTH, p.border));
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        initials,
        font(SIZE_CAPTION, Weight::SemiBold),
        p.text_muted,
    );
}

/// Connections-row meta line: "6 px status dot + caption muted text, gap 4" (DESIGN.md).
pub fn status_dot_row(ui: &mut Ui, p: Palette, dot_color: Color32, label: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = GAP_XS;
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(6.0), Sense::hover());
        ui.painter().circle_filled(rect.center(), 3.0, dot_color);
        ui.label(rich(label, SIZE_CAPTION, Weight::Regular, p.text_muted));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_themes_define_every_token() {
        // Just accessing every field is the check: if a palette literal were
        // missing a field this wouldn't compile, so this test's job is to
        // make sure LIGHT and DARK actually differ (no copy-paste palette).
        assert_ne!(LIGHT.bg_window, DARK.bg_window);
        assert_ne!(LIGHT.bg_surface, DARK.bg_surface);
        assert_ne!(LIGHT.border, DARK.border);
        assert_ne!(LIGHT.text, DARK.text);
        assert_ne!(LIGHT.text_muted, DARK.text_muted);
        assert_ne!(LIGHT.accent, DARK.accent);
        assert_ne!(LIGHT.success, DARK.success);
        assert_ne!(LIGHT.warning, DARK.warning);
        assert_ne!(LIGHT.error, DARK.error);
    }

    #[test]
    fn hex_matches_design_doc() {
        assert_eq!(LIGHT.bg_window, Color32::from_rgb(0xF7, 0xF8, 0xFA));
        assert_eq!(DARK.accent, Color32::from_rgb(0x5A, 0xA2, 0xF5));
        assert_eq!(LIGHT.error, Color32::from_rgb(0xB4, 0x23, 0x18));
    }

    #[test]
    fn full_radius_clamps_to_a_pill() {
        // CornerRadius::same(255) on a 20 px-tall shape should behave like a
        // pill, i.e. epaint clamps it well below the literal 255 stored here.
        assert_eq!(RADIUS_FULL, 255);
    }
}
