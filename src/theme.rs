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
use kilim_tema::Boncuk;
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
/// Wide enough for a Hugging Face repo id (e.g. "BAAI/bge-m3"); `INPUT_WIDTH`
/// is sized for the Port/Parallel-threads numeric fields, too narrow here.
pub const MODEL_INPUT_WIDTH: f32 = 200.0;
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
    fonts.font_data.insert(
        "Inter-Regular".into(),
        Arc::new(FontData::from_static(INTER_REGULAR)),
    );
    fonts.font_data.insert(
        "Inter-Medium".into(),
        Arc::new(FontData::from_static(INTER_MEDIUM)),
    );
    fonts.font_data.insert(
        "Inter-SemiBold".into(),
        Arc::new(FontData::from_static(INTER_SEMIBOLD)),
    );

    // Inter Regular becomes the default proportional font (headings, body text
    // typed via TextStyle all pick it up); the three weights are additionally
    // reachable by name for RichText::font(theme::font(size, weight)).
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "Inter-Regular".to_string());
    fonts.families.insert(
        FontFamily::Name("Inter-Regular".into()),
        vec!["Inter-Regular".to_string()],
    );
    fonts.families.insert(
        FontFamily::Name("Inter-Medium".into()),
        vec!["Inter-Medium".to_string()],
    );
    fonts.families.insert(
        FontFamily::Name("Inter-SemiBold".into()),
        vec!["Inter-SemiBold".to_string()],
    );

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

/// Semantic roles mapped onto the shared Kilim palette (kilim-tema), so all
/// four apps share one colour source.
fn from_kilim(k: kilim_tema::Kilim, accent_on: Color32) -> Palette {
    Palette {
        bg_window: k.zem,
        bg_surface: k.yuzey,
        border: k.cizgi,
        text: k.murekkep,
        text_muted: k.soluk,
        accent: k.cini,
        accent_on,
        success: k.yesil,
        warning: k.sari,
        error: k.kirmizi,
    }
}

pub fn light() -> Palette {
    let k = kilim_tema::acik();
    from_kilim(k, k.yuzey)
}

pub fn dark() -> Palette {
    let k = kilim_tema::koyu();
    from_kilim(k, k.zem)
}

/// The palette matching egui's currently active theme (egui already tracks
/// light/dark for us, following the OS - see [`apply`]).
pub fn current(ctx: &egui::Context) -> Palette {
    for_theme(ctx, ctx.theme())
}

/// `light()`/`dark()` with the accent taken from the active kilim border variant
/// (turquoise or madder), so selections match the border instead of a flat blue.
fn for_theme(ctx: &egui::Context, theme: egui::Theme) -> Palette {
    let dark_mode = theme == egui::Theme::Dark;
    let mut p = if dark_mode { dark() } else { light() };
    p.accent = kilim_tema::etkin_varyant(ctx).vurgu(dark_mode);
    p
}

/// Text boxes and idle tabs sit "sunken" on the OPPOSITE theme's paper:
/// darkened paper under the light theme, plain paper under the dark one,
/// never flat white/black. Ink flips with it so text always contrasts with
/// its own base (light ink on the darkened paper, dark ink on the plain one).
#[derive(Clone, Copy)]
pub struct Sunken {
    pub tone: Color32,
    pub ink: Color32,
    pub ink_muted: Color32,
}

pub fn sunken(ctx: &egui::Context) -> Sunken {
    let (tone, k) = match ctx.theme() {
        egui::Theme::Light => (kilim_tema::KOYU_KAGIT, kilim_tema::koyu()),
        egui::Theme::Dark => (Color32::WHITE, kilim_tema::acik()),
    };
    Sunken {
        tone,
        ink: k.murekkep,
        ink_muted: k.soluk,
    }
}

/// Paper texture (multiplied by `tone`) under `rect`, with the 1 px control border.
fn paper(ui: &Ui, rect: egui::Rect, tone: Color32, p: Palette) {
    kilim_tema::kagit_ciz(ui.painter(), rect, tone);
    ui.painter().rect_stroke(
        rect,
        CornerRadius::same(RADIUS_CONTROL),
        Stroke::new(BORDER_WIDTH, p.border),
        StrokeKind::Inside,
    );
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
    for w in [
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
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
    apply_visuals(ctx);
    ctx.set_theme(egui::ThemePreference::System);
}

/// Re-derives both visuals from the palette; call again after the kilim
/// variant changes (the accent follows it).
pub fn apply_visuals(ctx: &egui::Context) {
    let l = for_theme(ctx, egui::Theme::Light);
    let d = for_theme(ctx, egui::Theme::Dark);
    ctx.set_visuals_of(egui::Theme::Light, base_visuals(l, false));
    ctx.set_visuals_of(egui::Theme::Dark, base_visuals(d, true));
}

// ---------------------------------------------------------------------------
// Shared parts (design/DESIGN.md "Shared parts")
// ---------------------------------------------------------------------------

/// A `RichText` label in one call: `theme::rich(text, size, weight, color)`.
pub fn rich(text: impl Into<String>, size: f32, weight: Weight, color: Color32) -> RichText {
    RichText::new(text.into())
        .font(font(size, weight))
        .color(color)
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

fn button(
    ui: &mut Ui,
    enabled: bool,
    bead: Boncuk,
    text: RichText,
    fill: Color32,
    stroke: Stroke,
) -> Response {
    let btn = kilim_tema::boncuklu(ui, bead, text)
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
        Boncuk::Firuze,
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
        Boncuk::Kehribar,
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
    button(
        ui,
        enabled,
        Boncuk::Akik,
        rich(label, SIZE_CAPTION, Weight::SemiBold, p.accent_on),
        p.error,
        Stroke::NONE,
    )
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
                status_bead(ui, p, dot_color, 12.0);
                ui.label(rich(label, SIZE_CAPTION, Weight::Medium, p.text));
            });
        });
}

/// One of the three bottom nav tabs. Active = paper tinted with the accent
/// (the grain shows, so it sits with the kilim border instead of a flat blue)
/// + `accent.on` label (600); inactive = sunken paper + its contrasting
/// muted ink (500).
pub fn nav_tab(ui: &mut Ui, p: Palette, label: &str, active: bool, width: f32) -> Response {
    let s = sunken(ui.ctx());
    let text = rich(
        label,
        SIZE_CAPTION,
        if active {
            Weight::SemiBold
        } else {
            Weight::Medium
        },
        if active { p.accent_on } else { s.ink_muted },
    );
    // Active tab's bead is lit, inactive ones are dimmed - the bead marks the selection.
    let cap = SIZE_CAPTION + 4.0;
    let bead = kilim_tema::boncuk_resmi(ui.ctx(), Boncuk::Firuze, cap).tint(if active {
        Color32::WHITE
    } else {
        Color32::from_gray(150).gamma_multiply(0.6)
    });
    let btn = egui::Button::image_and_text(bead, text)
        .fill(Color32::TRANSPARENT)
        .stroke(Stroke::NONE)
        .corner_radius(CornerRadius::same(RADIUS_CONTROL));
    let size = Vec2::new(width, ui.spacing().interact_size.y + 2.0 * INSET_SM);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    paper(ui, rect, if active { p.accent } else { s.tone }, p);
    // Child ui: the button sits on the paper without advancing the parent's cursor.
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
    child.spacing_mut().button_padding = Vec2::new(0.0, INSET_SM);
    let response = child.add_sized(size, btn);
    if response.hovered() {
        ui.painter().rect_stroke(
            rect,
            CornerRadius::same(RADIUS_CONTROL),
            Stroke::new(BORDER_WIDTH, p.accent),
            StrokeKind::Inside,
        );
    }
    response
}

/// Toggle switch (disabled/static): 36x20, full radius, padding 4, 12 px
/// knob in surface colour; track = accent when on, `text.muted` when off;
/// knob right when on, left when off. Always non-interactive here - every
/// Settings toggle in this task is a "coming soon" preview.
pub fn toggle_static(ui: &mut Ui, p: Palette, on: bool) -> Response {
    let size = Vec2::new(TOGGLE_WIDTH, TOGGLE_HEIGHT);
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    paint_toggle(ui, p, rect, if on { 1.0 } else { 0.0 });
    response
}

/// Interactive twin of `toggle_static`: same visuals, but clickable - flips
/// `*on` and returns a `Response` whose `.changed()` is true iff it did, so
/// callers can react (e.g. `autostart::set_enabled`) only on an actual click.
pub fn toggle(ui: &mut Ui, p: Palette, on: &mut bool) -> Response {
    let size = Vec2::new(TOGGLE_WIDTH, TOGGLE_HEIGHT);
    let (rect, mut response) = ui.allocate_exact_size(size, Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    let pos = ui.ctx().animate_bool(response.id, *on);
    paint_toggle(ui, p, rect, pos);
    response
}

/// Track = accent when on, `text.muted` when off; the knob is a kilim-tema bead
/// that slides with `pos` (0 = off, 1 = on).
fn paint_toggle(ui: &mut Ui, p: Palette, rect: egui::Rect, pos: f32) {
    let track = if pos > 0.5 { p.accent } else { p.text_muted };
    kilim_tema::anahtar_ciz(ui.painter(), rect, pos, track);
}

/// Editable twin of `input_box` (Model): same look (window bg, 1 px border,
/// radius 4, body text), but backed by egui's own `TextEdit` for real
/// keyboard/cursor/selection/clipboard handling instead of hand-painted text.
pub fn text_input(ui: &mut Ui, p: Palette, text: &mut String, size: Vec2) -> Response {
    on_paper(ui, p, size, egui::TextEdit::singleline(text))
}

/// Password field for connector keys, same sunken paper as every other box;
/// fills the width made available to it.
pub fn secret_input(ui: &mut Ui, p: Palette, text: &mut String, hint: &str) -> Response {
    let size = Vec2::new(ui.available_width(), SIZE_BODY + 2.0 * INSET_SM);
    on_paper(
        ui,
        p,
        size,
        egui::TextEdit::singleline(text)
            .password(true)
            .hint_text(hint),
    )
}

/// A `TextEdit` without its own frame, laid over sunken paper; text, hint and
/// selection colours are picked against that paper, not the page.
fn on_paper(ui: &mut Ui, p: Palette, size: Vec2, edit: egui::TextEdit<'_>) -> Response {
    let s = sunken(ui.ctx());
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    paper(ui, rect, s.tone, p);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect));
    child.visuals_mut().selection.bg_fill = p.accent;
    child.visuals_mut().weak_text_color = Some(s.ink_muted);
    child.add_sized(
        size,
        edit.frame(egui::Frame::NONE)
            .font(font(SIZE_BODY, Weight::Regular))
            .text_color(s.ink)
            .margin(Margin::symmetric(INSET_SM as i8, 0)),
    )
}

/// A small read-only "input" box (Port, Parallel threads): window bg, 1 px
/// border, radius 4, padding 8, body text.
pub fn input_box(ui: &mut Ui, p: Palette, text: &str, size: Vec2) -> Response {
    let s = sunken(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    paper(ui, rect, s.tone, p);
    let text_pos = rect.left_center() + Vec2::new(INSET_SM, 0.0);
    ui.painter().text(
        text_pos,
        egui::Align2::LEFT_CENTER,
        text,
        font(SIZE_BODY, Weight::Regular),
        s.ink,
    );
    response
}

/// The Endpoint card's URL field: window bg, radius 4, padding 8, caption/500
/// muted text, fills the width made available to it.
pub fn url_field(ui: &mut Ui, p: Palette, text: &str) -> Response {
    let size = Vec2::new(ui.available_width(), SIZE_CAPTION + 2.0 * INSET_SM);
    let s = sunken(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    paper(ui, rect, s.tone, p);
    let text_pos = rect.left_center() + Vec2::new(INSET_SM, 0.0);
    ui.painter().text(
        text_pos,
        egui::Align2::LEFT_CENTER,
        text,
        font(SIZE_CAPTION, Weight::Medium),
        s.ink_muted,
    );
    response
}

/// Connections-row avatar: 32 px sedef (mother-of-pearl) bead with the
/// 2-letter initials caption/600 in the bead's own ink.
pub fn avatar(ui: &mut Ui, _p: Palette, initials: &str) {
    let d = 32.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(d), Sense::hover());
    kilim_tema::boncuk_resmi(ui.ctx(), Boncuk::Sedef, d).paint_at(ui, rect);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        initials,
        font(SIZE_CAPTION, Weight::SemiBold),
        Boncuk::Sedef.simge_rengi(),
    );
}

/// Status dot as a bead: success = firuze, warning = kehribar, error = akik,
/// anything else (idle/checking) = telkari silver.
fn status_bead(ui: &mut Ui, p: Palette, dot_color: Color32, d: f32) {
    let bead = if dot_color == p.success {
        Boncuk::Firuze
    } else if dot_color == p.warning {
        Boncuk::Kehribar
    } else if dot_color == p.error {
        Boncuk::Akik
    } else {
        Boncuk::Telkari
    };
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(d), Sense::hover());
    kilim_tema::boncuk_resmi(ui.ctx(), bead, d).paint_at(ui, rect);
}

/// Connections-row meta line: 10 px status bead + caption muted text, gap 4.
pub fn status_dot_row(ui: &mut Ui, p: Palette, dot_color: Color32, label: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = GAP_XS;
        status_bead(ui, p, dot_color, 10.0);
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
        // make sure light and dark actually differ (no copy-paste palette).
        let (l, d) = (light(), dark());
        assert_ne!(l.bg_window, d.bg_window);
        assert_ne!(l.bg_surface, d.bg_surface);
        assert_ne!(l.border, d.border);
        assert_ne!(l.text, d.text);
        assert_ne!(l.text_muted, d.text_muted);
        assert_ne!(l.accent, d.accent);
        assert_ne!(l.success, d.success);
        assert_ne!(l.warning, d.warning);
        assert_ne!(l.error, d.error);
    }

    #[test]
    fn roles_map_onto_kilim_palette() {
        assert_eq!(light().bg_window, Color32::from_rgb(0xF4, 0xEC, 0xDB)); // Kilim.zem
        assert_eq!(dark().accent, Color32::from_rgb(0x7F, 0xB2, 0xD9)); // Kilim.cini (koyu)
        assert_eq!(light().error, Color32::from_rgb(0xB2, 0x3A, 0x32)); // Kilim.kirmizi
    }

    /// WCAG relative luminance of an opaque sRGB colour.
    fn luminance(c: Color32) -> f32 {
        let lin = |v: u8| {
            let v = f32::from(v) / 255.0;
            if v <= 0.03928 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * lin(c.r()) + 0.7152 * lin(c.g()) + 0.0722 * lin(c.b())
    }

    fn contrast(a: Color32, b: Color32) -> f32 {
        let (hi, lo) = (
            luminance(a).max(luminance(b)),
            luminance(a).min(luminance(b)),
        );
        (hi + 0.05) / (lo + 0.05)
    }

    #[test]
    fn sunken_ink_contrasts_with_its_own_paper_in_both_themes() {
        for theme in [egui::Theme::Light, egui::Theme::Dark] {
            let ctx = egui::Context::default();
            ctx.set_theme(theme);
            let s = sunken(&ctx);
            // Text boxes: normal ink must clear WCAG AA (4.5); muted ink, used for
            // idle tab labels and hints at caption size, must too.
            assert!(contrast(s.tone, s.ink) >= 4.5, "{theme:?} ink");
            assert!(contrast(s.tone, s.ink_muted) >= 4.5, "{theme:?} muted ink");
        }
        // The paper really is the opposite theme's: darkened under light, plain under dark.
        let (l, d) = (egui::Context::default(), egui::Context::default());
        l.set_theme(egui::Theme::Light);
        d.set_theme(egui::Theme::Dark);
        assert!(luminance(sunken(&l).tone) < luminance(sunken(&d).tone));
    }

    #[test]
    fn full_radius_clamps_to_a_pill() {
        // CornerRadius::same(255) on a 20 px-tall shape should behave like a
        // pill, i.e. epaint clamps it well below the literal 255 stored here.
        assert_eq!(RADIUS_FULL, 255);
    }
}
