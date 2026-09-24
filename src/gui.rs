//! The three-screen egui GUI from `design/DESIGN.md`: Status, Connections,
//! Settings, switched by a bottom nav bar. All styling comes from
//! `theme.rs` - no inline hex/spacing/radius/size literals here.

use crate::{effective_host, effective_parallel, effective_port, Phase, Status};
use crate::theme::{self, Palette, Weight};
use eframe::egui::{self, Align, Color32, Layout, Ui};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Status,
    Connections,
    Settings,
}

pub struct GuiApp {
    status: Arc<Status>,
    screen: Screen,
}

impl GuiApp {
    pub fn new(status: Arc<Status>) -> Self {
        Self { status, screen: Screen::Status }
    }
}

impl eframe::App for GuiApp {
    // eframe 0.36 renamed App::update(ctx, frame) to App::ui(ui, frame); the
    // root Ui is handed in directly instead of the Context (CentralPanel::show
    // now takes &mut Ui too - see egui 0.36 containers/panel.rs).
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // The server thread updates status independently of egui's event loop,
        // so keep repainting on a timer rather than only on user input.
        ctx.request_repaint_after(Duration::from_millis(250));
        let p = theme::current(&ctx);

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(p.bg_window).inner_margin(theme::INSET_LG as i8))
            .show(ui, |ui| {
                let phase = self.status.phase.lock().unwrap().clone();

                header_row(ui, p, phase.as_ref());
                ui.add_space(theme::GAP_LG);

                match self.screen {
                    Screen::Status => status_screen(ui, p, &self.status, phase.as_ref()),
                    Screen::Connections => connections_screen(ui, p),
                    Screen::Settings => settings_screen(ui, p),
                }

                // Flexible spacer pushes the nav bar to the bottom.
                ui.with_layout(Layout::bottom_up(Align::Center), |ui| {
                    nav_bar(ui, p, &mut self.screen);
                });
            });
    }
}

/// Header row: title/subtitle on the left, status pill on the right.
fn header_row(ui: &mut Ui, p: Palette, phase: Option<&Phase>) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = theme::GAP_XS;
            ui.label(theme::rich("bge-embed-rs", theme::SIZE_TITLE, Weight::SemiBold, p.text));
            ui.label(theme::rich(
                "Local bge-m3 embedding server",
                theme::SIZE_CAPTION,
                Weight::Regular,
                p.text_muted,
            ));
        });
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let (color, label) = phase_pill(phase, &p);
            theme::status_pill(ui, p, color, label);
        });
    });
}

/// Phase -> (dot colour, pill label), per DESIGN.md's "Shared parts" table.
fn phase_pill(phase: Option<&Phase>, p: &Palette) -> (Color32, &'static str) {
    match phase {
        None => (p.warning, "Downloading"),
        Some(Phase::Downloading { .. }) => (p.warning, "Downloading"),
        Some(Phase::Loading) => (p.warning, "Loading"),
        Some(Phase::Ready { .. }) => (p.success, "Ready"),
        Some(Phase::Failed(_)) => (p.error, "Failed"),
    }
}

fn nav_bar(ui: &mut Ui, p: Palette, screen: &mut Screen) {
    egui::Frame::new()
        .fill(p.bg_surface)
        .stroke(egui::Stroke::new(theme::BORDER_WIDTH, p.border))
        .corner_radius(egui::CornerRadius::same(theme::RADIUS_CARD))
        .inner_margin(theme::GAP_XS as i8)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = theme::GAP_XS;
                let width = (ui.available_width() - 2.0 * theme::GAP_XS) / 3.0;
                for (label, target) in [
                    ("Status", Screen::Status),
                    ("Connections", Screen::Connections),
                    ("Settings", Screen::Settings),
                ] {
                    if theme::nav_tab(ui, p, label, *screen == target, width).clicked() {
                        *screen = target;
                    }
                }
            });
        });
}

// ---------------------------------------------------------------------------
// Screen: Status
// ---------------------------------------------------------------------------

fn status_screen(ui: &mut Ui, p: Palette, status: &Status, phase: Option<&Phase>) {
    ui.spacing_mut().item_spacing.y = theme::GAP_LG;

    theme::card(ui, p, |ui| model_card_body(ui, p, phase));
    theme::card(ui, p, |ui| endpoint_card_body(ui, p, phase));
    stats_row(ui, p, status);
}

fn model_card_body(ui: &mut Ui, p: Palette, phase: Option<&Phase>) {
    let fraction = phase.and_then(Phase::fraction);
    ui.horizontal(|ui| {
        ui.label(theme::rich("Model \u{b7} BAAI/bge-m3", theme::SIZE_BODY, Weight::SemiBold, p.text));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let right = match phase {
                Some(Phase::Downloading { .. }) => match fraction {
                    Some(frac) => format!("{}%", (frac * 100.0) as u32),
                    None => "\u{2026}".to_string(),
                },
                Some(Phase::Ready { .. }) => "Loaded".to_string(),
                Some(Phase::Loading) => "\u{2026}".to_string(),
                Some(Phase::Failed(_)) => "Failed".to_string(),
                None => "\u{2026}".to_string(),
            };
            ui.label(theme::rich(right, theme::SIZE_BODY, Weight::Medium, p.text));
        });
    });
    ui.add_space(theme::GAP_SM);

    ui.scope(|ui| {
        ui.visuals_mut().extreme_bg_color = p.border;
        let bar = match phase {
            Some(Phase::Downloading { .. }) => match fraction {
                Some(frac) => egui::ProgressBar::new(frac),
                None => egui::ProgressBar::new(0.0).animate(true),
            },
            Some(Phase::Loading) | None => egui::ProgressBar::new(0.0).animate(true),
            Some(Phase::Ready { .. }) => egui::ProgressBar::new(1.0),
            Some(Phase::Failed(_)) => egui::ProgressBar::new(0.0),
        };
        ui.add(
            bar.desired_height(theme::PROGRESS_HEIGHT)
                .corner_radius(theme::RADIUS_CONTROL)
                .fill(p.accent),
        );
    });
    ui.add_space(theme::GAP_XS);

    let (detail, color) = match phase {
        Some(Phase::Downloading { file, done_bytes, total_bytes }) => {
            let done_gb = *done_bytes as f64 / 1_000_000_000.0;
            let detail = match total_bytes {
                Some(total) => format!(
                    "Downloading {file} \u{b7} {done_gb:.2} / {:.2} GB",
                    *total as f64 / 1_000_000_000.0
                ),
                None => format!("Downloading {file} \u{b7} {done_gb:.2} GB"),
            };
            (detail, p.text_muted)
        }
        Some(Phase::Loading) | None => ("Loading model into memory\u{2026}".to_string(), p.text_muted),
        Some(Phase::Ready { .. }) => ("In memory \u{b7} 1024-dim vectors \u{b7} CPU (AVX2)".to_string(), p.text_muted),
        Some(Phase::Failed(msg)) => (msg.clone(), p.error),
    };
    ui.label(theme::rich(detail, theme::SIZE_CAPTION, Weight::Regular, color));
}

fn endpoint_card_body(ui: &mut Ui, p: Palette, phase: Option<&Phase>) {
    ui.label(theme::rich("Endpoint", theme::SIZE_BODY, Weight::SemiBold, p.text));
    ui.add_space(theme::GAP_SM);

    let ready_url = match phase {
        Some(Phase::Ready { url }) => Some(url.clone()),
        _ => None,
    };
    let display_url = ready_url.clone().unwrap_or_else(|| {
        format!("http://{}:{}/v1/embeddings", effective_host(), effective_port())
    });

    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        if theme::primary_button(ui, p, "Copy", ready_url.is_some()).clicked() {
            if let Some(url) = &ready_url {
                ui.ctx().copy_text(url.clone());
            }
        }
        ui.add_space(theme::GAP_SM);
        theme::url_field(ui, p, &display_url);
    });
    ui.add_space(theme::GAP_XS);

    let hint = if ready_url.is_some() {
        "OpenAI-compatible \u{b7} model name: bge-m3"
    } else {
        "Available once the model is loaded. OpenAI-compatible, model name: bge-m3"
    };
    ui.label(theme::rich(hint, theme::SIZE_CAPTION, Weight::Regular, p.text_muted));
}

fn stats_row(ui: &mut Ui, p: Palette, status: &Status) {
    let last_latency = status.last_latency_ms.load(Ordering::Relaxed);
    let last_latency = if status.requests_served.load(Ordering::Relaxed) == 0 {
        "\u{2014} ms".to_string()
    } else {
        format!("{last_latency} ms")
    };
    let stats = [
        ("Requests", status.requests_served.load(Ordering::Relaxed).to_string()),
        ("Texts embedded", status.texts_embedded.load(Ordering::Relaxed).to_string()),
        ("Last latency", last_latency),
    ];
    ui.spacing_mut().item_spacing.x = theme::GAP_MD;
    ui.columns(3, |cols| {
        for (col, (label, value)) in cols.iter_mut().zip(stats) {
            theme::card(col, p, |ui| {
                ui.spacing_mut().item_spacing.y = theme::GAP_XS;
                ui.label(theme::rich(label, theme::SIZE_CAPTION, Weight::Regular, p.text_muted));
                ui.label(theme::rich(value, theme::SIZE_SUBTITLE, Weight::SemiBold, p.text));
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Screen: Connections
// ---------------------------------------------------------------------------

fn connections_screen(ui: &mut Ui, p: Palette) {
    ui.spacing_mut().item_spacing.y = theme::GAP_LG;

    theme::card(ui, p, |ui| {
        ui.label(theme::rich(
            "Detected on this computer",
            theme::SIZE_BODY,
            Weight::SemiBold,
            p.text,
        ));
        ui.add_space(theme::GAP_MD);
        // ponytail: connector detection is a separate future task - this is
        // the empty state the task card asked for, not a placeholder list.
        ui.label(theme::rich(
            "Detection comes in the next version.",
            theme::SIZE_CAPTION,
            Weight::Regular,
            p.text_muted,
        ));
    });

    theme::card(ui, p, |ui| {
        ui.label(theme::rich(
            "Other tools (manual setup)",
            theme::SIZE_BODY,
            Weight::SemiBold,
            p.text,
        ));
        ui.add_space(theme::GAP_MD);
        ui.label(theme::rich(
            "Detection comes in the next version.",
            theme::SIZE_CAPTION,
            Weight::Regular,
            p.text_muted,
        ));
    });

    ui.label(theme::rich(
        "Tools running in Docker reach this server at host.docker.internal:11435. Enable network access in Settings on Linux.",
        theme::SIZE_CAPTION,
        Weight::Regular,
        p.text_muted,
    ));
}

// ---------------------------------------------------------------------------
// Screen: Settings
// ---------------------------------------------------------------------------

/// A Settings row: label/hint on the left, a fixed-width control on the
/// right (gap 12, per DESIGN.md). `control_width` reserves room for the
/// control so the hint wraps instead of running under it.
fn setting_row(ui: &mut Ui, p: Palette, label: &str, hint: &str, control_width: f32, control: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(ui.available_width() - control_width - theme::GAP_MD);
            ui.spacing_mut().item_spacing.y = theme::GAP_XS;
            ui.label(theme::rich(label, theme::SIZE_BODY, Weight::Medium, p.text));
            ui.label(theme::rich(hint, theme::SIZE_CAPTION, Weight::Regular, p.text_muted));
        });
        ui.with_layout(Layout::right_to_left(Align::Center), control);
    });
}

fn settings_screen(ui: &mut Ui, p: Palette) {
    ui.spacing_mut().item_spacing.y = theme::GAP_LG;

    theme::card(ui, p, |ui| {
        ui.spacing_mut().item_spacing.y = theme::GAP_MD;
        let input_size = egui::Vec2::new(theme::INPUT_WIDTH, theme::INPUT_HEIGHT);
        setting_row(
            ui,
            p,
            "Port",
            "The address tools connect to. Restart required.",
            theme::INPUT_WIDTH,
            |ui| {
                theme::input_box(ui, p, &effective_port(), input_size);
            },
        );
        setting_row(
            ui,
            p,
            "Parallel threads",
            "More threads = faster, but uses more CPU. 4 is a good default.",
            theme::INPUT_WIDTH,
            |ui| {
                theme::input_box(ui, p, &effective_parallel().to_string(), input_size);
            },
        );
    });

    theme::card(ui, p, |ui| {
        ui.spacing_mut().item_spacing.y = theme::GAP_MD;
        setting_row(
            ui,
            p,
            "Allow network and Docker access",
            "Listens on all interfaces. No password - only enable on trusted networks. Coming soon.",
            theme::TOGGLE_WIDTH,
            |ui| {
                theme::toggle_static(ui, p, false);
            },
        );
        setting_row(
            ui,
            p,
            "Start when I log in",
            "Coming soon.",
            theme::TOGGLE_WIDTH,
            |ui| {
                theme::toggle_static(ui, p, true);
            },
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_pill_maps_every_phase() {
        assert_eq!(phase_pill(None, &theme::LIGHT).0, theme::LIGHT.warning);
        assert_eq!(
            phase_pill(
                Some(&Phase::Downloading { file: "f".into(), done_bytes: 0, total_bytes: None }),
                &theme::LIGHT
            ),
            (theme::LIGHT.warning, "Downloading")
        );
        assert_eq!(phase_pill(Some(&Phase::Loading), &theme::LIGHT), (theme::LIGHT.warning, "Loading"));
        assert_eq!(
            phase_pill(Some(&Phase::Ready { url: "u".into() }), &theme::LIGHT),
            (theme::LIGHT.success, "Ready")
        );
        assert_eq!(
            phase_pill(Some(&Phase::Failed("e".into())), &theme::LIGHT),
            (theme::LIGHT.error, "Failed")
        );
    }
}
