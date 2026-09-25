//! The three-screen egui GUI from `design/DESIGN.md`: Status, Connections,
//! Settings, switched by a bottom nav bar. All styling comes from
//! `theme.rs` - no inline hex/spacing/radius/size literals here.

use crate::connectors::{self, Tool};
use crate::{effective_host, effective_parallel, effective_port, Phase, Status};
use crate::theme::{self, Palette, Weight};
use eframe::egui::{self, Align, Color32, Layout, Ui};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Status,
    Connections,
    Settings,
}

/// Per-tool Connections-screen state, shared with the background thread that
/// runs `detect`/`status`/`connect` off the GUI thread (they're blocking
/// HTTP calls; see connectors.rs). One of these per `Tool` variant.
struct ToolState {
    /// `None` until the first check completes ("Checking..." in the meantime).
    tool_status: Mutex<Option<connectors::Status>>,
    checking: AtomicBool,
    password: Mutex<String>,
    /// Which of `Tool::candidate_base_urls()` last answered `detect()` -
    /// `None` once a check has run and found nothing (`Status::NotFound`).
    base_url: Mutex<Option<&'static str>>,
    /// True only when the last `connect()` in this session actually changed
    /// the tool's embedding engine/model (not just re-confirmed it) - gates
    /// the re-index/re-embed caption.
    changed: AtomicBool,
    /// Set when the last `connect()` attempt came back
    /// `ConnectOutcome::NeedsResetConfirmation` (AnythingLLM only) - the row
    /// shows an inline "this deletes your data" confirmation instead of the
    /// normal action button until the user confirms or cancels.
    needs_reset_confirm: AtomicBool,
    /// Error message from the last failed `connect()` (e.g. AnythingLLM's
    /// `update-env` validation error), shown inline in `status.error` colour.
    last_error: Mutex<Option<String>>,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            tool_status: Mutex::new(None),
            checking: AtomicBool::new(false),
            password: Mutex::new(String::new()),
            base_url: Mutex::new(None),
            changed: AtomicBool::new(false),
            needs_reset_confirm: AtomicBool::new(false),
            last_error: Mutex::new(None),
        }
    }
}

/// Connections-screen state: one `ToolState` per `Tool::ALL` entry, plus the
/// single "Runs in Docker" toggle shared by every tool's `embed_url`.
struct ConnState {
    tools: [ToolState; 3],
    docker: AtomicBool,
}

impl Default for ConnState {
    fn default() -> Self {
        Self { tools: Default::default(), docker: AtomicBool::new(false) }
    }
}

fn tool_index(tool: Tool) -> usize {
    Tool::ALL.iter().position(|t| *t == tool).unwrap()
}

/// Spawns a background thread to (re-)run `tool`'s detection, unless one is
/// already in flight for it. Never touches the GUI thread.
fn spawn_check(tool: Tool, conn: Arc<ConnState>) {
    let state = &conn.tools[tool_index(tool)];
    if state.checking.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        let state = &conn.tools[tool_index(tool)];
        let embed_url = connectors::embedding_url(&effective_port(), conn.docker.load(Ordering::Relaxed));
        let base_url = tool.candidate_base_urls().iter().find(|u| tool.detect(u)).copied();
        *state.base_url.lock().unwrap() = base_url;
        *state.last_error.lock().unwrap() = None;
        state.needs_reset_confirm.store(false, Ordering::Relaxed);
        let status = match base_url {
            Some(base_url) => {
                let pw = state.password.lock().unwrap().clone();
                let pw = if pw.is_empty() { None } else { Some(pw.as_str()) };
                tool.status(base_url, &embed_url, pw)
            }
            None => connectors::Status::NotFound,
        };
        *state.tool_status.lock().unwrap() = Some(status);
        state.checking.store(false, Ordering::SeqCst);
    });
}

/// Spawns a background thread to run `tool`'s `connect()`, then re-checks
/// status. `confirm_reset` is only meaningful for AnythingLLM (see
/// `connectors::ConnectOutcome`) - passed through unconditionally since the
/// other two tools' `connect()` ignores it.
fn spawn_connect(tool: Tool, conn: Arc<ConnState>, confirm_reset: bool) {
    let state = &conn.tools[tool_index(tool)];
    if state.checking.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        let state = &conn.tools[tool_index(tool)];
        let Some(base_url) = *state.base_url.lock().unwrap() else {
            state.checking.store(false, Ordering::SeqCst);
            return;
        };
        let embed_url = connectors::embedding_url(&effective_port(), conn.docker.load(Ordering::Relaxed));
        let pw = state.password.lock().unwrap().clone();
        let pw_opt = if pw.is_empty() { None } else { Some(pw.as_str()) };
        *state.last_error.lock().unwrap() = None;
        state.needs_reset_confirm.store(false, Ordering::Relaxed);
        match tool.connect(base_url, &embed_url, pw_opt, confirm_reset) {
            Ok(connectors::ConnectOutcome::Done(changed)) => {
                state.changed.store(changed, Ordering::Relaxed);
            }
            Ok(connectors::ConnectOutcome::NeedsResetConfirmation) => {
                state.needs_reset_confirm.store(true, Ordering::Relaxed);
            }
            Err(e) => {
                *state.last_error.lock().unwrap() = Some(e.to_string());
            }
        }
        let status = tool.status(base_url, &embed_url, pw_opt);
        *state.tool_status.lock().unwrap() = Some(status);
        state.checking.store(false, Ordering::SeqCst);
    });
}

pub struct GuiApp {
    status: Arc<Status>,
    screen: Screen,
    prev_screen: Screen,
    conn: Arc<ConnState>,
}

impl GuiApp {
    pub fn new(status: Arc<Status>) -> Self {
        Self {
            status,
            screen: Screen::Status,
            prev_screen: Screen::Status,
            conn: Arc::new(ConnState::default()),
        }
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

                // "Detection runs on screen open" (DESIGN.md) - trigger once
                // per transition into the Connections screen.
                if self.screen == Screen::Connections && self.prev_screen != Screen::Connections {
                    for tool in Tool::ALL {
                        spawn_check(tool, self.conn.clone());
                    }
                }
                self.prev_screen = self.screen;

                match self.screen {
                    Screen::Status => status_screen(ui, p, &self.status, phase.as_ref()),
                    Screen::Connections => connections_screen(ui, p, &self.conn),
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

/// One row of the "Detected on this computer" / "Other tools" cards: avatar,
/// name + status-dot meta line, and a caller-supplied button (DESIGN.md).
fn connector_row(ui: &mut Ui, p: Palette, initials: &str, name: &str, dot: Color32, meta: &str, button: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        theme::avatar(ui, p, initials);
        ui.add_space(theme::GAP_SM);
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = theme::GAP_XS;
            ui.label(theme::rich(name, theme::SIZE_BODY, Weight::SemiBold, p.text));
            theme::status_dot_row(ui, p, dot, meta);
        });
        ui.with_layout(Layout::right_to_left(Align::Center), button);
    });
}

/// `"http://host:port"` -> `"host:port"`, for the row's meta line.
fn host_only(url: &str) -> &str {
    url.trim_start_matches("http://").trim_start_matches("https://")
}

/// One tool's row in the "Detected on this computer" card. Returns `false`
/// when the row should be hidden entirely (not detected on this computer).
fn tool_row(ui: &mut Ui, p: Palette, tool: Tool, conn: &Arc<ConnState>) -> bool {
    let state = &conn.tools[tool_index(tool)];
    let status = *state.tool_status.lock().unwrap();
    if status == Some(connectors::Status::NotFound) {
        return false;
    }
    let host = host_only(state.base_url.lock().unwrap().unwrap_or_else(|| tool.candidate_base_urls()[0]));
    let (dot, meta, show_connect) = match status {
        None => (p.text_muted, "Checking\u{2026}".to_string(), false),
        Some(connectors::Status::Connected) => (p.success, format!("Connected \u{b7} {host}"), false),
        Some(connectors::Status::NeedsKey) => (p.warning, format!("{} \u{b7} {host}", tool.needs_key_meta()), true),
        Some(connectors::Status::Found) => (p.text_muted, format!("Found \u{b7} {host}"), true),
        Some(connectors::Status::NotFound) => unreachable!("returned above"),
    };

    let awaiting_reset_confirm = state.needs_reset_confirm.load(Ordering::Relaxed);

    connector_row(ui, p, tool.initials(), tool.name(), dot, &meta, |ui| {
        if awaiting_reset_confirm {
            // Buttons live in the confirmation block below instead.
        } else if status == Some(connectors::Status::Connected) {
            if theme::secondary_button(ui, p, "Re-check", !state.checking.load(Ordering::Relaxed)).clicked() {
                spawn_check(tool, conn.clone());
            }
        } else if show_connect {
            if theme::primary_button(ui, p, "Connect", !state.checking.load(Ordering::Relaxed)).clicked() {
                spawn_connect(tool, conn.clone(), false);
            }
        } else {
            if theme::secondary_button(ui, p, "Re-check", !state.checking.load(Ordering::Relaxed)).clicked() {
                spawn_check(tool, conn.clone());
            }
        }
    });

    if status == Some(connectors::Status::NeedsKey) {
        ui.add_space(theme::GAP_XS);
        let mut pw = state.password.lock().unwrap().clone();
        if ui.add(egui::TextEdit::singleline(&mut pw).password(true).hint_text(tool.key_label())).changed() {
            *state.password.lock().unwrap() = pw;
        }
    }

    // AnythingLLM only: connect() found that applying the change would
    // delete every workspace's embedded documents and refused to write
    // anything until confirmed (connectors::ConnectOutcome::NeedsResetConfirmation).
    // Shown inline in the row, not a modal, per the coordinator's review.
    if awaiting_reset_confirm {
        ui.add_space(theme::GAP_XS);
        ui.label(theme::rich(
            "Connecting changes AnythingLLM's embedder. AnythingLLM will DELETE all embedded documents in every workspace; you must re-embed them.",
            theme::SIZE_CAPTION,
            Weight::Regular,
            p.error,
        ));
        ui.add_space(theme::GAP_XS);
        ui.horizontal(|ui| {
            if theme::secondary_button(ui, p, "Cancel", !state.checking.load(Ordering::Relaxed)).clicked() {
                state.needs_reset_confirm.store(false, Ordering::Relaxed);
            }
            ui.add_space(theme::GAP_SM);
            if theme::danger_button(ui, p, "Delete and connect", !state.checking.load(Ordering::Relaxed)).clicked() {
                spawn_connect(tool, conn.clone(), true);
            }
        });
    }

    // Surfaces a failed connect() (e.g. AnythingLLM's update-env validation
    // error - see ToolState::last_error) inline, in the row.
    if let Some(err) = state.last_error.lock().unwrap().clone() {
        ui.add_space(theme::GAP_XS);
        ui.label(theme::rich(err, theme::SIZE_CAPTION, Weight::Regular, p.error));
    }

    // Only shown when a connect() in this session actually changed the
    // embedding config (not just re-confirmed it) - see ToolState::changed.
    if status == Some(connectors::Status::Connected) && state.changed.load(Ordering::Relaxed) {
        ui.add_space(theme::GAP_XS);
        ui.label(theme::rich(tool.changed_caption(), theme::SIZE_CAPTION, Weight::Regular, p.text_muted));
    }

    true
}

fn connections_screen(ui: &mut Ui, p: Palette, conn: &Arc<ConnState>) {
    ui.spacing_mut().item_spacing.y = theme::GAP_LG;

    theme::card(ui, p, |ui| {
        ui.label(theme::rich("Detected on this computer", theme::SIZE_BODY, Weight::SemiBold, p.text));
        ui.add_space(theme::GAP_MD);
        let mut shown_any = false;
        for tool in Tool::ALL {
            if shown_any {
                ui.add_space(theme::GAP_MD);
            }
            if tool_row(ui, p, tool, conn) {
                shown_any = true;
            }
        }
        if !shown_any {
            ui.label(theme::rich(
                "None detected yet.",
                theme::SIZE_CAPTION,
                Weight::Regular,
                p.text_muted,
            ));
        }
        ui.add_space(theme::GAP_SM);
        let mut docker = conn.docker.load(Ordering::Relaxed);
        if ui.checkbox(&mut docker, "Runs in Docker").changed() {
            conn.docker.store(docker, Ordering::Relaxed);
        }
    });

    theme::card(ui, p, |ui| {
        ui.label(theme::rich("Other tools (manual setup)", theme::SIZE_BODY, Weight::SemiBold, p.text));
        ui.add_space(theme::GAP_MD);
        let embed_url = connectors::embedding_url(&effective_port(), conn.docker.load(Ordering::Relaxed));
        for (i, tool) in connectors::MANUAL_TOOLS.iter().enumerate() {
            if i > 0 {
                ui.add_space(theme::GAP_MD);
            }
            connector_row(ui, p, tool.initials, tool.name, p.text_muted, "Manual setup", |ui| {
                if theme::secondary_button(ui, p, "Copy config", true).clicked() {
                    ui.ctx().copy_text((tool.config)(&embed_url));
                }
            });
        }
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
