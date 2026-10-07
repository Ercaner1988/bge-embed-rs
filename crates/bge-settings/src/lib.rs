// Per-user settings (env > saved file > default) and the Phase/Status the server
// publishes for the window. No dependencies: the server and the GUI both build on it.
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

/// Server defaults, also read back by the Settings/Status screens (`gui.rs`)
/// so they never show an address the server didn't actually bind to.
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: &str = "11435";
pub const DEFAULT_PARALLEL: usize = 4;
/// Only BGE-M3-architecture-compatible repos work: `EmbedModel::load` assumes
/// an XLM-RoBERTa-style position-id offset (padding-based, not plain 0..len -
/// see the POSITION-ID NOTE above) and a `pytorch_model.bin` weights file. A
/// different architecture would load without error and silently produce
/// wrong embeddings.
pub const DEFAULT_MODEL: &str = "BAAI/bge-m3";

/// Loopback only by default. BGE_HOST=0.0.0.0 exposes the server to your LAN
/// and to Docker containers on Linux - there is no authentication.
pub fn effective_host() -> String {
    std::env::var("BGE_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string())
}

pub fn effective_port() -> String {
    std::env::var("BGE_PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string())
}

pub fn effective_parallel() -> usize {
    std::env::var("BGE_PARALLEL")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(DEFAULT_PARALLEL)
}

/// The app's identity on disk, in the registry and in other tools: settings
/// directory, autostart value, theme files, Open Notebook credential. Named
/// after Ibn al-Nadim (README "Name"); the catalogue CLI in el-Fihrist is the
/// separate `ibnunnedim`.
pub const APP_ID: &str = "ibnun-nedim";
/// Display name (window title, header).
pub const DISPLAY_NAME: &str = "İbnü'n-Nedîm Gömme";
/// The name before 2026-10: `migrate_old_name` carries its data over.
pub const OLD_APP_ID: &str = "bge-embed-rs";

fn settings_base() -> std::path::PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(std::path::PathBuf::from);
    #[cfg(not(windows))]
    let base = std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config"));
    base.unwrap_or_else(std::env::temp_dir)
}

/// The per-user settings directory (`%APPDATA%\ibnun-nedim` on Windows,
/// `$HOME/.config/ibnun-nedim` elsewhere) - created on first write.
pub fn settings_dir() -> std::path::PathBuf {
    settings_base().join(APP_ID)
}

/// kilim-tema's per-app file (`<config>/kilim-tema/<app>.<ext>`). Mirrors
/// kilim-tema's private `paket::uygulama_dosyasi` (rev a62bfc7, paket.rs:238);
/// only `migrate_old_name` uses it, to carry the saved theme across the rename.
fn kilim_tema_file(app: &str, ext: &str) -> Option<std::path::PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(std::path::PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME")
            .map(|h| std::path::PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config"))
            })
    }?;
    Some(base.join("kilim-tema").join(format!("{app}.{ext}")))
}

/// Carries data saved under `OLD_APP_ID` over to `APP_ID`, once: the settings
/// directory is moved, kilim-tema's theme choice and theme package are
/// copied. Anything already present under the new name wins and is left
/// alone, so running it on every start is harmless. Returns what was moved,
/// for the log; failures are reported, not fatal (the app then starts with
/// defaults, as on a fresh install).
pub fn migrate_old_name() -> Vec<String> {
    let mut done = Vec::new();
    match migrate_dir(&settings_base().join(OLD_APP_ID), &settings_dir()) {
        Ok(true) => done.push(format!("settings: {OLD_APP_ID} -> {APP_ID}")),
        Ok(false) => {}
        Err(e) => done.push(format!(
            "settings not moved ({OLD_APP_ID} -> {APP_ID}): {e}"
        )),
    }
    for ext in ["txt", "tema"] {
        if let (Some(old), Some(new)) = (
            kilim_tema_file(OLD_APP_ID, ext),
            kilim_tema_file(APP_ID, ext),
        ) {
            match copy_if_missing(&old, &new) {
                Ok(true) => done.push(format!("theme: {}", new.display())),
                Ok(false) => {}
                Err(e) => done.push(format!("theme not copied ({}): {e}", old.display())),
            }
        }
    }
    done
}

/// Moves `old` to `new` when only `old` exists. `Ok(true)` if it moved.
fn migrate_dir(old: &std::path::Path, new: &std::path::Path) -> std::io::Result<bool> {
    if new.exists() || !old.is_dir() {
        return Ok(false);
    }
    std::fs::rename(old, new)?;
    Ok(true)
}

/// Copies `old` to `new` when only `old` exists. `Ok(true)` if it copied.
fn copy_if_missing(old: &std::path::Path, new: &std::path::Path) -> std::io::Result<bool> {
    if new.exists() || !old.is_file() {
        return Ok(false);
    }
    if let Some(dir) = new.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::copy(old, new)?;
    Ok(true)
}

pub fn model_config_path() -> std::path::PathBuf {
    settings_dir().join("model.txt")
}

/// `BGE_MODEL` env var wins (so `start.ps1`-style launchers stay in control);
/// otherwise the Settings screen's last saved choice; otherwise `DEFAULT_MODEL`.
pub fn effective_model() -> String {
    if let Ok(v) = std::env::var("BGE_MODEL") {
        if !v.trim().is_empty() {
            return v;
        }
    }
    if let Ok(saved) = std::fs::read_to_string(model_config_path()) {
        let saved = saved.trim();
        if !saved.is_empty() {
            return saved.to_string();
        }
    }
    DEFAULT_MODEL.to_string()
}

/// Persists the Settings screen's model field so it survives a restart.
/// Takes effect on next launch only - loading a different model requires
/// re-downloading it and re-initializing the encoder.
pub fn save_model_setting(repo_id: &str) -> std::io::Result<()> {
    let dir = settings_dir();
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("model.txt"), repo_id.trim())
}

/// Current phase of the server, shared with the GUI thread.
#[derive(Clone, Debug)]
pub enum Phase {
    Downloading {
        file: String,
        done_bytes: u64,
        total_bytes: Option<u64>,
    },
    Loading,
    Ready {
        url: String,
    },
    Failed(String),
}

impl Phase {
    /// Fraction in [0, 1] for a progress bar, or None if the total is unknown.
    pub fn fraction(&self) -> Option<f32> {
        match self {
            Phase::Downloading {
                done_bytes,
                total_bytes: Some(total),
                ..
            } if *total > 0 => Some(*done_bytes as f32 / *total as f32),
            _ => None,
        }
    }
}

/// Shared status: phase behind a mutex (rarely changes), counters as atomics
/// (updated on every request without contending the phase lock).
#[derive(Default)]
pub struct Status {
    pub phase: Mutex<Option<Phase>>,
    pub requests_served: AtomicU64,
    pub texts_embedded: AtomicU64,
    pub last_latency_ms: AtomicU64,
}

impl Status {
    pub fn set_phase(&self, phase: Phase) {
        *self.phase.lock().unwrap() = Some(phase);
    }

    pub fn record_request(&self, texts: usize, latency_ms: u64) {
        self.requests_served.fetch_add(1, Ordering::Relaxed);
        self.texts_embedded
            .fetch_add(texts as u64, Ordering::Relaxed);
        self.last_latency_ms.store(latency_ms, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(ad: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("bge-settings-{ad}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn migrate_dir_moves_only_when_new_is_absent() {
        let d = temp("dir");
        let (old, new) = (d.join("old"), d.join("new"));
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("model.txt"), "BAAI/bge-m3").unwrap();
        assert!(migrate_dir(&old, &new).unwrap());
        assert_eq!(
            std::fs::read_to_string(new.join("model.txt")).unwrap(),
            "BAAI/bge-m3"
        );
        assert!(!old.exists());
        // Second run: nothing left to move.
        assert!(!migrate_dir(&old, &new).unwrap());
        // New already present: the old one is never moved over it.
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("model.txt"), "eski").unwrap();
        assert!(!migrate_dir(&old, &new).unwrap());
        assert_eq!(
            std::fs::read_to_string(new.join("model.txt")).unwrap(),
            "BAAI/bge-m3"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn copy_if_missing_never_overwrites() {
        let d = temp("copy");
        let (old, new) = (d.join("a.txt"), d.join("sub/b.txt"));
        assert!(!copy_if_missing(&old, &new).unwrap());
        std::fs::write(&old, "koyu").unwrap();
        assert!(copy_if_missing(&old, &new).unwrap());
        assert_eq!(std::fs::read_to_string(&new).unwrap(), "koyu");
        std::fs::write(&old, "acik").unwrap();
        assert!(!copy_if_missing(&old, &new).unwrap());
        assert_eq!(std::fs::read_to_string(&new).unwrap(), "koyu");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn progress_fraction_none_when_total_unknown() {
        let p = Phase::Downloading {
            file: "pytorch_model.bin".into(),
            done_bytes: 1_000,
            total_bytes: None,
        };
        assert_eq!(p.fraction(), None);
    }

    #[test]
    fn progress_fraction_known_total() {
        let p = Phase::Downloading {
            file: "pytorch_model.bin".into(),
            done_bytes: 50,
            total_bytes: Some(200),
        };
        assert_eq!(p.fraction(), Some(0.25));
    }

    #[test]
    fn failed_phase_carries_message() {
        let p = Phase::Failed("network error".to_string());
        match p {
            Phase::Failed(msg) => assert_eq!(msg, "network error"),
            _ => panic!("expected Failed"),
        }
    }
}
