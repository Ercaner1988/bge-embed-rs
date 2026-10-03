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

/// The per-user settings directory (`%APPDATA%\bge-embed-rs` on Windows,
/// `$HOME/.config/bge-embed-rs` elsewhere) - created on first write.
pub fn settings_dir() -> std::path::PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(std::path::PathBuf::from);
    #[cfg(not(windows))]
    let base = std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config"));
    base.unwrap_or_else(std::env::temp_dir).join("bge-embed-rs")
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
