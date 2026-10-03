// bge-m3 embedding server - pure Rust (candle), no C/C++ runtime.
// OpenAI-compatible POST /v1/embeddings + GET /health.
// Verified to produce the same vector space as the reference llama.cpp
// server for bge-m3 (CLS pooling, cosine similarity 0.99999) - see note below.
//
// POSITION-ID NOTE: candle_transformers::models::bert::BertModel builds
// position ids as a plain 0..seq_len (bert.rs, "TODO: Proper absolute
// positions?"). bge-m3 is XLM-RoBERTa based (config.json:
// model_type=xlm-roberta, pad_token_id=1) and the RoBERTa family offsets
// positions by padding_idx:
//   position_id = 1-based index among non-pad tokens + pad_token_id
// (see HF transformers create_position_ids_from_input_ids). Without the
// offset cosine similarity to the reference stalls at ~0.90. BertModel's
// embeddings field is private, so instead of forking the whole model we
// rebuild only the embedding layer here with the right offset and reuse
// candle_transformers' public BertEncoder unchanged.
//
// GUI NOTE: double-clicking the binary opens the egui window from
// design/DESIGN.md (crates/bge-theme + crates/bge-gui) while the server runs in a background
// thread. `--headless` (or BGE_HEADLESS=1) skips the window entirely and
// behaves like the original CLI-only server.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use anyhow::{Error as E, Result, anyhow};
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, Module, VarBuilder, embedding, layer_norm};
use candle_transformers::models::bert::{BertEncoder, Config, DTYPE};
use hf_hub::{Cache, Repo, RepoType, api::tokio::Api};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokenizers::{Tokenizer, TruncationParams};

use bge_settings::{
    Phase, Status, effective_host, effective_model, effective_parallel, effective_port,
};

/// Reports hf-hub download progress into `Status`. hf-hub 0.4's tokio API
/// (`ApiRepo::download_with_progress`, src/api/tokio.rs) drives this trait
/// itself: `init` on download start with the total size, `update` per chunk
/// received, `finish` when the file is renamed into the cache.
#[derive(Clone)]
struct StatusProgress {
    status: Arc<Status>,
}

impl hf_hub::api::tokio::Progress for StatusProgress {
    async fn init(&mut self, size: usize, filename: &str) {
        self.status.set_phase(Phase::Downloading {
            file: filename.to_string(),
            done_bytes: 0,
            total_bytes: if size > 0 { Some(size as u64) } else { None },
        });
    }

    async fn update(&mut self, size: usize) {
        let mut phase = self.status.phase.lock().unwrap();
        if let Some(Phase::Downloading { done_bytes, .. }) = phase.as_mut() {
            *done_bytes += size as u64;
        }
    }

    async fn finish(&mut self) {}
}

struct EmbedModel {
    tokenizer: Tokenizer,
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    embeddings_layer_norm: LayerNorm,
    encoder: BertEncoder,
    pad_token_id: u32,
    parallel: usize,
    device: Device,
}

impl EmbedModel {
    /// Fetches `filename` from the standard HF cache if present, otherwise
    /// downloads it with progress reported into `status`. Checking the cache
    /// ourselves (instead of `ApiRepo::get`) is what lets us attach progress
    /// only to an actual download, so an existing cache never re-downloads.
    async fn fetch(
        repo: &hf_hub::api::tokio::ApiRepo,
        cache: &Cache,
        cache_repo_id: &Repo,
        filename: &str,
        status: &Arc<Status>,
    ) -> Result<std::path::PathBuf> {
        if let Some(path) = cache.repo(cache_repo_id.clone()).get(filename) {
            return Ok(path);
        }
        let progress = StatusProgress {
            status: status.clone(),
        };
        Ok(repo.download_with_progress(filename, progress).await?)
    }

    async fn load(status: Arc<Status>) -> Result<Self> {
        let device = Device::Cpu;
        let model_name = effective_model();
        println!(
            "Loading {model_name} (downloaded from the Hugging Face hub on first run if not cached)..."
        );
        let api = Api::new()?;
        let repo_id = Repo::new(model_name, RepoType::Model);
        let repo = api.repo(repo_id.clone());
        // Same default cache location Api::new() uses (Cache::default()), so
        // an already-populated cache from a previous run is found unchanged.
        let cache = Cache::default();

        let config_path = Self::fetch(&repo, &cache, &repo_id, "config.json", &status).await?;
        let tokenizer_path =
            Self::fetch(&repo, &cache, &repo_id, "tokenizer.json", &status).await?;
        // Assumes the repo ships pytorch_model.bin (true for BAAI/bge-m3);
        // a safetensors-only alternate model repo would fail to fetch here.
        let weights_path =
            Self::fetch(&repo, &cache, &repo_id, "pytorch_model.bin", &status).await?;
        status.set_phase(Phase::Loading);

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
        // Positions run 2..=seq_len+1, so the model accepts at most max_position_embeddings-2 tokens.
        let tokenizer = load_tokenizer(&tokenizer_path, config.max_position_embeddings - 2)?;
        let vb = VarBuilder::from_pth(&weights_path, DTYPE, &device)?;

        let embeddings_vb = vb.pp("embeddings");
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_vb.pp("token_type_embeddings"),
        )?;
        let embeddings_layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            embeddings_vb.pp("LayerNorm"),
        )?;
        let encoder = BertEncoder::load(vb.pp("encoder"), &config)?;

        println!("Model loaded. hidden_size={}", config.hidden_size);
        Ok(Self {
            tokenizer,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_layer_norm,
            encoder,
            pad_token_id: config.pad_token_id as u32,
            parallel: effective_parallel(),
            device,
        })
    }

    /// CLS pooling + L2 normalization (the recipe verified against the reference server).
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let ids = encoding.get_ids();
        let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids = input_ids.zeros_like()?;

        let seq_len = ids.len();
        let position_ids: Vec<u32> = (0..seq_len as u32)
            .map(|i| i + 1 + self.pad_token_id)
            .collect();
        let position_ids = Tensor::new(&position_ids[..], &self.device)?;

        let input_embeddings = self.word_embeddings.forward(&input_ids)?;
        let token_type_embeds = self.token_type_embeddings.forward(&token_type_ids)?;
        let embeddings = (&input_embeddings + token_type_embeds)?;
        let embeddings =
            embeddings.broadcast_add(&self.position_embeddings.forward(&position_ids)?)?;
        let embedding_output = self.embeddings_layer_norm.forward(&embeddings)?;

        // A single sequence has no padding, so the extended mask is a no-op;
        // the standard formula is kept for correctness.
        let attention_mask = input_ids.ones_like()?;
        let extended_mask = get_extended_attention_mask(&attention_mask, DTYPE)?;

        let output = self.encoder.forward(&embedding_output, &extended_mask)?;
        let cls = output.i((.., 0, ..))?;
        let cls_norm = normalize_l2(&cls)?;
        Ok(cls_norm.squeeze(0)?.to_vec1()?)
    }
}

/// tokenizer.json defines no truncation; over-long input would overflow the position table. Truncates silently.
fn load_tokenizer(path: &std::path::Path, max_tokens: usize) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(E::msg)?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_tokens,
            ..Default::default()
        }))
        .map_err(E::msg)?;
    Ok(tokenizer)
}

fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(1)?;
    let attention_mask = attention_mask.to_dtype(dtype)?;
    Ok(
        (attention_mask.ones_like()? - &attention_mask)?.broadcast_mul(
            &Tensor::try_from(f32::MIN)?
                .to_device(attention_mask.device())?
                .to_dtype(dtype)?,
        )?,
    )
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?)?)
}

/// Embeds inputs on at most `parallel` threads; output keeps input order.
/// On the first error the other workers stop taking new inputs and the error is returned.
fn embed_parallel<F>(inputs: &[String], parallel: usize, f: F) -> Result<Vec<Vec<f32>>>
where
    F: Fn(&str) -> Result<Vec<f32>> + Sync,
{
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let workers = parallel.min(inputs.len()).max(1);
    let parts: Vec<Result<Vec<(usize, Vec<f32>)>>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| -> Result<Vec<(usize, Vec<f32>)>> {
                    let mut out = Vec::new();
                    while !failed.load(Ordering::Relaxed) {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= inputs.len() {
                            break;
                        }
                        match f(&inputs[i]) {
                            Ok(v) => out.push((i, v)),
                            Err(e) => {
                                failed.store(true, Ordering::Relaxed);
                                return Err(e);
                            }
                        }
                    }
                    Ok(out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow!("embedding thread panicked")))
            })
            .collect()
    });
    let mut all = Vec::with_capacity(inputs.len());
    for part in parts {
        all.extend(part?);
    }
    all.sort_by_key(|(i, _)| *i);
    Ok(all.into_iter().map(|(_, v)| v).collect())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

impl EmbeddingInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::One(s) => vec![s],
            EmbeddingInput::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    model: String,
    input: EmbeddingInput,
}

#[derive(Serialize)]
struct EmbeddingObject {
    embedding: Vec<f32>,
    index: usize,
    object: &'static str,
}

#[derive(Serialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingObject>,
    object: &'static str,
    model: String,
}

/// axum handler state: the model plus the shared status counters it updates.
#[derive(Clone)]
struct AppState {
    model: Arc<EmbedModel>,
    status: Arc<Status>,
}

async fn embeddings_handler(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, String)> {
    let inputs = req.input.into_vec();
    let text_count = inputs.len();
    let started = std::time::Instant::now();
    let model = state.model.clone();
    // Iki serit: tek ve kisa metinli istek (arama sorgusu) kapiyi atlar; toplu istekler
    // (Open Notebook) TOPLU_KAPI'dan birer birer gecer. Kapi yokken N eszamanli toplu istek
    // N*parallel is parcacigi aciyordu ve kisa sorgu CPU'dan 1/(N*parallel+1) pay aliyordu
    // (2026-09-25 olculdu: ayni sorgu yuk altinda 7,4 / 47 / 2,8 sn, bosken ~0,65 sn).
    let _izin = if kisa_mi(&inputs) {
        None
    } else {
        Some(
            TOPLU_KAPI
                .acquire()
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
        )
    };
    // Sequential embedding keeps only ~3 cores busy on candle's CPU backend (non-matmul ops
    // are single-threaded). Measured on an idle 6-core/12-thread Ryzen, 20 x ~1560-char
    // chunks, s/chunk: BGE_PARALLEL 1=4.23 2=3.00 3=2.32 4=1.91 6=1.96 -> default 4.
    let result = tokio::task::spawn_blocking(move || {
        embed_parallel(&inputs, model.parallel, |text| model.embed_one(text))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    state
        .status
        .record_request(text_count, started.elapsed().as_millis() as u64);

    let data = result
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingObject {
            embedding,
            index,
            object: "embedding",
        })
        .collect();

    Ok(Json(EmbeddingResponse {
        data,
        object: "list",
        model: req.model,
    }))
}

/// Toplu isteklerin gectigi kapi; izin sayisi `BGE_TOPLU_IZIN` (ayar dugmesi).
/// Olcum (2026-09-25, 3 istemci, sirasi degisen 3 tur, canli sunucu da yuklu):
///   kapisiz  : kisa sorgu medyan ~5,6 sn · toplu 0,63 parca/sn
///   izin = 1 : kisa sorgu medyan ~1,8 sn · toplu 0,46 parca/sn (-%27)
/// Ust uste binen toplu istekler birbirinin bos is parcacigini kullaniyor; izin
/// arttikca toplu verim geri gelir, kisa sorgu yavaslar.
static TOPLU_KAPI: std::sync::LazyLock<tokio::sync::Semaphore> = std::sync::LazyLock::new(|| {
    let izin = std::env::var("BGE_TOPLU_IZIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n >= 1)
        .unwrap_or(VARSAYILAN_TOPLU_IZIN);
    tokio::sync::Semaphore::new(izin)
});
const VARSAYILAN_TOPLU_IZIN: usize = 1;

/// Hizli serit: tek girdi ve arama sorgusu boyunda (~512 karakter; tipik sorgu 20-80).
fn kisa_mi(inputs: &[String]) -> bool {
    inputs.len() == 1 && inputs[0].chars().count() <= 512
}

async fn health() -> impl IntoResponse {
    StatusCode::OK
}

/// Loads the model and serves the HTTP API. Identical in headless and GUI
/// mode; only who calls it (main thread vs. a background thread) differs.
async fn run_server(status: Arc<Status>) -> Result<()> {
    let model = Arc::new(EmbedModel::load(status.clone()).await?);

    let host = effective_host();
    let port = effective_port();
    let addr = format!("{host}:{port}");
    let url = format!("http://{addr}/v1/embeddings");

    let app = Router::new()
        .route("/v1/embeddings", post(embeddings_handler))
        .route("/health", axum::routing::get(health))
        .with_state(AppState {
            model,
            status: status.clone(),
        });

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("bge-embed-rs listening on {url}");
    status.set_phase(Phase::Ready { url });
    axum::serve(listener, app).await?;
    Ok(())
}

fn headless_requested() -> bool {
    std::env::args().any(|a| a == "--headless")
        || std::env::var("BGE_HEADLESS").is_ok_and(|v| v == "1")
}

/// The binary is built for x86-64-v3; without AVX2+FMA candle's CPU ops would
/// SIGILL. Checked once and shared by both startup paths below.
fn avx2_fma_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

const AVX2_FMA_MISSING_MSG: &str =
    "this build needs a CPU with AVX2 and FMA (Intel 2013+, AMD 2015+)";

fn main() -> Result<()> {
    let headless = headless_requested();

    // A GUI-subsystem Windows binary has no console; reattach to the
    // launching terminal's so --headless output is still visible there.
    #[cfg(windows)]
    if headless {
        unsafe {
            windows_sys::Win32::System::Console::AttachConsole(
                windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS,
            );
        }
    }

    let status = Arc::new(Status::default());

    if headless {
        // No window to show a message in, so fail loudly on stdout/exit code
        // instead of limping into a server that would SIGILL on first request.
        if !avx2_fma_available() {
            return Err(anyhow!(AVX2_FMA_MISSING_MSG));
        }
        let rt = tokio::runtime::Runtime::new()?;
        return rt.block_on(run_server(status));
    }

    // GUI mode: eframe owns the main thread (required on macOS), a background
    // thread owns its own multi-thread tokio runtime for model loading + serving.
    // A windows_subsystem = "windows" build has no console, so an early
    // `return Err(...)` here would be invisible - the app would just fail to
    // open with no explanation. Instead skip starting the server and let the
    // window itself show the failure via Phase::Failed.
    if avx2_fma_available() {
        let server_status = status.clone();
        std::thread::spawn(move || {
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    server_status.set_phase(Phase::Failed(e.to_string()));
                    return;
                }
            };
            if let Err(e) = rt.block_on(run_server(server_status.clone())) {
                server_status.set_phase(Phase::Failed(e.to_string()));
            }
        });
    } else {
        status.set_phase(Phase::Failed(AVX2_FMA_MISSING_MSG.to_string()));
    }

    let native_options = eframe::NativeOptions {
        // DX12 via wgpu, not glow/OpenGL - see the Cargo.toml comment on the
        // `eframe` dependency (el-fihrist kural 9037960d582ebfe7: glow blanks
        // to a black window on some Windows 11 + AMD/Intel-iGPU combinations).
        renderer: eframe::Renderer::Wgpu,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("bge-embed-rs")
            .with_inner_size([480.0, 560.0])
            .with_min_inner_size([480.0, 560.0])
            .with_transparent(false),
        ..Default::default()
    };
    eframe::run_native(
        "bge-embed-rs",
        native_options,
        Box::new(|cc| {
            bge_theme::apply(&cc.egui_ctx);
            Ok(Box::new(bge_gui::GuiApp::new(status, &cc.egui_ctx)))
        }),
    )
    .map_err(|e| anyhow!("gui error: {e}"))
    // ponytail: closing the window returns here and main() ends, which tears
    // down the whole process (and the server thread with it) - no separate
    // shutdown signal needed. Add graceful shutdown if in-flight requests
    // ever need to drain before exit.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_input_order_under_parallelism() {
        let inputs: Vec<String> = (0..40).map(|i| i.to_string()).collect();
        let out = embed_parallel(&inputs, 4, |t| {
            let n: u64 = t.parse().unwrap();
            std::thread::sleep(std::time::Duration::from_millis((40 - n) % 7));
            Ok(vec![n as f32])
        })
        .unwrap();
        let got: Vec<u64> = out.iter().map(|v| v[0] as u64).collect();
        assert_eq!(got, (0..40).collect::<Vec<u64>>());
    }

    // Skipped silently when bge-m3 is not in the local Hugging Face cache.
    #[test]
    fn truncates_to_model_limit() {
        let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
        else {
            return;
        };
        let snaps = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--BAAI--bge-m3/snapshots");
        let Some(dir) = std::fs::read_dir(snaps)
            .ok()
            .and_then(|mut d| d.next())
            .and_then(|e| e.ok())
        else {
            return;
        };
        let path = dir.path().join("tokenizer.json");
        if !path.exists() {
            return;
        }
        let tok = load_tokenizer(&path, 8192).unwrap();
        let long = "kelime ".repeat(20_000);
        assert_eq!(
            tok.encode(long.as_str(), true).unwrap().get_ids().len(),
            8192
        );
        assert!(tok.encode("kisa metin", true).unwrap().get_ids().len() < 10);
    }

    #[test]
    fn kisa_serit_yalniz_tek_ve_kisa_girdi() {
        assert!(kisa_mi(&["kök neden bulma".to_string()]));
        assert!(
            !kisa_mi(&["a".to_string(), "b".to_string()]),
            "iki girdi toplu sayılır"
        );
        assert!(!kisa_mi(&["ş".repeat(513)]), "uzun tek girdi toplu sayılır");
        assert!(!kisa_mi(&[]), "boş istek hızlı şeride girmez");
    }

    #[test]
    fn propagates_errors() {
        let inputs: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        let r = embed_parallel(&inputs, 3, |t| {
            if t == "5" {
                Err(anyhow!("boom"))
            } else {
                Ok(vec![0.0])
            }
        });
        assert!(r.is_err());
        assert!(
            embed_parallel(&[], 4, |_| Ok(vec![1.0]))
                .unwrap()
                .is_empty()
        );
    }
}
