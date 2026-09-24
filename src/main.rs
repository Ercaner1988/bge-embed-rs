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

use anyhow::{anyhow, Error as E, Result};
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, Module, VarBuilder};
use candle_transformers::models::bert::{BertEncoder, Config, DTYPE};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use tokenizers::{Tokenizer, TruncationParams};

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
    async fn load() -> Result<Self> {
        let device = Device::Cpu;
        println!("Loading BAAI/bge-m3 (downloaded from the Hugging Face hub on first run, ~2.2 GB)...");
        let api = Api::new()?;
        let repo = api.repo(Repo::new("BAAI/bge-m3".to_string(), RepoType::Model));

        let config_path = repo.get("config.json").await?;
        let tokenizer_path = repo.get("tokenizer.json").await?;
        // The BAAI/bge-m3 repo ships pytorch_model.bin only (no safetensors).
        let weights_path = repo.get("pytorch_model.bin").await?;

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
            parallel: std::env::var("BGE_PARALLEL")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(4),
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
    Ok((attention_mask.ones_like()? - &attention_mask)?.broadcast_mul(
        &Tensor::try_from(f32::MIN)?
            .to_device(attention_mask.device())?
            .to_dtype(dtype)?,
    )?)
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
            .map(|h| h.join().unwrap_or_else(|_| Err(anyhow!("embedding thread panicked"))))
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

async fn embeddings_handler(
    State(model): State<Arc<EmbedModel>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, String)> {
    let inputs = req.input.into_vec();
    // Sequential embedding keeps only ~3 cores busy on candle's CPU backend (non-matmul ops
    // are single-threaded). Measured on an idle 6-core/12-thread Ryzen, 20 x ~1560-char
    // chunks, s/chunk: BGE_PARALLEL 1=4.23 2=3.00 3=2.32 4=1.91 6=1.96 -> default 4.
    let result = tokio::task::spawn_blocking(move || {
        embed_parallel(&inputs, model.parallel, |text| model.embed_one(text))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

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

async fn health() -> impl IntoResponse {
    StatusCode::OK
}

#[tokio::main]
async fn main() -> Result<()> {
    // The binary is built for x86-64-v3; fail with a readable message instead of SIGILL.
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        return Err(anyhow!("this build needs a CPU with AVX2 and FMA (Intel 2013+, AMD 2015+)"));
    }

    let model = Arc::new(EmbedModel::load().await?);

    let app = Router::new()
        .route("/v1/embeddings", post(embeddings_handler))
        .route("/health", axum::routing::get(health))
        .with_state(model);

    // Loopback only by default. BGE_HOST=0.0.0.0 exposes the server to your LAN and to
    // Docker containers on Linux - there is no authentication.
    let host = std::env::var("BGE_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("BGE_PORT").unwrap_or_else(|_| "11435".to_string());
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("bge-embed-rs listening on http://{addr}/v1/embeddings");
    axum::serve(listener, app).await?;
    Ok(())
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
        let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else {
            return;
        };
        let snaps = std::path::Path::new(&home).join(".cache/huggingface/hub/models--BAAI--bge-m3/snapshots");
        let Some(dir) = std::fs::read_dir(snaps).ok().and_then(|mut d| d.next()).and_then(|e| e.ok()) else {
            return;
        };
        let path = dir.path().join("tokenizer.json");
        if !path.exists() {
            return;
        }
        let tok = load_tokenizer(&path, 8192).unwrap();
        let long = "kelime ".repeat(20_000);
        assert_eq!(tok.encode(long.as_str(), true).unwrap().get_ids().len(), 8192);
        assert!(tok.encode("kisa metin", true).unwrap().get_ids().len() < 10);
    }

    #[test]
    fn propagates_errors() {
        let inputs: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        let r = embed_parallel(&inputs, 3, |t| {
            if t == "5" { Err(anyhow!("boom")) } else { Ok(vec![0.0]) }
        });
        assert!(r.is_err());
        assert!(embed_parallel(&[], 4, |_| Ok(vec![1.0])).unwrap().is_empty());
    }
}
