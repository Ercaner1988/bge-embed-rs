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
// embeddings field is private, so the embedding layer is rebuilt here with the
// right offset. The encoder is also our own (Katman + src/hizli.rs): same BERT
// math as candle_transformers' BertEncoder (verified cosine 1.000000 against it)
// but batched with dynamic padding and with parallel/fused CPU kernels.
//
// GUI NOTE: double-clicking the binary opens the egui window from
// design/DESIGN.md (crates/bge-theme + crates/bge-gui) while the server runs in a background
// thread. `--headless` (or BGE_HEADLESS=1) skips the window entirely and
// behaves like the original CLI-only server.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use anyhow::{Error as E, Result, anyhow};
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, Module, VarBuilder, embedding, layer_norm};
use candle_transformers::models::bert::{Config, DTYPE, HiddenAct};
use hf_hub::{Cache, Repo, RepoType, api::tokio::Api};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokenizers::{Tokenizer, TruncationParams};

mod hizli;
use hizli::{ArtikLn, BiasGelu, Dikkat, linear};

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

/// Bir transformer katmanı; ağırlıklar yüklemede bir kez düzleştirilir (Q/K/V tek matriste,
/// bias ve LayerNorm vektörleri düz dilim) ki `hizli` çekirdekleri doğrudan kullansın.
struct Katman {
    qkv_w: Tensor,
    qkv_b: Vec<f32>,
    o_w: Tensor,
    o_b: Vec<f32>,
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    i_w: Tensor,
    i_b: Vec<f32>,
    d_w: Tensor,
    d_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
}

impl Katman {
    fn load(vb: VarBuilder, c: &Config) -> Result<Self> {
        let (h, ara) = (c.hidden_size, c.intermediate_size);
        let att = vb.pp("attention");
        let (s, o) = (att.pp("self"), att.pp("output"));
        let w = |vb: &VarBuilder, ad: &str, cikti: usize, girdi: usize| -> Result<Tensor> {
            Ok(vb.pp(ad).get((cikti, girdi), "weight")?)
        };
        let v = |vb: &VarBuilder, ad: &str, n: usize, alan: &str| -> Result<Vec<f32>> {
            Ok(vb.pp(ad).get(n, alan)?.to_vec1::<f32>()?)
        };
        let qkv_w = Tensor::cat(
            &[
                &w(&s, "query", h, h)?,
                &w(&s, "key", h, h)?,
                &w(&s, "value", h, h)?,
            ],
            0,
        )?;
        let mut qkv_b = Vec::with_capacity(3 * h);
        for ad in ["query", "key", "value"] {
            qkv_b.extend(v(&s, ad, h, "bias")?);
        }
        let (inter, cik) = (vb.pp("intermediate"), vb.pp("output"));
        Ok(Self {
            qkv_w,
            qkv_b,
            o_w: w(&o, "dense", h, h)?,
            o_b: v(&o, "dense", h, "bias")?,
            ln1_w: v(&o, "LayerNorm", h, "weight")?,
            ln1_b: v(&o, "LayerNorm", h, "bias")?,
            i_w: w(&inter, "dense", ara, h)?,
            i_b: v(&inter, "dense", ara, "bias")?,
            d_w: w(&cik, "dense", h, ara)?,
            d_b: v(&cik, "dense", h, "bias")?,
            ln2_w: v(&cik, "LayerNorm", h, "weight")?,
            ln2_b: v(&cik, "LayerNorm", h, "bias")?,
        })
    }
}

struct EmbedModel {
    tokenizer: Tokenizer,
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    embeddings_layer_norm: LayerNorm,
    katmanlar: Vec<Katman>,
    heads: usize,
    ln_eps: f32,
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
        if config.hidden_act != HiddenAct::Gelu {
            return Err(anyhow!(
                "yalnız hidden_act=gelu (erf) destekleniyor: {:?}",
                config.hidden_act
            ));
        }
        let katmanlar = (0..config.num_hidden_layers)
            .map(|i| Katman::load(vb.pp(format!("encoder.layer.{i}")), &config))
            .collect::<Result<Vec<_>>>()?;

        println!("Model loaded. hidden_size={}", config.hidden_size);
        Ok(Self {
            tokenizer,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_layer_norm,
            katmanlar,
            heads: config.num_attention_heads,
            ln_eps: config.layer_norm_eps as f32,
            pad_token_id: config.pad_token_id as u32,
            parallel: effective_parallel(),
            device,
        })
    }

    /// Bir partiyi (sağdan 0-doldurulmuş, en uzun girdiye kadar) tek ileri geçişte gömer.
    /// CLS pooling + L2 normalizasyonu (referans sunucuyla doğrulanan tarif). Doldurulan
    /// anahtarlar dikkatten çıkarıldığından bir girdinin sonucu partiye bağlı değildir.
    fn embed_batch(&self, ids: &[&[u32]], toplu: bool) -> Result<Vec<Vec<f32>>> {
        let b = ids.len();
        let uzun: Vec<usize> = ids.iter().map(|s| s.len()).collect();
        let l = uzun.iter().copied().max().unwrap_or(0);
        let pad = self.pad_token_id;
        let mut tok = vec![pad; b * l];
        let mut pos = vec![pad; b * l];
        for (r, s) in ids.iter().enumerate() {
            tok[r * l..r * l + s.len()].copy_from_slice(s);
            for i in 0..s.len() {
                pos[r * l + i] = i as u32 + 1 + pad;
            }
        }
        let input_ids = Tensor::from_vec(tok, (b, l), &self.device)?;
        let position_ids = Tensor::from_vec(pos, (b, l), &self.device)?;
        let token_type_ids = input_ids.zeros_like()?;

        let embeddings = (&self.word_embeddings.forward(&input_ids)?
            + self.token_type_embeddings.forward(&token_type_ids)?)?;
        let embeddings = (embeddings + self.position_embeddings.forward(&position_ids)?)?;
        let h = embeddings.dim(2)?;
        let mut x = self
            .embeddings_layer_norm
            .forward(&embeddings)?
            .reshape((b * l, h))?;

        for k in &self.katmanlar {
            if toplu {
                hizliya_yol_ver();
            }
            let qkv = linear(&x, &k.qkv_w)?;
            let baglam = qkv.apply_op1_no_bwd(&Dikkat {
                bias: &k.qkv_b,
                uzunluk: &uzun,
                bas: self.heads,
                bas_boyu: h / self.heads,
                l,
            })?;
            let a = linear(&baglam, &k.o_w)?;
            x = a.apply_op2_no_bwd(
                &x,
                &ArtikLn {
                    bias: &k.o_b,
                    agirlik: &k.ln1_w,
                    kayma: &k.ln1_b,
                    eps: self.ln_eps,
                },
            )?;
            let ara = linear(&x, &k.i_w)?.apply_op1_no_bwd(&BiasGelu(&k.i_b))?;
            let d = linear(&ara, &k.d_w)?;
            x = d.apply_op2_no_bwd(
                &x,
                &ArtikLn {
                    bias: &k.d_b,
                    agirlik: &k.ln2_w,
                    kayma: &k.ln2_b,
                    eps: self.ln_eps,
                },
            )?;
        }
        let cls = x.reshape((b, l, h))?.i((.., 0, ..))?;
        Ok(normalize_l2(&cls)?.to_vec2()?)
    }

    /// Girdileri uzunluğa göre sıralayıp parti-parti gömer; sonuç girdi sırasındadır.
    /// Parti en uzun üyesine doldurulduğundan benzer uzunlukları yan yana toplamak boşa
    /// hesabı en aza indirir (eskiden her girdi tek tek, doldurmasız geçiyordu).
    fn embed_many(&self, texts: &[String], toplu: bool) -> Result<(Vec<Vec<f32>>, Profil)> {
        let t0 = std::time::Instant::now();
        let enc: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| {
                let e = self.tokenizer.encode(t.as_str(), true).map_err(E::msg)?;
                Ok(e.get_ids().to_vec())
            })
            .collect::<Result<_>>()?;
        let tok_ms = t0.elapsed().as_millis() as u64;
        let partiler = partile(&enc.iter().map(Vec::len).collect::<Vec<_>>());
        let sonuc = embed_parallel(&partiler, self.parallel, |parti| {
            let ids: Vec<&[u32]> = parti.iter().map(|&i| enc[i].as_slice()).collect();
            self.embed_batch(&ids, toplu)
        })?;
        let mut out = vec![Vec::new(); texts.len()];
        for (parti, vs) in partiler.iter().zip(sonuc) {
            for (&i, v) in parti.iter().zip(vs) {
                out[i] = v;
            }
        }
        let profil = Profil {
            token: enc.iter().map(Vec::len).sum(),
            parti: partiler.len(),
            tok_ms,
            ileri_ms: t0.elapsed().as_millis() as u64 - tok_ms,
        };
        Ok((out, profil))
    }
}

/// `BGE_PROFIL=1`: her istek için zaman damgalı bir satır (UTC) basılır.
struct Profil {
    token: usize,
    parti: usize,
    tok_ms: u64,
    ileri_ms: u64,
}

static PROFIL_ACIK: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("BGE_PROFIL").is_ok_and(|v| v == "1"));

fn profil_yaz(girdi: usize, p: &Profil, toplam_ms: u64, serit: &str) {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() % 86_400_000);
    println!(
        "[{:02}:{:02}:{:02}.{:03} UTC] istek: {girdi} girdi, {} token, {} parti, {serit}; \
         tokenizer {} ms, ileri geçiş {} ms, toplam {toplam_ms} ms",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000,
        p.token,
        p.parti,
        p.tok_ms,
        p.ileri_ms
    );
}

/// Parti başına doldurulmuş token üst sınırı (parti × en uzun girdi). Büyük parti gemm'i
/// verimli yapar; sınır ara aktivasyonları (token × 4096 × 4 bayt) sınırlı tutar.
fn parti_token_siniri() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("BGE_PARTI_TOKEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n >= 1)
            .unwrap_or(2048)
    });
    *N
}

/// Uzunluklara göre indeks partileri: uzunluğa göre sıralanmış girdiler, doldurulmuş boyut
/// (parti × en uzun) `parti_token_siniri`'nı aşmayacak kadar art arda toplanır.
/// Tek girdi sınırı aşsa da kendi başına bir parti olur.
fn partile(uzunluklar: &[usize]) -> Vec<Vec<usize>> {
    let mut sira: Vec<usize> = (0..uzunluklar.len()).collect();
    sira.sort_by_key(|&i| uzunluklar[i]);
    let mut partiler: Vec<Vec<usize>> = Vec::new();
    for i in sira {
        match partiler.last_mut() {
            // sıralı gidildiğinden i, partinin en uzunu olur
            Some(p) if (p.len() + 1) * uzunluklar[i] <= parti_token_siniri() => p.push(i),
            _ => partiler.push(vec![i]),
        }
    }
    partiler
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

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?)?)
}

/// Embeds inputs on at most `parallel` threads; output keeps input order.
/// On the first error the other workers stop taking new inputs and the error is returned.
fn embed_parallel<T: Sync, R: Send, F>(inputs: &[T], parallel: usize, f: F) -> Result<Vec<R>>
where
    F: Fn(&T) -> Result<R> + Sync,
{
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let workers = parallel.min(inputs.len()).max(1);
    // Tek iş çağıranın iş parçacığında: hızlı şeridin havuzu (`install`) ayrı bir std iş
    // parçacığına geçince kaybolurdu (oradaki par_iter genel havuza düşer).
    if workers == 1 {
        return inputs.iter().map(&f).collect();
    }
    let parts: Vec<Result<Vec<(usize, R)>>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| -> Result<Vec<(usize, R)>> {
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
    model: Arc<Yuva>,
    status: Arc<Status>,
}

/// Sistem belleği bu yüzdeyi aşınca boştaki model bellekten atılır (f32 ağırlıklar ~2,3 GB).
/// `BGE_BELLEK_ESIGI` (1-100) ile değişir; 100 yalnız bellek tümüyle dolunca boşaltır.
fn bellek_esigi() -> u32 {
    static N: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("BGE_BELLEK_ESIGI")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| (1..=100).contains(n))
            .unwrap_or(90)
    });
    *N
}
/// Son istekten bu kadar sonra "boşta" sayılır.
const BOSTA: std::time::Duration = std::time::Duration::from_secs(30);

/// Model yuvası: bellek darken boşaltılır, ilk istekte yeniden yüklenir.
struct Yuva {
    model: std::sync::Mutex<Option<Arc<EmbedModel>>>,
    yukleyici: Arc<tokio::sync::Mutex<()>>,
    son_istek: std::sync::Mutex<std::time::Instant>,
    status: Arc<Status>,
}

impl Yuva {
    fn hazir(&self) -> Option<Arc<EmbedModel>> {
        self.model.lock().unwrap().clone()
    }

    async fn al(self: &Arc<Self>) -> Result<Arc<EmbedModel>> {
        *self.son_istek.lock().unwrap() = std::time::Instant::now();
        if let Some(m) = self.hazir() {
            return Ok(m);
        }
        let kilit = self.yukleyici.clone().lock_owned().await;
        if let Some(m) = self.hazir() {
            return Ok(m);
        }
        // Ayrı görevde: istemci vazgeçse de yükleme biter (kilit de onunla), ikinci yükleme olmaz.
        let ben = self.clone();
        tokio::spawn(async move {
            let _kilit = kilit;
            let evre = ben.status.phase.lock().unwrap().clone();
            let m = Arc::new(EmbedModel::load(ben.status.clone()).await?);
            if let Some(e) = evre {
                ben.status.set_phase(e);
            }
            println!("model yeniden yüklendi");
            *ben.model.lock().unwrap() = Some(m.clone());
            Ok(m)
        })
        .await?
    }

    /// Bellek dar ve model boştaysa (hiçbir istek tutmuyor) bırakır; bıraktıysa yükü döner.
    fn bosalt_gerekirse(&self) -> Option<u32> {
        let yuk = bellek_yuku().filter(|&y| y >= bellek_esigi())?;
        if self.son_istek.lock().unwrap().elapsed() < BOSTA {
            return None;
        }
        let mut m = self.model.lock().unwrap();
        // strong_count == 1: yalnız yuva tutuyor, uçuşta istek yok.
        m.take_if(|m| Arc::strong_count(m) == 1).map(|_| yuk)
    }
}

/// Sistem belleği yükü (%). Windows dışında ölçülmez: model hiç boşaltılmaz.
fn bellek_yuku() -> Option<u32> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: MEMORYSTATUSEX düz veri; sıfır geçerli, dwLength işlevin istediği gibi yazılır.
        let mut d: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        d.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: `d` bu çerçevede yaşayan, boyutu yazılmış yerel yapı.
        (unsafe { GlobalMemoryStatusEx(&mut d) } != 0).then_some(d.dwMemoryLoad)
    }
    #[cfg(not(windows))]
    None
}

async fn embeddings_handler(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, String)> {
    let inputs = req.input.into_vec();
    let text_count = inputs.len();
    let started = std::time::Instant::now();
    let model = state
        .model
        .al()
        .await
        .map_err(|e| (StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;
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
    // Girdiler uzunluğa göre partilenir (en fazla `BGE_PARALLEL` parti aynı anda); her parti
    // tek ileri geçiştir ve rayon ile tüm çekirdeklere yayılır (bkz. hizli.rs).
    let serit = if _izin.is_none() {
        "hızlı şerit"
    } else {
        "toplu şerit"
    };
    let hizli = _izin.is_none();
    let _suren = hizli.then(HizliSuren::yeni);
    let (result, profil) = tokio::task::spawn_blocking(move || match hizli {
        true => HIZLI_HAVUZ.install(|| model.embed_many(&inputs, false)),
        false => model.embed_many(&inputs, true),
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let toplam_ms = started.elapsed().as_millis() as u64;
    state.status.record_request(text_count, toplam_ms);
    if *PROFIL_ACIK {
        profil_yaz(text_count, &profil, toplam_ms, serit);
    }

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

/// Hızlı şerit kapıyı atlamakla kalmaz, üç şeyle önce geçer (ölçüm 2026-10-05, 32×800
/// belirteçlik toplu yük döngüsü altında 7 belirteçlik sorgu):
/// - kendi rayon havuzu: genel havuzda sorgunun görevleri, toplu işin o an süren katmanının
///   kuyruğa koyduğu görevlerin arkasında bekliyordu (katman ~2,7 sn);
/// - havuz iş parçacıkları bir kademe yüksek öncelikli: süren katmanla çakışırken CPU payı;
/// - toplu iş katman aralarında bekler (`hizliya_yol_ver`): sorgu da bütün ağırlıkları
///   (2,3 GB) bellekten okur, bellek bant genişliğini öncelik paylaştırmaz.
///
/// Eski şerit 3-12 sn; yalnız bekleme 1,8-3 sn; bekleme + havuz 0,8-2,8 sn; üçü 0,12-0,32 sn.
static HIZLI_HAVUZ: std::sync::LazyLock<rayon::ThreadPool> = std::sync::LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("hizli-serit-{i}"))
        .start_handler(|_| oncelik_yukselt())
        .build()
        .expect("hızlı şerit havuzu kurulamadı")
});

/// Bu iş parçacığını süreç önceliğinin bir kademe üstüne alır (Windows; başka yerde etkisiz).
fn oncelik_yukselt() {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
        };
        // SAFETY: GetCurrentThread sözde tutamaç döndürür (kapatılmaz); yalnız bu iş parçacığı etkilenir.
        if unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) } == 0 {
            eprintln!("hızlı şerit önceliği yükseltilemedi");
        }
    }
}

/// Uçuştaki hızlı şerit istekleri; toplu iş katman aralarında bunlar bitene dek bekler.
static HIZLI_SUREN: AtomicUsize = AtomicUsize::new(0);

struct HizliSuren;

impl HizliSuren {
    fn yeni() -> Self {
        HIZLI_SUREN.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for HizliSuren {
    fn drop(&mut self) {
        HIZLI_SUREN.fetch_sub(1, Ordering::SeqCst);
    }
}

fn hizliya_yol_ver() {
    while HIZLI_SUREN.load(Ordering::SeqCst) > 0 {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

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
    let model = Arc::new(Yuva {
        model: std::sync::Mutex::new(Some(Arc::new(EmbedModel::load(status.clone()).await?))),
        yukleyici: Arc::default(),
        son_istek: std::sync::Mutex::new(std::time::Instant::now()),
        status: status.clone(),
    });
    let izleyici = model.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            if let Some(yuk) = izleyici.bosalt_gerekirse() {
                println!("bellek %{yuk}: model bellekten atıldı, ilk istekte yüklenecek");
            }
        }
    });

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
    println!("{} listening on {url}", bge_settings::APP_ID);
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

    // Before anything reads settings or the theme: carry over what was saved
    // under the old name (bge-embed-rs). Harmless once done.
    for line in bge_settings::migrate_old_name() {
        eprintln!("{line}");
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
            .with_title(bge_settings::DISPLAY_NAME)
            .with_inner_size([480.0, 560.0])
            .with_min_inner_size([480.0, 560.0])
            .with_transparent(false),
        ..Default::default()
    };
    eframe::run_native(
        bge_settings::APP_ID,
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
    fn partile_siraya_gore_gruplar_ve_siniri_asmaz() {
        let uz = [900usize, 5, 7, 1500, 6, 1000, 2500];
        let p = partile(&uz);
        let mut hepsi: Vec<usize> = p.iter().flatten().copied().collect();
        hepsi.sort();
        assert_eq!(
            hepsi,
            (0..uz.len()).collect::<Vec<_>>(),
            "her girdi bir kez"
        );
        for parti in &p {
            let en_uzun = parti.iter().map(|&i| uz[i]).max().unwrap();
            assert!(
                parti.len() == 1 || parti.len() * en_uzun <= parti_token_siniri(),
                "dolgulu boyut sınırı aşıldı: {parti:?}"
            );
        }
        assert!(partile(&[]).is_empty());
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
            embed_parallel(&[] as &[String], 4, |_| Ok(vec![1.0]))
                .unwrap()
                .is_empty()
        );
    }
}
