//! Connector layer for the Connections screen (design/DESIGN.md).
//!
//! Three tools talk over HTTP (Open Notebook, Open WebUI, AnythingLLM), each
//! as its own plain struct with `detect`/`status`/`connect` methods - same
//! shape as before, just three of them now. `Tool` is the promised thin
//! match-based dispatcher over the three, for GUI code that wants to loop
//! over "all HTTP-probed tools" without repeating itself three times.
//! LibreChat/Dify below are still "manual setup" only (no network calls), so
//! they don't need the same shape and aren't part of `Tool`.
//!
//! HTTP client: `ureq` (blocking, rustls). It was already resolved
//! transitively via hf-hub's sync API (see Cargo.lock before this change),
//! so declaring it directly adds no new crate to the dependency tree - just
//! a Cargo.toml entry. Blocking + a plain `std::thread` is a much smaller
//! surface than wiring reqwest/tokio (already used by the embedding server)
//! into a fire-and-forget GUI-thread probe.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

/// Where the Connections screen's "Detected on this computer" row lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Connected,
    Found,
    NeedsKey,
    NotFound,
}

/// `http://127.0.0.1:{port}/v1` by default, or `http://host.docker.internal:{port}/v1`
/// when the row's "Runs in Docker" checkbox is on (DESIGN.md Connections screen).
pub fn embedding_url(port: &str, docker: bool) -> String {
    let host = if docker { "host.docker.internal" } else { "127.0.0.1" };
    format!("http://{host}:{port}/v1")
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new().timeout(timeout).build()
}

/// The three HTTP-probed connectors, for GUI code that wants to loop over
/// "all detectable tools" instead of repeating the same row logic three
/// times. Each variant just forwards to that tool's own struct - see the
/// `OpenNotebook`/`OpenWebUi`/`AnythingLlm` impls below for the actual logic
/// and source citations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    OpenNotebook,
    OpenWebUi,
    AnythingLlm,
}

impl Tool {
    pub const ALL: [Tool; 3] = [Tool::OpenNotebook, Tool::OpenWebUi, Tool::AnythingLlm];

    pub fn name(self) -> &'static str {
        match self {
            Tool::OpenNotebook => OpenNotebook::NAME,
            Tool::OpenWebUi => OpenWebUi::NAME,
            Tool::AnythingLlm => AnythingLlm::NAME,
        }
    }

    pub fn initials(self) -> &'static str {
        match self {
            Tool::OpenNotebook => OpenNotebook::INITIALS,
            Tool::OpenWebUi => OpenWebUi::INITIALS,
            Tool::AnythingLlm => AnythingLlm::INITIALS,
        }
    }

    /// Base URL(s) to probe, in order - most tools have one, Open WebUI has
    /// two common default ports.
    pub fn candidate_base_urls(self) -> &'static [&'static str] {
        match self {
            Tool::OpenNotebook => &[OpenNotebook::DEFAULT_BASE_URL],
            Tool::OpenWebUi => &[OpenWebUi::DEFAULT_BASE_URL, OpenWebUi::ALT_BASE_URL],
            Tool::AnythingLlm => &[AnythingLlm::DEFAULT_BASE_URL],
        }
    }

    /// Label for the password/API-key field on this tool's row.
    pub fn key_label(self) -> &'static str {
        match self {
            Tool::OpenNotebook => "Admin password",
            Tool::OpenWebUi => "Admin API key",
            Tool::AnythingLlm => "API key",
        }
    }

    /// "Needs a ..." meta-line phrasing (DESIGN.md: "Needs an admin API key").
    pub fn needs_key_meta(self) -> &'static str {
        match self {
            Tool::OpenNotebook => "Needs a password",
            Tool::OpenWebUi => "Needs an admin API key",
            Tool::AnythingLlm => "Needs an API key",
        }
    }

    /// Caption shown once `connect()` has actually changed the tool's
    /// embedding engine/model in this session (gated the same way as
    /// `ConnState::default_changed` in gui.rs).
    pub fn changed_caption(self) -> &'static str {
        match self {
            Tool::OpenNotebook => {
                "Default embedding model changed. Existing sources may need re-embedding for consistent search."
            }
            Tool::OpenWebUi => "Embedding model changed. Re-index your knowledge bases in Open WebUI.",
            Tool::AnythingLlm => "Embedding model changed. Existing workspaces need to be re-embedded.",
        }
    }

    pub fn detect(self, base_url: &str) -> bool {
        match self {
            Tool::OpenNotebook => OpenNotebook::detect(base_url),
            Tool::OpenWebUi => OpenWebUi::detect(base_url),
            Tool::AnythingLlm => AnythingLlm::detect(base_url),
        }
    }

    pub fn status(self, base_url: &str, embed_url: &str, key: Option<&str>) -> Status {
        match self {
            Tool::OpenNotebook => OpenNotebook::status(base_url, embed_url, key),
            Tool::OpenWebUi => OpenWebUi::status(base_url, embed_url, key),
            Tool::AnythingLlm => AnythingLlm::status(base_url, embed_url, key),
        }
    }

    /// `confirm_reset` only matters for `Tool::AnythingLlm` (see
    /// `ConnectOutcome`/`AnythingLlm::plan` below) - Open Notebook and Open
    /// WebUI's `connect()` never delete anything, so it's ignored for them.
    pub fn connect(self, base_url: &str, embed_url: &str, key: Option<&str>, confirm_reset: bool) -> anyhow::Result<ConnectOutcome> {
        match self {
            Tool::OpenNotebook => OpenNotebook::connect(base_url, embed_url, key).map(ConnectOutcome::Done),
            Tool::OpenWebUi => OpenWebUi::connect(base_url, embed_url, key).map(ConnectOutcome::Done),
            Tool::AnythingLlm => AnythingLlm::connect(base_url, embed_url, key, confirm_reset),
        }
    }
}

/// Result of a `Tool::connect` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectOutcome {
    /// Connected (or already was) - `bool` is whether this call actually
    /// wrote anything.
    Done(bool),
    /// AnythingLLM only: applying the change would delete every workspace's
    /// embedded documents (see `AnythingLlm::plan`) and the caller didn't
    /// pass `confirm_reset: true` - nothing was written. Re-call `connect`
    /// with `confirm_reset: true` to proceed.
    NeedsResetConfirmation,
}

// ---------------------------------------------------------------------------
// Open Notebook
//
// Source: C:\Users\buzbe\Desktop\claude yazılım\projeler\ZOPAY çalışma odası\open-notebook
// - Fingerprint:        GET  /                      -> {"message": "Open Notebook API is running"}  (api/main.py:409-411)
// - Auth:                    Authorization: Bearer <password>, 401 when missing/wrong               (api/middleware.py:12,25-31,49-78)
// - List credentials:   GET  /api/credentials?provider=openai_compatible                             (api/routers/credentials.py:115-139)
// - Create credential:  POST /api/credentials {name, provider, base_url}                              (api/routers/credentials.py:161-207, api/models.py:676-709)
// - List models:        GET  /api/models?type=embedding                                               (api/routers/models.py:173-202)
// - Create model:       POST /api/models {name, provider, type, credential}                            (api/routers/models.py:205-259, api/models.py:97-108)
// - Read defaults:      GET  /api/models/defaults                                                      (api/routers/models.py:307-330)
// - Write defaults:     PUT  /api/models/defaults {default_embedding_model}                            (api/routers/models.py:339-384, api/models.py:127)
// ---------------------------------------------------------------------------

pub struct OpenNotebook;

impl OpenNotebook {
    pub const NAME: &'static str = "Open Notebook";
    pub const INITIALS: &'static str = "ON";
    pub const DEFAULT_BASE_URL: &'static str = "http://127.0.0.1:5055";
    /// The name we register/look up the embedding model under.
    const MODEL_NAME: &'static str = "bge-m3";
    const PROVIDER: &'static str = "openai_compatible";
    /// The credential/model name we create - distinguishes ours from any
    /// other openai_compatible credential the user already has.
    const CREDENTIAL_NAME: &'static str = "bge-embed-rs";
    /// Open Notebook's provider client may send a bearer token at embed time;
    /// our server ignores auth entirely, but a placeholder is safer than
    /// leaving `api_key` unset (api/models.py:656-673 only *requires* it for
    /// anthropic_compatible, but a working credential normally has one).
    const API_KEY_PLACEHOLDER: &'static str = "sk-local";

    /// Cheap "is Open Notebook running here" check: a bare port scan would
    /// match anything, so this also checks the identifying `message` field.
    pub fn detect(base_url: &str) -> bool {
        let agent = agent(Duration::from_millis(1500));
        match agent.get(&format!("{base_url}/")).call() {
            Ok(resp) => resp
                .into_json::<RootResponse>()
                .map(|r| r.message == "Open Notebook API is running")
                .unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Connected when the default embedding model's *credential* points at
    /// our `embed_url` - the model can be named anything (see
    /// `find_or_create_model`: a pre-existing `bge-m3` bound to someone
    /// else's server must not be mistaken for ours just because of its name).
    /// Found when the server answers but isn't wired to us; NeedsKey when a
    /// password is required (missing or wrong); NotFound when nothing
    /// answered `detect()`.
    pub fn status(base_url: &str, embed_url: &str, password: Option<&str>) -> Status {
        if !Self::detect(base_url) {
            return Status::NotFound;
        }
        let agent = agent(Duration::from_millis(1500));
        let defaults = match get_json::<DefaultModels>(&agent, base_url, "/api/models/defaults", password) {
            Ok(d) => d,
            Err(e) if matches!(e.downcast_ref::<ApiError>(), Some(ApiError::Unauthorized)) => return Status::NeedsKey,
            Err(_) => return Status::Found,
        };
        let Some(model_id) = defaults.default_embedding_model else {
            return Status::Found;
        };
        let models = match get_json::<Vec<ModelSummary>>(&agent, base_url, "/api/models?type=embedding", password) {
            Ok(m) => m,
            Err(_) => return Status::Found,
        };
        let Some(model) = models.into_iter().find(|m| m.id == model_id) else {
            return Status::Found;
        };
        let Some(cred_id) = model.credential else {
            return Status::Found;
        };
        let creds = match get_json::<Vec<CredentialSummary>>(&agent, base_url, "/api/credentials", password) {
            Ok(c) => c,
            Err(_) => return Status::Found,
        };
        match creds.into_iter().find(|c| c.id == cred_id) {
            Some(c) if c.base_url.as_deref() == Some(embed_url) => Status::Connected,
            _ => Status::Found,
        }
    }

    /// Idempotent and non-destructive: reuses an existing credential with the
    /// same `base_url` and an existing embedding model already bound to
    /// *that credential* instead of creating duplicates, then points
    /// `default_embedding_model` at it. Never updates or deletes an existing
    /// credential or model - Open Notebook has no update endpoint for either,
    /// and DELETE is destructive to data we don't own.
    ///
    /// Returns `Ok(true)` iff this call actually changed
    /// `default_embedding_model` (i.e. it pointed somewhere else before).
    /// IMPORTANT (per DESIGN.md caption): switching the default embedding
    /// model means existing Open Notebook sources must be re-embedded for
    /// search to stay consistent - this function only flips the pointer, it
    /// never triggers or performs re-embedding of anything.
    pub fn connect(base_url: &str, embed_url: &str, password: Option<&str>) -> anyhow::Result<bool> {
        let agent = agent(Duration::from_secs(5));
        let cred_id = Self::find_or_create_credential(&agent, base_url, embed_url, password)?;
        let model_id = Self::find_or_create_model(&agent, base_url, &cred_id, embed_url, password)?;

        let defaults = get_json::<DefaultModels>(&agent, base_url, "/api/models/defaults", password)?;
        let already_default = defaults.default_embedding_model.as_deref() == Some(model_id.as_str());
        if !already_default {
            put_json(
                &agent,
                base_url,
                "/api/models/defaults",
                password,
                &DefaultsPatch { default_embedding_model: &model_id },
            )?;
        }
        Ok(!already_default)
    }

    fn find_or_create_credential(
        agent: &ureq::Agent,
        base_url: &str,
        embed_url: &str,
        password: Option<&str>,
    ) -> anyhow::Result<String> {
        let existing = get_json::<Vec<CredentialSummary>>(
            agent,
            base_url,
            "/api/credentials?provider=openai_compatible",
            password,
        )?;
        if let Some(c) = existing.into_iter().find(|c| c.base_url.as_deref() == Some(embed_url)) {
            return Ok(c.id);
        }
        let created = post_json::<_, CredentialSummary>(
            agent,
            base_url,
            "/api/credentials",
            password,
            &CreateCredentialReq {
                name: Self::CREDENTIAL_NAME,
                provider: Self::PROVIDER,
                base_url: embed_url,
                modalities: &["embedding"],
                api_key: Self::API_KEY_PLACEHOLDER,
            },
        )?;
        Ok(created.id)
    }

    /// Finds an embedding model to use, or creates one - never touches an
    /// existing model or credential that belongs to someone else.
    ///
    /// 1. Any embedding model already bound to *our* credential wins outright
    ///    (whatever it's named) - that's the model a previous `connect()`
    ///    created, or one the user already pointed at us by hand.
    /// 2. Otherwise create one named `bge-m3`, unless that (provider, name)
    ///    is already taken by a model on a *different* credential (e.g. the
    ///    user's existing bge-m3 wired to some other server on another port)
    ///    - Open Notebook has no rename/update endpoint and DELETE would
    ///    destroy someone else's working setup, so we pick a distinct name
    ///    instead. The model field in `/v1/embeddings` requests is ignored by
    ///    this server, so the name is free-form.
    fn find_or_create_model(
        agent: &ureq::Agent,
        base_url: &str,
        credential_id: &str,
        embed_url: &str,
        password: Option<&str>,
    ) -> anyhow::Result<String> {
        let list = |agent: &ureq::Agent| -> anyhow::Result<Vec<ModelSummary>> {
            get_json(agent, base_url, "/api/models?type=embedding", password)
        };
        let ours = |models: &[ModelSummary]| {
            models
                .iter()
                .find(|m| m.credential.as_deref() == Some(credential_id))
                .map(|m| m.id.clone())
        };

        let models = list(agent)?;
        if let Some(id) = ours(&models) {
            return Ok(id);
        }
        let name_taken = models
            .iter()
            .any(|m| m.name.eq_ignore_ascii_case(Self::MODEL_NAME) && m.provider == Self::PROVIDER);
        let name = if name_taken {
            format!("{} (bge-embed-rs :{})", Self::MODEL_NAME, port_of(embed_url))
        } else {
            Self::MODEL_NAME.to_string()
        };

        match post_json::<_, ModelSummary>(
            agent,
            base_url,
            "/api/models",
            password,
            &CreateModelReq {
                name: &name,
                provider: Self::PROVIDER,
                r#type: "embedding",
                credential: credential_id,
            },
        ) {
            Ok(created) => Ok(created.id),
            // The server 400s on a duplicate (provider, name, type) - a
            // concurrent creator raced us to the same name, so re-check
            // whether *our* credential now has a model before giving up.
            Err(e) if matches!(e.downcast_ref::<ApiError>(), Some(ApiError::Http(400))) => {
                ours(&list(agent)?).ok_or(e)
            }
            Err(e) => Err(e),
        }
    }
}

/// `"http://host:port/v1"` -> `"port"`, for the disambiguated model name.
fn port_of(embed_url: &str) -> &str {
    embed_url.rsplit(':').next().unwrap_or("").trim_end_matches("/v1")
}

#[derive(Deserialize, Default)]
struct RootResponse {
    #[serde(default)]
    message: String,
}

#[derive(Deserialize)]
struct CredentialSummary {
    id: String,
    #[serde(default)]
    base_url: Option<String>,
}

#[derive(Deserialize)]
struct ModelSummary {
    id: String,
    name: String,
    provider: String,
    #[serde(default)]
    credential: Option<String>,
}

#[derive(Deserialize, Default)]
struct DefaultModels {
    #[serde(default)]
    default_embedding_model: Option<String>,
}

#[derive(Serialize)]
struct CreateCredentialReq<'a> {
    name: &'a str,
    provider: &'a str,
    base_url: &'a str,
    modalities: &'a [&'a str],
    api_key: &'a str,
}

#[derive(Serialize)]
struct CreateModelReq<'a> {
    name: &'a str,
    provider: &'a str,
    r#type: &'a str,
    credential: &'a str,
}

#[derive(Serialize)]
struct DefaultsPatch<'a> {
    default_embedding_model: &'a str,
}

// ponytail: no `thiserror` dependency for a 2-variant enum - Display/Error by hand.
#[derive(Debug)]
enum ApiError {
    Unauthorized,
    Http(u16),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Unauthorized => write!(f, "unauthorized"),
            ApiError::Http(code) => write!(f, "http {code}"),
        }
    }
}
impl std::error::Error for ApiError {}

fn auth(req: ureq::Request, password: Option<&str>) -> ureq::Request {
    match password {
        Some(pw) => req.set("Authorization", &format!("Bearer {pw}")),
        None => req,
    }
}

fn map_status(err: ureq::Error) -> anyhow::Error {
    match err {
        ureq::Error::Status(401, _) => ApiError::Unauthorized.into(),
        ureq::Error::Status(code, _) => ApiError::Http(code).into(),
        ureq::Error::Transport(t) => anyhow::anyhow!("network error: {t}"),
    }
}

fn get_json<T: serde::de::DeserializeOwned>(
    agent: &ureq::Agent,
    base_url: &str,
    path: &str,
    password: Option<&str>,
) -> anyhow::Result<T> {
    let resp = auth(agent.get(&format!("{base_url}{path}")), password)
        .call()
        .map_err(map_status)?;
    Ok(resp.into_json::<T>()?)
}

fn post_json<B: Serialize, T: serde::de::DeserializeOwned>(
    agent: &ureq::Agent,
    base_url: &str,
    path: &str,
    password: Option<&str>,
    body: &B,
) -> anyhow::Result<T> {
    let resp = auth(agent.post(&format!("{base_url}{path}")), password)
        .send_json(serde_json::to_value(body)?)
        .map_err(map_status)?;
    Ok(resp.into_json::<T>()?)
}

fn put_json<B: Serialize>(
    agent: &ureq::Agent,
    base_url: &str,
    path: &str,
    password: Option<&str>,
    body: &B,
) -> anyhow::Result<Value> {
    let resp = auth(agent.put(&format!("{base_url}{path}")), password)
        .send_json(serde_json::to_value(body)?)
        .map_err(map_status)?;
    Ok(resp.into_json::<Value>().unwrap_or(Value::Null))
}

// ---------------------------------------------------------------------------
// Open WebUI
//
// Source: github.com/open-webui/open-webui @ 8bd8b4fac5e059578ac0c74b3c18d11139f88b7d (main, 2026-09-21)
// - Mount:              app.include_router(retrieval.router, prefix="/api/v1/retrieval", ...)   (backend/open_webui/main.py:852)
// - Fingerprint:        GET  /api/config -> {"status": true, "name": ..., "default_locale": ..., "features": {...}}, unauthenticated (backend/open_webui/main.py:2224-2225,2312-2316)
// - Auth:                    Authorization: Bearer <admin API key or JWT>, admin-only routes 401 when missing/wrong/non-admin (backend/open_webui/utils/auth.py:383-388 get_http_authorization_cred, :625-630 get_admin_user)
// - Read config:        GET  /api/v1/retrieval/embedding                                        (backend/open_webui/routers/retrieval.py:468-489)
// - Write config:       POST /api/v1/retrieval/embedding/update {RAG_EMBEDDING_ENGINE, RAG_EMBEDDING_MODEL, RAG_EMBEDDING_BATCH_SIZE, ENABLE_ASYNC_EMBEDDING, RAG_EMBEDDING_CONCURRENT_REQUESTS, openai_config:{url,key}} (backend/open_webui/routers/retrieval.py:507-621)
// ---------------------------------------------------------------------------

pub struct OpenWebUi;

impl OpenWebUi {
    pub const NAME: &'static str = "Open WebUI";
    pub const INITIALS: &'static str = "OW";
    pub const DEFAULT_BASE_URL: &'static str = "http://127.0.0.1:8080";
    /// Open WebUI's other very common default port (e.g. the `docker run`
    /// one-liner in its README maps container :8080 to host :3000).
    pub const ALT_BASE_URL: &'static str = "http://127.0.0.1:3000";
    const MODEL_NAME: &'static str = "bge-m3";
    /// `RAG_EMBEDDING_ENGINE` value for "OpenAI-compatible" (retrieval.py:551-553).
    const ENGINE: &'static str = "openai";
    /// Open WebUI's OpenAI-compatible embedding client sends this as a bearer
    /// token; our server ignores auth entirely, but an empty string is a
    /// worse default than an obvious placeholder.
    const API_KEY_PLACEHOLDER: &'static str = "sk-local";

    /// `GET /api/config` is public (no auth needed to view login-page
    /// config) and returns a handful of Open-WebUI-specific keys together -
    /// a bare port scan can't fake `status: true` + `default_locale` set.
    pub fn detect(base_url: &str) -> bool {
        let agent = agent(Duration::from_millis(1500));
        match agent.get(&format!("{base_url}/api/config")).call() {
            Ok(resp) => resp
                .into_json::<OwuiConfigFingerprint>()
                .map(|c| c.status && c.default_locale.is_some())
                .unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Connected when the engine is `openai` and its `openai_config.url`
    /// already points at our `embed_url` - mirrors `OpenNotebook::status`'s
    /// "credential base_url is the source of truth" approach. NeedsKey on a
    /// 401 from the admin-only read; Found if the server answers but isn't
    /// wired to us (or the read fails for any other reason).
    pub fn status(base_url: &str, embed_url: &str, key: Option<&str>) -> Status {
        if !Self::detect(base_url) {
            return Status::NotFound;
        }
        let agent = agent(Duration::from_millis(1500));
        let cfg = match get_json::<OwuiEmbeddingConfig>(&agent, base_url, "/api/v1/retrieval/embedding", key) {
            Ok(c) => c,
            Err(e) if matches!(e.downcast_ref::<ApiError>(), Some(ApiError::Unauthorized)) => return Status::NeedsKey,
            Err(_) => return Status::Found,
        };
        if cfg.rag_embedding_engine == Self::ENGINE
            && cfg.openai_config.as_ref().and_then(|o| o.url.as_deref()) == Some(embed_url)
        {
            Status::Connected
        } else {
            Status::Found
        }
    }

    /// Idempotent: a no-op (returns `Ok(false)`) when the engine/model/url
    /// already match. Otherwise flips the engine to `openai` pointed at
    /// `embed_url` with model `bge-m3`, preserving whatever batch-size/async
    /// settings were already configured (Open WebUI has no partial-update
    /// endpoint - the POST replaces the whole form, so unrelated fields are
    /// read back first and forwarded unchanged rather than reset to their
    /// pydantic defaults).
    ///
    /// Per DESIGN.md: switching the embedding engine means existing
    /// knowledge bases need re-indexing in Open WebUI - this only flips the
    /// config, it never triggers or performs that reindex itself.
    pub fn connect(base_url: &str, embed_url: &str, key: Option<&str>) -> anyhow::Result<bool> {
        let agent = agent(Duration::from_secs(5));
        let cfg = get_json::<OwuiEmbeddingConfig>(&agent, base_url, "/api/v1/retrieval/embedding", key)?;
        let already = cfg.rag_embedding_engine == Self::ENGINE
            && cfg.rag_embedding_model == Self::MODEL_NAME
            && cfg.openai_config.as_ref().and_then(|o| o.url.as_deref()) == Some(embed_url);
        if already {
            return Ok(false);
        }
        post_json::<_, Value>(
            &agent,
            base_url,
            "/api/v1/retrieval/embedding/update",
            key,
            &OwuiUpdateEmbeddingReq {
                openai_config: OwuiOpenAiConfig { url: embed_url, key: Self::API_KEY_PLACEHOLDER },
                rag_embedding_engine: Self::ENGINE,
                rag_embedding_model: Self::MODEL_NAME,
                rag_embedding_batch_size: cfg.rag_embedding_batch_size,
                enable_async_embedding: cfg.enable_async_embedding,
                rag_embedding_concurrent_requests: cfg.rag_embedding_concurrent_requests,
            },
        )?;
        Ok(true)
    }
}

#[derive(Deserialize, Default)]
struct OwuiConfigFingerprint {
    #[serde(default)]
    status: bool,
    #[serde(default)]
    default_locale: Option<Value>,
}

#[derive(Deserialize, Default)]
struct OwuiOpenAiConfigResp {
    #[serde(default)]
    url: Option<String>,
}

#[derive(Deserialize)]
struct OwuiEmbeddingConfig {
    #[serde(rename = "RAG_EMBEDDING_ENGINE")]
    rag_embedding_engine: String,
    #[serde(rename = "RAG_EMBEDDING_MODEL")]
    rag_embedding_model: String,
    #[serde(rename = "RAG_EMBEDDING_BATCH_SIZE", default)]
    rag_embedding_batch_size: Option<i64>,
    #[serde(rename = "ENABLE_ASYNC_EMBEDDING", default)]
    enable_async_embedding: Option<bool>,
    #[serde(rename = "RAG_EMBEDDING_CONCURRENT_REQUESTS", default)]
    rag_embedding_concurrent_requests: Option<i64>,
    #[serde(default)]
    openai_config: Option<OwuiOpenAiConfigResp>,
}

#[derive(Serialize)]
struct OwuiOpenAiConfig<'a> {
    url: &'a str,
    key: &'a str,
}

#[derive(Serialize)]
struct OwuiUpdateEmbeddingReq<'a> {
    openai_config: OwuiOpenAiConfig<'a>,
    #[serde(rename = "RAG_EMBEDDING_ENGINE")]
    rag_embedding_engine: &'a str,
    #[serde(rename = "RAG_EMBEDDING_MODEL")]
    rag_embedding_model: &'a str,
    #[serde(rename = "RAG_EMBEDDING_BATCH_SIZE")]
    rag_embedding_batch_size: Option<i64>,
    #[serde(rename = "ENABLE_ASYNC_EMBEDDING")]
    enable_async_embedding: Option<bool>,
    #[serde(rename = "RAG_EMBEDDING_CONCURRENT_REQUESTS")]
    rag_embedding_concurrent_requests: Option<i64>,
}

// ---------------------------------------------------------------------------
// AnythingLLM
//
// Source: github.com/Mintplex-Labs/anything-llm @ master (2026-09-25)
// - Mount:              app.use("/api", apiRouter); developerEndpoints(app, apiRouter) -> apiSystemEndpoints(apiRouter) (server/index.js:51,81,97; server/endpoints/api/index.js:5,20)
// - Fingerprint:        GET  /api/ping -> {"online": true}, unauthenticated                      (server/endpoints/system.js:83-85, mounted at /api via server/index.js:13,82)
// - Auth:                    Authorization: Bearer <developer API key>; 403 {"error":"No valid api key found."} when missing/invalid (server/utils/middleware/validApiKey.js:4-24)
// - Read settings:      GET  /api/v1/system -> {"settings": {EmbeddingEngine, EmbeddingBasePath, EmbeddingModelPref, EmbeddingModelMaxChunkLength, ...}} (server/endpoints/api/system/index.js:37-70; server/models/systemSettings.js:454,480-491)
// - Write settings:     POST /api/v1/system/update-env {<Key>: <value>, ...} -> {newValues, error} (server/endpoints/api/system/index.js:106-149)
// - Setting keys (-> env var): EmbeddingEngine->EMBEDDING_ENGINE, EmbeddingBasePath->EMBEDDING_BASE_PATH,
//   EmbeddingModelPref->EMBEDDING_MODEL_PREF, EmbeddingModelMaxChunkLength->EMBEDDING_MODEL_MAX_CHUNK_LENGTH,
//   GenericOpenAiEmbeddingApiKey->GENERIC_OPEN_AI_EMBEDDING_API_KEY               (server/utils/helpers/updateENV.js:247-282)
// - Engine value:       "generic-openai" is a supported embedding engine        (server/utils/helpers/updateENV.js:1225-1240)
//   and that engine's client reads EMBEDDING_BASE_PATH / EMBEDDING_MODEL_PREF / GENERIC_OPEN_AI_EMBEDDING_API_KEY (server/utils/EmbeddingEngines/genericOpenAi/index.js:7-20)
// - Max chunk length:   AnythingLLM's own setting is EmbeddingModelMaxChunkLength (default 1000 tokens if unset,
//   server/utils/helpers/index.js:579-588); bge-m3 supports up to 8192 tokens per input, so that's the value we set.
// ---------------------------------------------------------------------------

pub struct AnythingLlm;

/// A planned `update-env` write. Order matters (see `AnythingLlm::plan`).
pub struct AlWrite {
    pub key: &'static str,
    pub value: String,
}

/// What `AnythingLlm::connect` would do, computed by `plan()` so the GUI can
/// ask for confirmation *before* anything destructive happens.
pub struct AlPlan {
    /// Ordered writes; empty means "already connected" (nothing to do).
    pub writes: Vec<AlWrite>,
    /// True when `EmbeddingEngine` and/or `EmbeddingModelPref` are actually
    /// changing, which deletes every workspace's embedded documents - see
    /// the citation on `AnythingLlm::plan` below.
    pub triggers_reset: bool,
}

impl AnythingLlm {
    pub const NAME: &'static str = "AnythingLLM";
    pub const INITIALS: &'static str = "AL";
    pub const DEFAULT_BASE_URL: &'static str = "http://127.0.0.1:3001";
    const MODEL_NAME: &'static str = "bge-m3";
    const ENGINE: &'static str = "generic-openai";
    /// bge-m3's max input length in tokens (AnythingLLM's `EmbeddingModelMaxChunkLength`).
    const MAX_CHUNK_LENGTH: &'static str = "8192";
    /// AnythingLLM's generic-openai embedder sends this as a bearer token;
    /// our server ignores auth entirely, but an obvious placeholder beats an
    /// empty string (see `OpenWebUi::API_KEY_PLACEHOLDER` for the same call).
    const API_KEY_PLACEHOLDER: &'static str = "sk-local";

    /// `GET /api/ping` needs no auth - matches `{"online": true}` exactly,
    /// same "small custom endpoint, exact shape" approach as the other two
    /// tools' fingerprints.
    pub fn detect(base_url: &str) -> bool {
        let agent = agent(Duration::from_millis(1500));
        match agent.get(&format!("{base_url}/api/ping")).call() {
            Ok(resp) => resp.into_json::<AlPingResponse>().map(|r| r.online).unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Connected when `EmbeddingEngine` is `generic-openai` and
    /// `EmbeddingBasePath` already points at our `embed_url`. `/api/v1/system`
    /// requires a developer API key (`validApiKey` -> 403, not 401, when
    /// missing/wrong) -> NeedsKey; any other failure -> Found.
    pub fn status(base_url: &str, embed_url: &str, key: Option<&str>) -> Status {
        if !Self::detect(base_url) {
            return Status::NotFound;
        }
        let agent = agent(Duration::from_millis(1500));
        let settings = match get_json::<AlSystemResp>(&agent, base_url, "/api/v1/system", key) {
            Ok(s) => s.settings,
            Err(e) if matches!(e.downcast_ref::<ApiError>(), Some(ApiError::Http(403))) => return Status::NeedsKey,
            Err(_) => return Status::Found,
        };
        if settings.embedding_engine.as_deref() == Some(Self::ENGINE)
            && settings.embedding_base_path.as_deref() == Some(embed_url)
        {
            Status::Connected
        } else {
            Status::Found
        }
    }

    /// Reads current settings and works out exactly which keys `connect()`
    /// would need to write, and whether doing so deletes data.
    ///
    /// `EmbeddingEngine` and `EmbeddingModelPref` both carry
    /// `postUpdate: [handleVectorStoreReset]` (server/utils/helpers/updateENV.js:247-260),
    /// and `handleVectorStoreReset` calls `resetAllVectorStores` whenever the
    /// value actually *changes* (updateENV.js:1320-1336). `resetAllVectorStores`
    /// purges the vector cache and deletes every `Document`/`DocumentVectors`
    /// row and every workspace's vector-db namespace
    /// (server/utils/vectorStore/resetAllVectorStores.js:11-46) - i.e. it wipes
    /// all embedded documents in every workspace. So `triggers_reset` is true
    /// only when engine or model would actually change; `EmbeddingBasePath`,
    /// `EmbeddingModelMaxChunkLength` and `GenericOpenAiEmbeddingApiKey` have
    /// no such hook and are always safe to write.
    pub fn plan(base_url: &str, embed_url: &str, key: Option<&str>) -> anyhow::Result<AlPlan> {
        let agent = agent(Duration::from_secs(5));
        let settings = get_json::<AlSystemResp>(&agent, base_url, "/api/v1/system", key)?.settings;

        let triggers_reset = settings.embedding_engine.as_deref() != Some(Self::ENGINE)
            || settings.embedding_model_pref.as_deref() != Some(Self::MODEL_NAME);

        // Safe fields first. `connect()` sends one write per key and stops
        // at the first error, so listing the reset-triggering keys last
        // means a validation failure on a safe field (e.g. `EmbeddingBasePath`
        // failing AnythingLLM's Docker-loopback check) can never let a
        // destructive write slip through beforehand.
        let mut writes = Vec::new();
        if settings.embedding_base_path.as_deref() != Some(embed_url) {
            writes.push(AlWrite { key: "EmbeddingBasePath", value: embed_url.to_string() });
        }
        if settings.embedding_model_max_chunk_length.as_deref() != Some(Self::MAX_CHUNK_LENGTH) {
            writes.push(AlWrite {
                key: "EmbeddingModelMaxChunkLength",
                value: Self::MAX_CHUNK_LENGTH.to_string(),
            });
        }
        if !settings.generic_open_ai_embedding_api_key {
            writes.push(AlWrite {
                key: "GenericOpenAiEmbeddingApiKey",
                value: Self::API_KEY_PLACEHOLDER.to_string(),
            });
        }
        if triggers_reset {
            writes.push(AlWrite { key: "EmbeddingEngine", value: Self::ENGINE.to_string() });
            writes.push(AlWrite { key: "EmbeddingModelPref", value: Self::MODEL_NAME.to_string() });
        }

        Ok(AlPlan { writes, triggers_reset })
    }

    /// Idempotent and confirmation-gated: `plan()` first. If nothing would
    /// change, no-op (`Done(false)`). If the plan would delete embedded
    /// documents (`triggers_reset`) and `confirm_reset` isn't `true`,
    /// returns `NeedsResetConfirmation` and writes *nothing at all* - not
    /// even the safe fields, so a caller can't be surprised by a half-applied
    /// change. Otherwise writes each planned key one request at a time (see
    /// `plan()`'s ordering note) and stops at the first error, surfacing
    /// `update-env`'s own `error` message (server/endpoints/api/system/index.js:106-149
    /// returns `{newValues, error}`, `error` is `false` when there's none -
    /// server/utils/helpers/updateENV.js:1477).
    ///
    /// Per DESIGN.md: switching the embedder means existing workspaces need
    /// re-embedding in AnythingLLM - this never triggers or performs that
    /// re-embed itself, only the (confirmed) config write.
    pub fn connect(base_url: &str, embed_url: &str, key: Option<&str>, confirm_reset: bool) -> anyhow::Result<ConnectOutcome> {
        let plan = Self::plan(base_url, embed_url, key)?;
        if plan.writes.is_empty() {
            return Ok(ConnectOutcome::Done(false));
        }
        if plan.triggers_reset && !confirm_reset {
            return Ok(ConnectOutcome::NeedsResetConfirmation);
        }

        let agent = agent(Duration::from_secs(5));
        for write in &plan.writes {
            let mut body = serde_json::Map::new();
            body.insert(write.key.to_string(), Value::String(write.value.clone()));
            let resp = post_json::<_, AlUpdateEnvResp>(&agent, base_url, "/api/v1/system/update-env", key, &Value::Object(body))?;
            if let Some(msg) = resp.error.as_str() {
                let hint = if write.key == "EmbeddingBasePath" {
                    format!("{msg} (if AnythingLLM runs in Docker, try the \"Runs in Docker\" toggle)")
                } else {
                    msg.to_string()
                };
                anyhow::bail!(hint);
            }
        }
        Ok(ConnectOutcome::Done(true))
    }
}

#[derive(Deserialize, Default)]
struct AlPingResponse {
    #[serde(default)]
    online: bool,
}

#[derive(Deserialize)]
struct AlSystemResp {
    settings: AlSettings,
}

#[derive(Deserialize, Default)]
struct AlSettings {
    #[serde(rename = "EmbeddingEngine", default)]
    embedding_engine: Option<String>,
    #[serde(rename = "EmbeddingBasePath", default)]
    embedding_base_path: Option<String>,
    #[serde(rename = "EmbeddingModelPref", default)]
    embedding_model_pref: Option<String>,
    #[serde(rename = "EmbeddingModelMaxChunkLength", default)]
    embedding_model_max_chunk_length: Option<String>,
    /// `!!process.env.GENERIC_OPEN_AI_EMBEDDING_API_KEY` - a presence
    /// boolean, not the key itself (server/models/systemSettings.js:494-495).
    #[serde(rename = "GenericOpenAiEmbeddingApiKey", default)]
    generic_open_ai_embedding_api_key: bool,
}

/// `POST /api/v1/system/update-env` always answers 200 with
/// `{newValues, error}`; `error` is the string of validation messages
/// joined with `\n`, or JSON `false` when there's none (updateENV.js:1477).
#[derive(Deserialize, Default)]
struct AlUpdateEnvResp {
    #[serde(default)]
    error: Value,
}

// ---------------------------------------------------------------------------
// Manual-setup tools (DESIGN.md "Other tools (manual setup)") - no HTTP probe,
// just a name/initials and a ready-to-paste config snippet for the row's
// "Copy config" button.
// ---------------------------------------------------------------------------

pub struct ManualTool {
    pub name: &'static str,
    pub initials: &'static str,
    pub config: fn(embed_url: &str) -> String,
}

pub const MANUAL_TOOLS: &[ManualTool] = &[
    ManualTool {
        name: "LibreChat",
        initials: "LC",
        config: librechat_config,
    },
    ManualTool {
        name: "Dify",
        initials: "DF",
        config: dify_config,
    },
];

/// RAG_OPENAI_BASEURL/RAG_OPENAI_API_KEY/EMBEDDINGS_PROVIDER per
/// https://www.librechat.ai/docs/configuration/rag_api ("RAG API" env vars).
fn librechat_config(embed_url: &str) -> String {
    format!(
        "# .env (RAG API)\nEMBEDDINGS_PROVIDER=openai\nEMBEDDINGS_MODEL=bge-m3\nRAG_OPENAI_BASEURL={embed_url}\nRAG_OPENAI_API_KEY=sk-local\n"
    )
}

fn dify_config(embed_url: &str) -> String {
    format!(
        "Dify -> Settings -> Model Provider -> OpenAI-API-compatible -> Add Model:\n\
         1. Model Type: Text Embedding\n\
         2. Model Name: bge-m3\n\
         3. API endpoint URL: {embed_url}\n\
         4. API Key: any non-empty value (not checked by this server)\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::{Query, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::routing::get;
    use axum::{Json, Router};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// Emulates just enough of Open Notebook's API (see citations above the
    /// `OpenNotebook` impl) to drive `connect`/`status` against a real HTTP
    /// server, without touching the actual instance on :5055.
    #[derive(Default)]
    struct FakeState {
        credentials: Vec<(String, String, String)>,     // (id, base_url, provider)
        models: Vec<(String, String, String, String)>,  // (id, name, provider, credential)
        default_embedding_model: Option<String>,
        next_id: u32,
        password: Option<&'static str>,
    }

    type Shared = Arc<Mutex<FakeState>>;

    /// `api/middleware.py`: missing/wrong `Authorization: Bearer <password>` -> 401.
    fn check_auth(state: &Shared, headers: &HeaderMap) -> Result<(), StatusCode> {
        let Some(expected) = state.lock().unwrap().password else {
            return Ok(());
        };
        let ok = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            == Some(expected);
        if ok { Ok(()) } else { Err(StatusCode::UNAUTHORIZED) }
    }

    async fn root() -> Json<Value> {
        Json(serde_json::json!({"message": "Open Notebook API is running"}))
    }

    async fn list_credentials(
        State(state): State<Shared>,
        headers: HeaderMap,
        Query(q): Query<HashMap<String, String>>,
    ) -> Result<Json<Value>, StatusCode> {
        check_auth(&state, &headers)?;
        let s = state.lock().unwrap();
        let provider_filter = q.get("provider").cloned();
        let items: Vec<_> = s
            .credentials
            .iter()
            .filter(|(_, _, p)| provider_filter.as_deref().is_none_or(|f| f == p))
            .map(|(id, base_url, _)| serde_json::json!({"id": id, "base_url": base_url}))
            .collect();
        Ok(Json(Value::Array(items)))
    }

    async fn create_credential(
        State(state): State<Shared>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Result<(StatusCode, Json<Value>), StatusCode> {
        check_auth(&state, &headers)?;
        let mut s = state.lock().unwrap();
        s.next_id += 1;
        let id = format!("credential:{}", s.next_id);
        let base_url = body["base_url"].as_str().unwrap_or_default().to_string();
        let provider = body["provider"].as_str().unwrap_or_default().to_string();
        s.credentials.push((id.clone(), base_url.clone(), provider));
        Ok((StatusCode::CREATED, Json(serde_json::json!({"id": id, "base_url": base_url}))))
    }

    async fn list_models(State(state): State<Shared>, headers: HeaderMap) -> Result<Json<Value>, StatusCode> {
        check_auth(&state, &headers)?;
        let s = state.lock().unwrap();
        let items: Vec<_> = s
            .models
            .iter()
            .map(|(id, name, provider, credential)| {
                serde_json::json!({"id": id, "name": name, "provider": provider, "credential": credential})
            })
            .collect();
        Ok(Json(Value::Array(items)))
    }

    async fn create_model(
        State(state): State<Shared>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Result<(StatusCode, Json<Value>), StatusCode> {
        check_auth(&state, &headers)?;
        let mut s = state.lock().unwrap();
        let name = body["name"].as_str().unwrap_or_default().to_string();
        let provider = body["provider"].as_str().unwrap_or_default().to_string();
        let credential = body["credential"].as_str().unwrap_or_default().to_string();
        // api/routers/models.py:217-232 - duplicate (provider, name, type) -> 400.
        if s.models.iter().any(|(_, n, p, _)| n.eq_ignore_ascii_case(&name) && *p == provider) {
            return Err(StatusCode::BAD_REQUEST);
        }
        s.next_id += 1;
        let id = format!("model:{}", s.next_id);
        s.models.push((id.clone(), name.clone(), provider.clone(), credential.clone()));
        Ok((StatusCode::CREATED, Json(serde_json::json!({"id": id, "name": name, "provider": provider, "credential": credential}))))
    }

    async fn get_defaults(State(state): State<Shared>, headers: HeaderMap) -> Result<Json<Value>, StatusCode> {
        check_auth(&state, &headers)?;
        let s = state.lock().unwrap();
        Ok(Json(serde_json::json!({"default_embedding_model": s.default_embedding_model})))
    }

    async fn put_defaults(
        State(state): State<Shared>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Result<Json<Value>, StatusCode> {
        check_auth(&state, &headers)?;
        let mut s = state.lock().unwrap();
        s.default_embedding_model = body["default_embedding_model"].as_str().map(|s| s.to_string());
        Ok(Json(serde_json::json!({"default_embedding_model": s.default_embedding_model})))
    }

    fn spawn_fake_server(password: Option<&'static str>) -> (String, Shared) {
        let mut init = FakeState::default();
        init.password = password;
        let state: Shared = Arc::new(Mutex::new(init));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap(); // required before tokio::net::TcpListener::from_std
        let base_url = format!("http://{addr}");

        let app_state = state.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let app = Router::new()
                    .route("/", get(root))
                    .route("/api/credentials", get(list_credentials).post(create_credential))
                    .route("/api/models", get(list_models).post(create_model))
                    .route("/api/models/defaults", get(get_defaults).put(put_defaults))
                    .with_state(app_state);
                axum::serve(listener, app).await.unwrap();
            });
        });
        // Give the background thread a moment to actually start listening.
        std::thread::sleep(Duration::from_millis(150));
        (base_url, state)
    }

    #[test]
    fn detect_matches_fingerprint_only() {
        let (base_url, _state) = spawn_fake_server(None);
        assert!(OpenNotebook::detect(&base_url));
        assert!(!OpenNotebook::detect("http://127.0.0.1:1")); // nothing listening
    }

    #[test]
    fn connect_is_idempotent_and_flips_default() {
        let (base_url, state) = spawn_fake_server(None);
        let embed_url = "http://127.0.0.1:11435/v1";

        assert_eq!(OpenNotebook::status(&base_url, embed_url, None), Status::Found);

        let changed = OpenNotebook::connect(&base_url, embed_url, None).unwrap();
        assert!(changed, "first connect must flip the default");
        {
            let s = state.lock().unwrap();
            assert_eq!(s.credentials.len(), 1);
            assert_eq!(s.models.len(), 1);
            assert_eq!(s.models[0].1, "bge-m3");
            assert_eq!(s.default_embedding_model.as_deref(), Some(s.models[0].0.as_str()));
        }
        assert_eq!(OpenNotebook::status(&base_url, embed_url, None), Status::Connected);

        // Second connect must not create a duplicate credential or model,
        // and must report nothing changed.
        let changed = OpenNotebook::connect(&base_url, embed_url, None).unwrap();
        assert!(!changed, "second connect is a no-op");
        let s = state.lock().unwrap();
        assert_eq!(s.credentials.len(), 1);
        assert_eq!(s.models.len(), 1);
    }

    #[test]
    fn password_protected_server_reports_needs_key_until_connect_supplies_it() {
        let (base_url, _state) = spawn_fake_server(Some("hunter2"));
        let embed_url = "http://127.0.0.1:11435/v1";

        assert_eq!(OpenNotebook::status(&base_url, embed_url, None), Status::NeedsKey);
        assert!(OpenNotebook::connect(&base_url, embed_url, None).is_err());

        OpenNotebook::connect(&base_url, embed_url, Some("hunter2")).unwrap();
        assert_eq!(OpenNotebook::status(&base_url, embed_url, Some("hunter2")), Status::Connected);
    }

    /// The user's real Open Notebook already has a `bge-m3` model bound to a
    /// *different* credential (an older server on another port). `connect()`
    /// must not reuse or rename that model - it must create a second,
    /// distinctly named model on our own credential, leaving the old
    /// model/credential pair completely untouched.
    #[test]
    fn preexisting_bge_m3_on_another_credential_is_left_alone() {
        let (base_url, state) = spawn_fake_server(None);
        let embed_url = "http://127.0.0.1:11435/v1";

        let (other_cred_id, other_model_id) = {
            let mut s = state.lock().unwrap();
            s.next_id += 1;
            let cred_id = format!("credential:{}", s.next_id);
            s.credentials.push((cred_id.clone(), "http://127.0.0.1:9999/v1".to_string(), "openai_compatible".to_string()));
            s.next_id += 1;
            let model_id = format!("model:{}", s.next_id);
            s.models.push((model_id.clone(), "bge-m3".to_string(), "openai_compatible".to_string(), cred_id.clone()));
            s.default_embedding_model = Some(model_id.clone());
            (cred_id, model_id)
        };

        let changed = OpenNotebook::connect(&base_url, embed_url, None).unwrap();
        assert!(changed);

        let s = state.lock().unwrap();
        assert_eq!(s.credentials.len(), 2, "our credential is new, the old one is untouched");
        assert_eq!(s.models.len(), 2, "our model is new, the old bge-m3 is untouched");

        // The old model/credential pair is exactly as it was.
        assert!(s.credentials.iter().any(|(id, url, _)| id == &other_cred_id && url == "http://127.0.0.1:9999/v1"));
        assert!(s.models.iter().any(|(id, name, _, cred)| id == &other_model_id && name == "bge-m3" && cred == &other_cred_id));

        // Our new model is bound to our credential, distinctly named (not
        // exactly "bge-m3", since that name is taken), and is now default.
        let our_cred_id = s.credentials.iter().find(|(_, url, _)| url == embed_url).unwrap().0.clone();
        let our_model = s.models.iter().find(|(_, _, _, cred)| cred == &our_cred_id).unwrap();
        assert_ne!(our_model.0, other_model_id);
        assert_ne!(our_model.1, "bge-m3");
        assert!(our_model.1.starts_with("bge-m3"), "name: {}", our_model.1);
        assert_eq!(s.default_embedding_model.as_deref(), Some(our_model.0.as_str()));
        drop(s);

        assert_eq!(OpenNotebook::status(&base_url, embed_url, None), Status::Connected);

        // Second connect creates nothing new at all.
        let changed = OpenNotebook::connect(&base_url, embed_url, None).unwrap();
        assert!(!changed);
        let s = state.lock().unwrap();
        assert_eq!(s.credentials.len(), 2);
        assert_eq!(s.models.len(), 2);
    }
}

#[cfg(test)]
mod owui_tests {
    use super::*;
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode};
    use axum::routing::get;
    use axum::{Json, Router};
    use std::sync::{Arc, Mutex};

    /// Emulates just enough of Open WebUI's API (see citations above the
    /// `OpenWebUi` impl) to drive `connect`/`status` against a real HTTP
    /// server, without touching a real instance on :8080.
    #[derive(Default)]
    struct FakeState {
        engine: String,
        model: String,
        openai_url: Option<String>,
        openai_key: Option<String>,
        batch_size: Option<i64>,
        async_embedding: Option<bool>,
        concurrent_requests: Option<i64>,
        admin_key: Option<&'static str>,
    }

    type Shared = Arc<Mutex<FakeState>>;

    /// `get_admin_user`: missing/wrong `Authorization: Bearer <token>` -> 401.
    fn check_auth(state: &Shared, headers: &HeaderMap) -> Result<(), StatusCode> {
        let Some(expected) = state.lock().unwrap().admin_key else {
            return Ok(());
        };
        let ok = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            == Some(expected);
        if ok { Ok(()) } else { Err(StatusCode::UNAUTHORIZED) }
    }

    async fn config() -> Json<Value> {
        Json(serde_json::json!({"status": true, "name": "Open WebUI", "default_locale": "en-US", "features": {}}))
    }

    /// A generic server's `/api/config` (or whatever unrelated endpoint) -
    /// used to prove `detect()` doesn't fire on just any JSON server.
    async fn generic_config() -> Json<Value> {
        Json(serde_json::json!({"status": true}))
    }

    async fn get_embedding(State(state): State<Shared>, headers: HeaderMap) -> Result<Json<Value>, StatusCode> {
        check_auth(&state, &headers)?;
        let s = state.lock().unwrap();
        Ok(Json(serde_json::json!({
            "status": true,
            "RAG_EMBEDDING_ENGINE": s.engine,
            "RAG_EMBEDDING_MODEL": s.model,
            "RAG_EMBEDDING_BATCH_SIZE": s.batch_size,
            "ENABLE_ASYNC_EMBEDDING": s.async_embedding,
            "RAG_EMBEDDING_CONCURRENT_REQUESTS": s.concurrent_requests,
            "openai_config": {"url": s.openai_url, "key": s.openai_key},
            "ollama_config": {"url": Value::Null, "key": Value::Null},
            "azure_openai_config": {"url": Value::Null, "key": Value::Null, "version": Value::Null},
        })))
    }

    async fn update_embedding(
        State(state): State<Shared>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Result<Json<Value>, StatusCode> {
        check_auth(&state, &headers)?;
        let mut s = state.lock().unwrap();
        s.engine = body["RAG_EMBEDDING_ENGINE"].as_str().unwrap_or_default().to_string();
        s.model = body["RAG_EMBEDDING_MODEL"].as_str().unwrap_or_default().to_string();
        s.openai_url = body["openai_config"]["url"].as_str().map(str::to_string);
        s.openai_key = body["openai_config"]["key"].as_str().map(str::to_string);
        s.batch_size = body["RAG_EMBEDDING_BATCH_SIZE"].as_i64();
        s.async_embedding = body["ENABLE_ASYNC_EMBEDDING"].as_bool();
        s.concurrent_requests = body["RAG_EMBEDDING_CONCURRENT_REQUESTS"].as_i64();
        Ok(Json(serde_json::json!({"status": true})))
    }

    fn spawn_fake_server(admin_key: Option<&'static str>) -> (String, Shared) {
        let mut init = FakeState::default();
        init.admin_key = admin_key;
        let state: Shared = Arc::new(Mutex::new(init));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let base_url = format!("http://{addr}");

        let app_state = state.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let app = Router::new()
                    .route("/api/config", get(config))
                    .route("/api/v1/retrieval/embedding", get(get_embedding))
                    .route("/api/v1/retrieval/embedding/update", axum::routing::post(update_embedding))
                    .with_state(app_state);
                axum::serve(listener, app).await.unwrap();
            });
        });
        std::thread::sleep(Duration::from_millis(150));
        (base_url, state)
    }

    fn spawn_generic_server() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let base_url = format!("http://{addr}");
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let app = Router::new().route("/api/config", get(generic_config));
                axum::serve(listener, app).await.unwrap();
            });
        });
        std::thread::sleep(Duration::from_millis(150));
        base_url
    }

    #[test]
    fn detect_matches_fingerprint_and_rejects_generic_server() {
        let (base_url, _state) = spawn_fake_server(None);
        assert!(OpenWebUi::detect(&base_url));
        assert!(!OpenWebUi::detect(&spawn_generic_server()), "a /api/config without default_locale must not match");
        assert!(!OpenWebUi::detect("http://127.0.0.1:1")); // nothing listening
    }

    #[test]
    fn needs_key_without_admin_key() {
        let (base_url, _state) = spawn_fake_server(Some("adminkey"));
        let embed_url = "http://127.0.0.1:11435/v1";
        assert_eq!(OpenWebUi::status(&base_url, embed_url, None), Status::NeedsKey);
        assert!(OpenWebUi::connect(&base_url, embed_url, None).is_err());
    }

    #[test]
    fn connect_is_idempotent_and_sets_openai_engine() {
        let (base_url, state) = spawn_fake_server(Some("adminkey"));
        let embed_url = "http://127.0.0.1:11435/v1";

        assert_eq!(OpenWebUi::status(&base_url, embed_url, Some("adminkey")), Status::Found);

        let changed = OpenWebUi::connect(&base_url, embed_url, Some("adminkey")).unwrap();
        assert!(changed, "first connect must write the embedding config");
        {
            let s = state.lock().unwrap();
            assert_eq!(s.engine, "openai");
            assert_eq!(s.model, "bge-m3");
            assert_eq!(s.openai_url.as_deref(), Some(embed_url));
        }
        assert_eq!(OpenWebUi::status(&base_url, embed_url, Some("adminkey")), Status::Connected);

        // Second connect must not write anything and must report no change.
        let changed = OpenWebUi::connect(&base_url, embed_url, Some("adminkey")).unwrap();
        assert!(!changed, "second connect is a no-op");
    }
}

#[cfg(test)]
mod anythingllm_tests {
    use super::*;
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode};
    use axum::routing::get;
    use axum::{Json, Router};
    use std::sync::{Arc, Mutex};

    /// Emulates just enough of AnythingLLM's developer API (see citations
    /// above the `AnythingLlm` impl) to drive `connect`/`status` against a
    /// real HTTP server, without touching a real instance on :3001.
    #[derive(Default)]
    struct FakeState {
        engine: String,
        base_path: String,
        model_pref: String,
        max_chunk_length: String,
        embedding_api_key: String,
        dev_key: Option<&'static str>,
        /// Number of `update-env` calls received - used to assert "wrote
        /// nothing" without inspecting every field individually.
        write_calls: u32,
        /// When set, `EmbeddingBasePath` writes are rejected with this
        /// message instead of applied - emulates `validDockerizedUrl`.
        reject_base_path_with: Option<&'static str>,
    }

    type Shared = Arc<Mutex<FakeState>>;

    /// `validApiKey`: missing/wrong `Authorization: Bearer <key>` -> 403
    /// (not 401 - AnythingLLM's own convention, unlike the other two tools).
    fn check_auth(state: &Shared, headers: &HeaderMap) -> Result<(), StatusCode> {
        let Some(expected) = state.lock().unwrap().dev_key else {
            return Ok(());
        };
        let ok = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            == Some(expected);
        if ok { Ok(()) } else { Err(StatusCode::FORBIDDEN) }
    }

    async fn ping() -> Json<Value> {
        Json(serde_json::json!({"online": true}))
    }

    /// A generic server's health endpoint - proves `detect()` doesn't fire
    /// on any JSON body, only the exact `{"online": true}` shape.
    async fn generic_ping() -> Json<Value> {
        Json(serde_json::json!({"ok": true}))
    }

    async fn get_system(State(state): State<Shared>, headers: HeaderMap) -> Result<Json<Value>, StatusCode> {
        check_auth(&state, &headers)?;
        let s = state.lock().unwrap();
        Ok(Json(serde_json::json!({
            "settings": {
                "EmbeddingEngine": s.engine,
                "EmbeddingBasePath": s.base_path,
                "EmbeddingModelPref": s.model_pref,
                "EmbeddingModelMaxChunkLength": s.max_chunk_length,
                "GenericOpenAiEmbeddingApiKey": !s.embedding_api_key.is_empty(),
            }
        })))
    }

    async fn update_env(
        State(state): State<Shared>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Result<Json<Value>, StatusCode> {
        check_auth(&state, &headers)?;
        let mut s = state.lock().unwrap();
        s.write_calls += 1;
        if let Some(v) = body["EmbeddingBasePath"].as_str() {
            if let Some(msg) = s.reject_base_path_with {
                // updateENV.js:1446-1449 - a validation error means this
                // key's value is never applied.
                return Ok(Json(serde_json::json!({"newValues": {}, "error": msg})));
            }
            s.base_path = v.to_string();
        }
        if let Some(v) = body["EmbeddingEngine"].as_str() {
            s.engine = v.to_string();
        }
        if let Some(v) = body["EmbeddingModelPref"].as_str() {
            s.model_pref = v.to_string();
        }
        if let Some(v) = body["EmbeddingModelMaxChunkLength"].as_str() {
            s.max_chunk_length = v.to_string();
        }
        if let Some(v) = body["GenericOpenAiEmbeddingApiKey"].as_str() {
            s.embedding_api_key = v.to_string();
        }
        Ok(Json(serde_json::json!({"newValues": body, "error": false})))
    }

    fn spawn_fake_server(dev_key: Option<&'static str>) -> (String, Shared) {
        spawn_fake_server_with(dev_key, None)
    }

    fn spawn_fake_server_with(dev_key: Option<&'static str>, reject_base_path_with: Option<&'static str>) -> (String, Shared) {
        let mut init = FakeState::default();
        init.dev_key = dev_key;
        init.reject_base_path_with = reject_base_path_with;
        let state: Shared = Arc::new(Mutex::new(init));
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let base_url = format!("http://{addr}");

        let app_state = state.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let app = Router::new()
                    .route("/api/ping", get(ping))
                    .route("/api/v1/system", get(get_system))
                    .route("/api/v1/system/update-env", axum::routing::post(update_env))
                    .with_state(app_state);
                axum::serve(listener, app).await.unwrap();
            });
        });
        std::thread::sleep(Duration::from_millis(150));
        (base_url, state)
    }

    fn spawn_generic_server() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let base_url = format!("http://{addr}");
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let app = Router::new().route("/api/ping", get(generic_ping));
                axum::serve(listener, app).await.unwrap();
            });
        });
        std::thread::sleep(Duration::from_millis(150));
        base_url
    }

    #[test]
    fn detect_matches_fingerprint_and_rejects_generic_server() {
        let (base_url, _state) = spawn_fake_server(None);
        assert!(AnythingLlm::detect(&base_url));
        assert!(!AnythingLlm::detect(&spawn_generic_server()), "a /api/ping with a different body must not match");
        assert!(!AnythingLlm::detect("http://127.0.0.1:1")); // nothing listening
    }

    #[test]
    fn needs_key_without_dev_key() {
        let (base_url, _state) = spawn_fake_server(Some("devkey"));
        let embed_url = "http://127.0.0.1:11435/v1";
        assert_eq!(AnythingLlm::status(&base_url, embed_url, None), Status::NeedsKey);
        assert!(AnythingLlm::connect(&base_url, embed_url, None, false).is_err());
    }

    /// (a) Engine differs from ours -> the plan would delete every
    /// workspace's embedded documents. Without `confirm_reset`, `connect()`
    /// must refuse and write *nothing at all*, not even the safe fields.
    #[test]
    fn reset_triggering_change_without_confirmation_writes_nothing() {
        let (base_url, state) = spawn_fake_server(Some("devkey"));
        let embed_url = "http://127.0.0.1:11435/v1";

        let plan = AnythingLlm::plan(&base_url, embed_url, Some("devkey")).unwrap();
        assert!(plan.triggers_reset, "fresh server has no engine set, so engine must change");

        let outcome = AnythingLlm::connect(&base_url, embed_url, Some("devkey"), false).unwrap();
        assert_eq!(outcome, ConnectOutcome::NeedsResetConfirmation);

        let s = state.lock().unwrap();
        assert_eq!(s.write_calls, 0, "must not write anything before confirmation");
        assert_eq!(s.engine, "");
        assert_eq!(s.base_path, "");
    }

    /// (b) Same scenario, but confirmed - everything gets written and the
    /// tool reports Connected afterwards.
    #[test]
    fn reset_triggering_change_with_confirmation_writes_everything() {
        let (base_url, state) = spawn_fake_server(Some("devkey"));
        let embed_url = "http://127.0.0.1:11435/v1";

        assert_eq!(AnythingLlm::status(&base_url, embed_url, Some("devkey")), Status::Found);

        let outcome = AnythingLlm::connect(&base_url, embed_url, Some("devkey"), true).unwrap();
        assert_eq!(outcome, ConnectOutcome::Done(true));
        {
            let s = state.lock().unwrap();
            assert_eq!(s.engine, "generic-openai");
            assert_eq!(s.base_path, embed_url);
            assert_eq!(s.model_pref, "bge-m3");
            assert_eq!(s.max_chunk_length, "8192");
        }
        assert_eq!(AnythingLlm::status(&base_url, embed_url, Some("devkey")), Status::Connected);

        // Second connect must not write anything and must report no change -
        // no confirmation needed since nothing would change.
        let outcome = AnythingLlm::connect(&base_url, embed_url, Some("devkey"), false).unwrap();
        assert_eq!(outcome, ConnectOutcome::Done(false));
    }

    /// (c) Engine/model already match; only the base path differs (e.g. our
    /// server's port changed). `plan()` must not touch the reset-triggering
    /// keys at all, and `connect()` must proceed without any confirmation.
    #[test]
    fn base_path_only_change_needs_no_confirmation() {
        let (base_url, state) = spawn_fake_server(Some("devkey"));
        let old_embed_url = "http://127.0.0.1:11111/v1";
        let new_embed_url = "http://127.0.0.1:22222/v1";

        // Pre-seed the server as if a previous connect() already ran against
        // a different port, so only EmbeddingBasePath is out of date.
        AnythingLlm::connect(&base_url, old_embed_url, Some("devkey"), true).unwrap();
        {
            let mut s = state.lock().unwrap();
            s.write_calls = 0;
        }

        let plan = AnythingLlm::plan(&base_url, new_embed_url, Some("devkey")).unwrap();
        assert!(!plan.triggers_reset);
        assert_eq!(plan.writes.len(), 1, "only EmbeddingBasePath should need writing");
        assert_eq!(plan.writes[0].key, "EmbeddingBasePath");

        let outcome = AnythingLlm::connect(&base_url, new_embed_url, Some("devkey"), false).unwrap();
        assert_eq!(outcome, ConnectOutcome::Done(true));
        let s = state.lock().unwrap();
        assert_eq!(s.write_calls, 1, "must write exactly one key");
        assert_eq!(s.base_path, new_embed_url);
        assert_eq!(s.engine, "generic-openai", "engine must be left alone");
        assert_eq!(s.model_pref, "bge-m3", "model must be left alone");
    }

    /// AnythingLLM's Docker loopback check rejects `EmbeddingBasePath` -
    /// `connect()` must surface that message (with a hint about the "Runs in
    /// Docker" toggle) instead of silently succeeding or panicking, and must
    /// stop before ever reaching the reset-triggering keys.
    #[test]
    fn base_path_validation_error_is_surfaced_and_stops_before_reset() {
        let (base_url, state) =
            spawn_fake_server_with(Some("devkey"), Some("Port is not running a reachable service on loopback"));
        let embed_url = "http://127.0.0.1:11435/v1";

        let err = AnythingLlm::connect(&base_url, embed_url, Some("devkey"), true).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("loopback"), "original AnythingLLM message must be preserved: {msg}");
        assert!(msg.contains("Runs in Docker"), "must hint at the Docker toggle: {msg}");

        let s = state.lock().unwrap();
        assert_eq!(s.write_calls, 1, "must stop at the first failing write");
        assert_eq!(s.engine, "", "must never reach the reset-triggering keys");
    }
}

// Manual smoke test against a real, running Open Notebook. Read-only: only
// detect()/status(), never connect(). Run with:
//   BGE_SMOKE_ON=http://127.0.0.1:5055 BGE_SMOKE_EMBED=http://127.0.0.1:11434/v1 cargo test --release -- --ignored smoke
#[cfg(test)]
mod smoke {
    use super::*;

    #[test]
    #[ignore]
    fn smoke_real_open_notebook_status() {
        let on = std::env::var("BGE_SMOKE_ON").unwrap_or_else(|_| "http://127.0.0.1:5055".into());
        let embed = std::env::var("BGE_SMOKE_EMBED").unwrap_or_else(|_| "http://127.0.0.1:11435/v1".into());
        println!("detect({on}) = {}", OpenNotebook::detect(&on));
        println!("status({on}, {embed}) = {:?}", OpenNotebook::status(&on, &embed, None));
    }
}
