//! Connector layer for the Connections screen (design/DESIGN.md).
//!
//! ponytail: only one tool (Open Notebook) actually talks over HTTP in this
//! task, so this is a plain struct with `detect`/`status`/`connect` methods -
//! not an enum/trait dispatch. Promote to an enum over `Tool` variants if/when
//! a second HTTP-probed connector shows up; LibreChat/Dify below are
//! "manual setup" only (no network calls), so they don't need the same shape.
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
