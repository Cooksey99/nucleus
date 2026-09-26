//! Grok / xAI provider — OAuth/API key auth + Responses API chat.

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::info;

use crate::models::EmbeddingModel;
use crate::Config;

use super::types::{ChatRequest, ChatResponse, Message, Provider, ProviderError, Result};

const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const DEVICE_URL: &str = "https://auth.x.ai/oauth2/device/code";
const TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
const SCOPE: &str = "openid profile email offline_access grok-cli:access api:access conversations:read conversations:write";
const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
const REASONING_EFFORTS: &[&str] = &["low", "medium", "high", "xhigh"];
const API_BASE: &str = "https://api.x.ai/v1";
const PROXY_BASE: &str = "https://cli-chat-proxy.grok.com/v1";
/// How long to wait for the next SSE chunk before treating the stream as stalled.
const STREAM_IDLE: Duration = Duration::from_secs(600);

/// Pick the host from the credential, not the model id.
///
/// Session tokens belong on the CLI chat proxy. API keys belong on the public API.
/// Localhost and either of those two hosts are not an explicit override.
fn resolve_base_url(configured: &str, using_api_key: bool) -> String {
    let configured = configured.trim().trim_end_matches('/');
    let unset = configured.is_empty()
        || configured.contains("localhost")
        || configured.contains("127.0.0.1");
    let known = configured == API_BASE || configured == PROXY_BASE;
    if !unset && !known {
        return configured.to_string();
    }
    if using_api_key {
        API_BASE.to_string()
    } else {
        PROXY_BASE.to_string()
    }
}

fn grok_cli_version() -> String {
    let fallback = "1.0.41".to_string();
    let Ok(home) = std::env::var("HOME") else {
        return fallback;
    };
    let Ok(raw) = fs::read_to_string(PathBuf::from(home).join(".grok/version.json")) else {
        return fallback;
    };
    serde_json::from_str::<Value>(&raw)
        .ok()
        .and_then(|v| {
            v.get("stable_version")
                .or_else(|| v.get("version"))
                .and_then(|s| s.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .unwrap_or(fallback)
}

fn new_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "{:x}-{:x}-{:x}",
        now(),
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tokens {
    access_token: String,
    refresh_token: String,
    /// Unix seconds when access_token should be treated as expired.
    expires_at: u64,
}

#[derive(Deserialize)]
struct DeviceCode {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    interval: Option<u64>,
}

#[derive(Deserialize)]
struct TokenResp {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
    error: Option<String>,
}

fn auth_file() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("NUCLEUS_GROK_AUTH_FILE") {
        return Ok(p.into());
    }
    let home = std::env::var_os("HOME")
        .ok_or_else(|| ProviderError::Other("HOME not set".into()))?;
    Ok(PathBuf::from(home).join(".nucleus/grok-auth.json"))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn save(tokens: &Tokens) -> Result<()> {
    let path = auth_file()?;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| ProviderError::Other(e.to_string()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
        }
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(tokens)?)
        .map_err(|e| ProviderError::Other(e.to_string()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600));
    }
    fs::rename(tmp, path).map_err(|e| ProviderError::Other(e.to_string()))
}

fn load() -> Result<Tokens> {
    let path = auth_file()?;
    let raw = fs::read_to_string(&path).map_err(|e| {
        if e.kind() == io::ErrorKind::NotFound {
            ProviderError::Other("not logged in; run `nucleus login`".into())
        } else {
            ProviderError::Other(e.to_string())
        }
    })?;
    Ok(serde_json::from_str(&raw)?)
}

/// Device-code login. Calls `prompt(url, user_code)` once, then polls until done.
pub async fn login(mut prompt: impl FnMut(&str, &str)) -> Result<()> {
    let http = reqwest::Client::new();
    let device: DeviceCode = http
        .post(DEVICE_URL)
        .form(&[("client_id", CLIENT_ID), ("scope", SCOPE)])
        .send()
        .await?
        .error_for_status()
        .map_err(|e| ProviderError::Other(e.to_string()))?
        .json()
        .await?;

    prompt(
        device
            .verification_uri_complete
            .as_deref()
            .unwrap_or(&device.verification_uri),
        &device.user_code,
    );

    let deadline = now() + device.expires_in.unwrap_or(900);
    let mut interval = device.interval.unwrap_or(5).max(1);

    loop {
        if now() >= deadline {
            return Err(ProviderError::Other("login timed out".into()));
        }
        tokio::time::sleep(Duration::from_secs(interval)).await;

        let resp = http
            .post(TOKEN_URL)
            .form(&[
                ("grant_type", DEVICE_GRANT),
                ("device_code", device.device_code.as_str()),
                ("client_id", CLIENT_ID),
            ])
            .send()
            .await?;
        let body: TokenResp = resp.json().await?;

        if let (Some(access), Some(refresh)) = (body.access_token, body.refresh_token) {
            save(&Tokens {
                access_token: access,
                refresh_token: refresh,
                expires_at: now() + body.expires_in.unwrap_or(3600) - 60,
            })?;
            return Ok(());
        }

        match body.error.as_deref() {
            Some("authorization_pending") => {}
            Some("slow_down") => interval = (interval + 5).min(30),
            Some("expired_token") => return Err(ProviderError::Other("device code expired".into())),
            Some("access_denied") | Some("authorization_denied") => {
                return Err(ProviderError::Other("login denied".into()))
            }
            other => {
                return Err(ProviderError::Api(format!(
                    "token error: {}",
                    other.unwrap_or("unknown")
                )))
            }
        }
    }
}

pub fn logout() -> Result<()> {
    match fs::remove_file(auth_file()?) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(ProviderError::Other(e.to_string())),
    }
}

/// Access token for API calls (refreshes when near expiry).
pub async fn access_token() -> Result<String> {
    let t = load()?;
    if now() < t.expires_at {
        return Ok(t.access_token);
    }

    let http = reqwest::Client::new();
    let resp = http
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", CLIENT_ID),
            ("refresh_token", t.refresh_token.as_str()),
        ])
        .send()
        .await?
        .error_for_status()
        .map_err(|_| ProviderError::Other("refresh failed; run `nucleus login`".into()))?;

    let body: TokenResp = resp.json().await?;
    let access = body
        .access_token
        .ok_or_else(|| ProviderError::Other("refresh missing access_token".into()))?;
    let refresh = body
        .refresh_token
        .ok_or_else(|| ProviderError::Other("refresh missing refresh_token".into()))?;

    save(&Tokens {
        access_token: access.clone(),
        refresh_token: refresh,
        expires_at: now() + body.expires_in.unwrap_or(3600) - 60,
    })?;
    Ok(access)
}

#[derive(Debug, Clone)]
pub struct GrokProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    session_id: String,
}

impl GrokProvider {
    pub fn new(config: &Config) -> Result<Self> {
        let api_key = config
            .llm
            .api_key
            .clone()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| std::env::var("XAI_API_KEY").ok().filter(|s| !s.trim().is_empty()));

        if api_key.is_none() && !auth_file().map(|p| p.is_file()).unwrap_or(false) {
            return Err(ProviderError::Other(
                "No Grok credentials. Run `nucleus login` or set XAI_API_KEY.".into(),
            ));
        }

        let base_url = resolve_base_url(&config.llm.base_url, api_key.is_some());
        info!(
            model = %config.llm.model,
            base_url = %base_url,
            auth = if api_key.is_some() { "api_key" } else { "oauth" },
            "Grok provider ready"
        );

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| ProviderError::Other(e.to_string()))?;

        Ok(Self {
            client,
            base_url,
            api_key,
            model: config.llm.model.clone(),
            session_id: new_id(),
        })
    }

    fn uses_cli_proxy(&self) -> bool {
        self.api_key.is_none() && !self.base_url.contains("api.x.ai")
    }

    fn handle_frame(
        &self,
        model: &str,
        frame: &str,
        callback: &mut dyn FnMut(ChatResponse),
        announced: &mut bool,
    ) -> Result<bool> {
        let mut saw = false;
        for line in frame.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                saw = true;
                continue;
            }
            let Ok(event) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            saw = true;
            if let Some(err) = stream_error(&event) {
                return Err(ProviderError::Api(err));
            }
            if let Some(note) = reasoning_delta(&event) {
                if !*announced {
                    info!("Grok is reasoning");
                    *announced = true;
                }
                eprint!("{note}");
                let _ = std::io::stderr().flush();
                continue;
            }
            if let Some(chunk) = parse_stream_event(model, &event) {
                if *announced {
                    eprintln!();
                    *announced = false;
                }
                callback(chunk);
                continue;
            }
            if !*announced {
                info!(event = event_type(&event), "Grok stream open");
                *announced = true;
            }
        }
        Ok(saw)
    }

    pub async fn bearer_token(&self) -> Result<String> {
        if let Some(key) = &self.api_key {
            Ok(key.clone())
        } else {
            access_token().await
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl Provider for GrokProvider {
    async fn chat<'a>(
        &'a self,
        request: ChatRequest,
        mut callback: Box<dyn FnMut(ChatResponse) + Send + 'a>,
    ) -> Result<()> {
        let token = self.bearer_token().await?;
        let model = if request.model.is_empty() {
            self.model.clone()
        } else {
            request.model.clone()
        };

        let mut body = json!({
            "model": model,
            "input": to_grok_input(&request.messages),
            "temperature": request.temperature,
            "stream": true,
            "store": false,
            "prompt_cache_key": self.session_id,
        });

        if let Some(effort) = request.reasoning_effort.as_deref() {
            let effort = effort.to_ascii_lowercase();
            if !REASONING_EFFORTS.contains(&effort.as_str()) {
                return Err(ProviderError::InvalidParam(format!(
                    "Grok reasoning_effort must be one of {REASONING_EFFORTS:?}, got {effort}"
                )));
            }
            body["reasoning"] = json!({ "effort": effort });
        }

        if let Some(tools) = request.tools.as_ref() {
            if !tools.is_empty() {
                body["tools"] = json!(to_grok_tools(tools));
            }
        }

        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        info!(%url, %model, "Grok request sent");
        let mut req = self
            .client
            .post(&url)
            .bearer_auth(&token)
            .header("Accept", "text/event-stream")
            .json(&body);
        if self.uses_cli_proxy() {
            // The proxy dispatches from x-grok-model-override, not the JSON model field.
            req = req
                .header("User-Agent", "xai-grok-cli")
                .header("X-XAI-Token-Auth", "xai-grok-cli")
                .header("x-grok-client-identifier", "grok-shell")
                .header("x-grok-client-version", grok_cli_version())
                .header("x-grok-client-mode", "interactive")
                .header("x-grok-model-override", &model)
                .header("x-grok-conv-id", &self.session_id)
                .header("x-grok-session-id", &self.session_id)
                .header("x-grok-req-id", new_id());
        }
        let response = req.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let err = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api(format!("Grok chat failed ({status}): {err}")));
        }

        info!(status = %response.status(), "Grok stream started");
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut saw_event = false;
        let mut announced = false;

        loop {
            let next = tokio::time::timeout(STREAM_IDLE, stream.next()).await;
            let chunk = match next {
                Ok(Some(chunk)) => chunk?,
                Ok(None) => break,
                Err(_) => {
                    return Err(ProviderError::Api(
                        "Grok stream stalled: no data for 10 minutes".into(),
                    ));
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            normalize_newlines(&mut buffer);

            while let Some(frame) = pop_sse_frame(&mut buffer) {
                saw_event |= self.handle_frame(&model, &frame, &mut callback, &mut announced)?;
            }
        }

        if !buffer.trim().is_empty() {
            saw_event |= self.handle_frame(&model, &buffer, &mut callback, &mut announced)?;
        }
        if announced {
            eprintln!();
        }
        if !saw_event {
            return Err(ProviderError::Api(
                "Grok stream closed before the first event".into(),
            ));
        }

        // Ensure callers always see a terminal chunk.
        callback(ChatResponse {
            model,
            content: String::new(),
            done: true,
            message: Message::assistant(None, ""),
        });

        Ok(())
    }

    async fn embed(&self, _text: &str, _model: &EmbeddingModel) -> Result<Vec<f32>> {
        Err(ProviderError::Other(
            "Grok embeddings not supported; use a local embedder for RAG".into(),
        ))
    }
}

fn normalize_newlines(buffer: &mut String) {
    if buffer.contains('\r') {
        *buffer = buffer.replace("\r\n", "\n").replace('\r', "\n");
    }
}

fn pop_sse_frame(buffer: &mut String) -> Option<String> {
    let idx = buffer.find("\n\n")?;
    let frame = buffer[..idx].to_string();
    buffer.drain(..idx + 2);
    Some(frame)
}

fn event_type(event: &Value) -> &str {
    event.get("type").and_then(|v| v.as_str()).unwrap_or("event")
}

fn reasoning_delta(event: &Value) -> Option<&str> {
    match event_type(event) {
        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => event
            .get("delta")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty()),
        _ => None,
    }
}

fn stream_error(event: &Value) -> Option<String> {
    let ty = event_type(event);
    if ty != "error" && ty != "response.failed" {
        return None;
    }
    let message = event
        .pointer("/error/message")
        .or_else(|| event.get("message"))
        .or_else(|| event.get("error"))
        .and_then(|v| v.as_str())
        .unwrap_or("Grok stream error");
    Some(message.to_string())
}

fn to_grok_input(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            json!({
                "role": m.role,
                "content": m.content,
            })
        })
        .collect()
}

fn to_grok_tools(tools: &[super::types::Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            })
        })
        .collect()
}

/// Map Responses API SSE events into Nucleus chat chunks.
fn parse_stream_event(model: &str, event: &Value) -> Option<ChatResponse> {
    let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match event_type {
        // Preferred Responses streaming events
        "response.output_text.delta" => {
            let delta = event.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if delta.is_empty() {
                return None;
            }
            Some(ChatResponse {
                model: model.to_string(),
                content: delta.to_string(),
                done: false,
                message: Message::assistant(None, delta),
            })
        }
        "response.completed" | "response.done" | "response.incomplete" => Some(ChatResponse {
            model: model.to_string(),
            content: String::new(),
            done: true,
            message: Message::assistant(None, ""),
        }),
        // Fallback: some gateways still emit chat.completion.chunk shapes
        _ => {
            if let Some(content) = event
                .pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str())
            {
                if content.is_empty() {
                    return None;
                }
                return Some(ChatResponse {
                    model: model.to_string(),
                    content: content.to_string(),
                    done: false,
                    message: Message::assistant(None, content),
                });
            }

            // Non-stream full Responses payload (if stream was ignored)
            if let Some(text) = extract_output_text(event) {
                return Some(ChatResponse {
                    model: model.to_string(),
                    content: text.clone(),
                    done: true,
                    message: Message::assistant(None, text),
                });
            }

            None
        }
    }
}

fn extract_output_text(value: &Value) -> Option<String> {
    // Responses API: output[].content[].text where type == output_text
    if let Some(output) = value.get("output").and_then(|v| v.as_array()) {
        let mut parts = Vec::new();
        for item in output {
            if item.get("type").and_then(|v| v.as_str()) != Some("message") {
                continue;
            }
            if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                for part in content {
                    if part.get("type").and_then(|v| v.as_str()) == Some("output_text") {
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            parts.push(text.to_string());
                        }
                    }
                }
            }
        }
        if !parts.is_empty() {
            return Some(parts.join(""));
        }
    }

    // Chat Completions fallback
    value
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oauth_defaults_to_cli_proxy() {
        assert_eq!(resolve_base_url("http://localhost:11434", false), PROXY_BASE);
        assert_eq!(resolve_base_url(API_BASE, false), PROXY_BASE);
        assert_eq!(resolve_base_url(PROXY_BASE, false), PROXY_BASE);
    }

    #[test]
    fn api_key_defaults_to_public_api() {
        assert_eq!(resolve_base_url("http://localhost:11434", true), API_BASE);
        assert_eq!(resolve_base_url(PROXY_BASE, true), API_BASE);
    }

    #[test]
    fn explicit_gateway_is_preserved() {
        assert_eq!(
            resolve_base_url("https://grok-proxy.example.com/v1/", false),
            "https://grok-proxy.example.com/v1"
        );
    }

    #[test]
    fn crlf_sse_frames_yield_text() {
        let mut buffer =
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\r\n\r\n"
                .to_string();
        normalize_newlines(&mut buffer);
        let frame = pop_sse_frame(&mut buffer).unwrap();
        let data = frame.strip_prefix("data:").unwrap().trim();
        let event: Value = serde_json::from_str(data).unwrap();
        let chunk = parse_stream_event("grok-4.6", &event).unwrap();
        assert_eq!(chunk.content, "hi");
        assert!(buffer.is_empty());
    }

    #[test]
    fn reasoning_delta_is_not_answer_text() {
        let event = json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "thinking"
        });
        assert!(parse_stream_event("grok-4.6", &event).is_none());
        assert_eq!(reasoning_delta(&event), Some("thinking"));
    }
}
