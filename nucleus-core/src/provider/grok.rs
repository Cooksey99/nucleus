//! Grok / xAI provider — OAuth/API key auth + Responses API chat.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::models::EmbeddingModel;
use crate::Config;

use super::types::{ChatRequest, ChatResponse, Message, Provider, ProviderError, Result};

const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const DEVICE_URL: &str = "https://auth.x.ai/oauth2/device/code";
const TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
const SCOPE: &str = "openid profile email offline_access grok-cli:access api:access conversations:read conversations:write";
const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
const REASONING_EFFORTS: &[&str] = &["low", "medium", "high", "xhigh"];
const DEFAULT_API_BASE: &str = "https://api.x.ai/v1";

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

// Same surface Warp uses for SuperGrok OAuth and API-key traffic.
        let configured = config.llm.base_url.trim();
        let local = configured.is_empty()
            || configured.contains("localhost")
            || configured.contains("127.0.0.1")
            || configured.contains("cli-chat-proxy.grok.com");
        let base_url = if !local {
            configured.trim_end_matches('/').into()
        } else {
            DEFAULT_API_BASE.into()
        };

        let client = Client::new();

        Ok(Self {
            client,
            base_url,
            api_key,
            model: config.llm.model.clone(),
        })
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
        let response = self
            .client
            .post(&url)
            .bearer_auth(&token)
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let err = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api(format!("Grok chat failed ({status}): {err}")));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            buffer.push_str(&String::from_utf8_lossy(&chunk?));

            while let Some(idx) = buffer.find("\n\n") {
                let frame = buffer[..idx].to_string();
                buffer.drain(..=idx + 1);

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
                        continue;
                    }

                    let Ok(event) = serde_json::from_str::<Value>(data) else {
                        continue;
                    };

                    if let Some(chunk) = parse_stream_event(&model, &event) {
                        callback(chunk);
                    }
                }
            }
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
        "response.completed" | "response.done" => Some(ChatResponse {
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
