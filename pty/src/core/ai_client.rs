use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::time::Duration;

const SOCKET_PATH: &str = "/tmp/llm-workspace.sock";

#[derive(Serialize, Clone)]
pub struct Message {
    role: String,
    content: String,
}

#[derive(Default)]
pub struct ConversationHistory {
    messages: Vec<Message>,
}

impl ConversationHistory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn add_user_message(&mut self, content: String) {
        self.messages.push(Message {
            role: "user".to_string(),
            content,
        });
    }

    pub fn add_assistant_message(&mut self, content: String) {
        self.messages.push(Message {
            role: "assistant".to_string(),
            content,
        });
    }

    pub fn get_messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

#[derive(Serialize)]
struct Request {
    r#type: String,
    content: String,
    pwd: Option<String>,
    history: Option<Vec<Message>>,
}

#[derive(Deserialize)]
struct StreamChunk {
    r#type: String,
    content: String,
    error: Option<String>,
}

pub struct AiClient;

impl AiClient {
    fn strip_think_tags(text: &str) -> String {
        let mut result = String::new();
        let mut in_think = false;
        let mut i = 0;
        let bytes = text.as_bytes();
        
        while i < bytes.len() {
            if i + 7 <= bytes.len() && &bytes[i..i+7] == b"<think>" {
                in_think = true;
                i += 7;
                continue;
            }
            
            if i + 8 <= bytes.len() && &bytes[i..i+8] == b"</think>" {
                in_think = false;
                i += 8;
                continue;
            }
            
            if !in_think {
                result.push(bytes[i] as char);
            }
            
            i += 1;
        }
        
        result.trim().to_string()
    }

    pub fn send_request(
        request_type: &str,
        content: &str,
        pwd: Option<&str>,
        history: Option<&ConversationHistory>,
    ) -> Result<String> {
        let mut stream = UnixStream::connect(SOCKET_PATH)
            .context("Failed to connect to AI server. Is it running?")?;

        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(Duration::from_secs(300))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(10))).ok();

        let history_msgs = history.map(|h| h.get_messages().to_vec());

        let request = Request {
            r#type: request_type.to_string(),
            content: content.to_string(),
            pwd: pwd.map(|s| s.to_string()),
            history: history_msgs.clone(),
        };

        let json = serde_json::to_string(&request)?;
        eprintln!("[DEBUG] Sending request with {} history messages", history_msgs.as_ref().map(|h| h.len()).unwrap_or(0));
        stream.write_all(json.as_bytes())?;
        stream.write_all(b"\n")?;
        stream.flush()?;

        use std::io::BufRead;
        let buf_reader = std::io::BufReader::new(stream);
        let mut result = String::new();
        
        for line in buf_reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            
            let chunk: StreamChunk = serde_json::from_str(&line)
                .context(format!("Failed to parse chunk: {}", line))?;
            
            match chunk.r#type.as_str() {
                "chunk" => {
                    result.push_str(&chunk.content);
                }
                "done" => {
                    if !chunk.content.is_empty() {
                        result = chunk.content;
                    }
                    break;
                }
                "error" => {
                    return Err(anyhow::anyhow!(
                        "AI request failed: {}",
                        chunk.error.unwrap_or_else(|| "Unknown error".to_string())
                    ));
                }
                _ => {}
            }
        }
        
        Ok(Self::strip_think_tags(&result))
    }

    pub fn chat(query: &str, pwd: Option<&str>, history: Option<&ConversationHistory>) -> Result<String> {
        Self::send_request("chat", query, pwd, history)
    }

    pub fn edit(request: &str, pwd: Option<&str>, history: Option<&ConversationHistory>) -> Result<String> {
        Self::send_request("edit", request, pwd, history)
    }

    pub fn add_knowledge(content: &str) -> Result<String> {
        Self::send_request("add", content, None, None)
    }

    pub fn index_directory(path: &str) -> Result<String> {
        Self::send_request("index", path, None, None)
    }

    pub fn stats() -> Result<String> {
        Self::send_request("stats", "", None, None)
    }
}
