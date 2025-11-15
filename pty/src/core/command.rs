use super::ai_client::{AiClient, ConversationHistory};
use anyhow::Result;

pub enum Command {
    AiChat(String),
    AiEdit(String),
    AiStats,
    AiAdd(String),
    AiIndex(String),
    ClearHistory,
    PassThrough,
}

impl Command {
    pub fn parse(line: &str) -> Self {
        let trimmed = line.trim();
        
        let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
        
        match parts.as_slice() {
            ["/ai", rest] => Command::AiChat(rest.to_string()),
            ["/edit", rest] => Command::AiEdit(rest.to_string()),
            ["/add", rest] => Command::AiAdd(rest.to_string()),
            ["/index", rest] => Command::AiIndex(rest.to_string()),
            ["/stats"] => Command::AiStats,
            ["/clear"] => Command::ClearHistory,
            _ => Command::PassThrough,
        }
    }
    
    pub fn execute(&self, pwd: Option<&str>, history: Option<&mut ConversationHistory>) -> Result<Option<String>> {
        match self {
            Command::AiChat(query) => {
                let response = AiClient::chat(query, pwd, history.as_ref().map(|h| &**h))?;
                if let Some(hist) = history {
                    eprintln!("[DEBUG] Before adding to history: {} messages", hist.get_messages().len());
                    hist.add_user_message(query.clone());
                    hist.add_assistant_message(response.clone());
                    eprintln!("[DEBUG] After adding to history: {} messages", hist.get_messages().len());
                }
                Ok(Some(response))
            }
            Command::AiEdit(request) => {
                let response = AiClient::edit(request, pwd, history.as_ref().map(|h| &**h))?;
                if let Some(hist) = history {
                    hist.add_user_message(request.clone());
                    hist.add_assistant_message(response.clone());
                }
                Ok(Some(response))
            }
            Command::AiStats => {
                let response = AiClient::stats()?;
                Ok(Some(response))
            }
            Command::AiAdd(content) => {
                let response = AiClient::add_knowledge(content)?;
                Ok(Some(response))
            }
            Command::AiIndex(path) => {
                let response = AiClient::index_directory(path)?;
                Ok(Some(response))
            }
            Command::ClearHistory => {
                if let Some(hist) = history {
                    hist.clear();
                }
                Ok(Some("Conversation history cleared".to_string()))
            }
            Command::PassThrough => Ok(None),
        }
    }
}
