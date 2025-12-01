use super::types::{Request, RequestType, StreamChunk};
use crate::{config::Config, ollama, rag};
use tokio::sync::mpsc;

pub type ChunkSender = mpsc::UnboundedSender<StreamChunk>;

/// Handles different request types and sends responses via channel.
pub struct RequestHandler {
    config: Config,
    ollama_client: ollama::Client,
    rag_manager: rag::Manager,
}

impl RequestHandler {
    pub fn new(config: Config, ollama_client: ollama::Client) -> Self {
        // Use persistent storage for RAG
        let rag_manager = rag::Manager::with_persistence(&config, ollama_client.clone());
        
        Self {
            config,
            ollama_client,
            rag_manager,
        }
    }
    
    /// Loads previously indexed documents from persistent storage.
    pub async fn load_rag(&self) -> Result<usize, rag::RagError> {
        self.rag_manager.load().await
    }
    
    /// Routes request to appropriate handler based on type.
    pub async fn handle(&self, request: Request, sender: ChunkSender) {
        match request.request_type {
            RequestType::Chat | RequestType::Edit => {
                self.handle_chat(request, sender).await
            }
            RequestType::Add => self.handle_add(request, sender).await,
            RequestType::Index => self.handle_index(request, sender).await,
            RequestType::Stats => self.handle_stats(sender).await,
        }
    }
    
    async fn handle_chat(&self, request: Request, sender: ChunkSender) {
        let messages = self.build_messages(request);
        
        let chat_request = ollama::ChatRequest::new(&self.config.llm.model, messages)
            .with_temperature(self.config.llm.temperature);
        
        let mut full_response = String::new();
        
        let result = self.ollama_client.chat(chat_request, |response| {
            if !response.message.content.is_empty() {
                full_response.push_str(&response.message.content);
                let _ = sender.send(StreamChunk::chunk(&response.message.content));
            }
        }).await;
        
        match result {
            Ok(_) => {
                let _ = sender.send(StreamChunk::done(&full_response));
            }
            Err(e) => {
                let _ = sender.send(StreamChunk::error(e.to_string()));
            }
        }
    }
    
    async fn handle_add(&self, request: Request, sender: ChunkSender) {
        match self.rag_manager.add_knowledge(&request.content, "user_input").await {
            Ok(_) => {
                let _ = sender.send(StreamChunk::done("Added to knowledge base"));
            }
            Err(e) => {
                let _ = sender.send(StreamChunk::error(format!("Failed to add: {}", e)));
            }
        }
    }
    
    async fn handle_index(&self, request: Request, sender: ChunkSender) {
        match self.rag_manager.index_directory(&request.content).await {
            Ok(count) => {
                let _ = sender.send(StreamChunk::done(format!(
                    "Indexed {} files from: {}",
                    count, request.content
                )));
            }
            Err(e) => {
                let _ = sender.send(StreamChunk::error(format!("Failed to index: {}", e)));
            }
        }
    }
    
    async fn handle_stats(&self, sender: ChunkSender) {
        let count = self.rag_manager.count();
        let _ = sender.send(StreamChunk::done(format!(
            "Knowledge base contains {} documents",
            count
        )));
    }
    
    fn build_messages(&self, request: Request) -> Vec<ollama::Message> {
        let mut messages = vec![ollama::Message::system(&self.config.system_prompt)];
        
        if let Some(history) = request.history {
            for msg in history {
                messages.push(ollama::Message {
                    role: "user".to_string(),
                    content: msg.content.clone(),
                    images: None,
                    tool_calls: None,
                });
            }
        }
        
        messages.push(ollama::Message::user(&request.content));
        messages
    }
}
