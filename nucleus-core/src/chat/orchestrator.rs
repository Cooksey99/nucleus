use anyhow::Result;
use std::collections::HashMap;

use super::ChatManager;

pub type AgentId = String;

pub struct AgentOrchestrator {
    agents: HashMap<AgentId, ChatManager>,
}

impl AgentOrchestrator {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    pub fn register(&mut self, id: impl Into<String>, manager: ChatManager) {
        self.agents.insert(id.into(), manager);
    }

    pub fn get(&self, id: &str) -> Option<&ChatManager> {
        self.agents.get(id)
    }

    pub fn remove(&mut self, id: &str) -> Option<ChatManager> {
        self.agents.remove(id)
    }

    pub fn list_ids(&self) -> Vec<&str> {
        self.agents.keys().map(|s| s.as_str()).collect()
    }

    pub async fn query(&self, agent_id: &str, message: &str) -> Result<String> {
        let manager = self.agents.get(agent_id)
            .ok_or_else(|| anyhow::anyhow!("Agent not found: {}", agent_id))?;
        manager.query(message).await
    }
}
