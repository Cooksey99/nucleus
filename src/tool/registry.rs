use super::{Plugin, PluginError, PluginOutput};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry for managing plugins.
///
/// Registration is synchronous and always succeeds. Approval of a call is a
/// later hook, not a permission checked at registration.
pub struct PluginRegistry {
    plugins: HashMap<String, Arc<dyn Plugin>>,
}

impl PluginRegistry {
    /// Create an empty plugin registry.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a plugin. Replaces any plugin already registered under the same name.
    pub fn register<T: Plugin + 'static>(&mut self, plugin: T) {
        let plugin = Arc::new(plugin);
        let name = plugin.name().to_string();
        self.plugins.insert(name, plugin);
    }

    /// Get the number of plugins in the registry.
    pub fn get_count(&self) -> usize {
        self.plugins.len()
    }

    /// Get a plugin by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Plugin>> {
        self.plugins.get(name)
    }

    /// Get all registered plugins.
    pub fn all(&self) -> Vec<&Arc<dyn Plugin>> {
        self.plugins.values().collect()
    }

    /// Execute a plugin by name.
    pub async fn execute(&self, name: &str, input: Value) -> Result<PluginOutput, PluginError> {
        let plugin = self
            .get(name)
            .ok_or_else(|| PluginError::Other(format!("Unknown plugin: {name}")))?;

        plugin.execute(input).await
    }

    /// Get plugin specifications for the LLM.
    pub fn plugin_specs(&self) -> Vec<Value> {
        self.plugins
            .values()
            .map(|plugin| {
                serde_json::json!({
                    "name": plugin.name(),
                    "description": plugin.description(),
                    "parameters": plugin.parameter_schema(),
                })
            })
            .collect()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct TestPlugin;

    #[async_trait]
    impl Plugin for TestPlugin {
        fn name(&self) -> &str {
            "test"
        }

        fn description(&self) -> &str {
            "A test plugin"
        }

        fn parameter_schema(&self) -> Value {
            serde_json::json!({})
        }

        async fn execute(&self, _input: Value) -> crate::tool::Result<PluginOutput> {
            Ok(PluginOutput::new("test output"))
        }
    }

    #[test]
    fn test_registry_registers() {
        let mut registry = PluginRegistry::new();
        registry.register(TestPlugin);
        assert!(registry.get("test").is_some());
    }
}
