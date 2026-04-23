use crate::{Permission, Plugin, PluginError, PluginOutput};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Registry for managing plugins.
///
/// The registry is responsible for:
/// - Registering plugins with permission checking
/// - Looking up plugins by name
/// - Executing plugins
/// - Providing plugin specifications to the LLM
pub struct PluginRegistry {
    plugins: HashMap<String, Arc<Mutex<dyn Plugin + Send + Sync>>>,
    granted_permissions: Permission,
}

impl PluginRegistry {
    /// Create a new plugin registry with the given permissions.
    pub fn new(granted_permissions: Permission) -> Self {
        Self {
            plugins: HashMap::new(),
            granted_permissions,
        }
    }

    /// Register a plugin if permissions allow.
    /// Returns true if the plugin was registered, false if denied by permissions.
    pub async fn register<T: Plugin + 'static>(&mut self, plugin: T) -> bool {
        let required = plugin.required_permission();
        let plugin = Arc::new(Mutex::new(plugin));

        if !self.granted_permissions.allows(&required) {
            return false;
        }

        let plugin_name = {
            let locked_plugin = plugin.lock().await;
            locked_plugin.name().to_string()
        };
        self.plugins.insert(plugin_name, plugin);
        true
    }

    /// Get the number plugins that exist in the registry
    pub fn get_count(&self) -> usize {
        self.plugins.iter().count()
    }

    /// Get a plugin by name.
    pub fn get(&self, name: &str) -> Option<&Arc<Mutex<dyn Plugin + Send + Sync>>> {
        self.plugins.get(name)
    }

    /// Get all registered plugins.
    pub fn all(&self) -> Vec<&Arc<Mutex<dyn Plugin + Send + Sync>>> {
        self.plugins.values().collect()
    }

    /// Execute a plugin by name.
    pub async fn execute(&self, name: &str, input: Value) -> Result<PluginOutput, PluginError> {
        let plugin = self
            .get(name)
            .ok_or_else(|| PluginError::Other(format!("Unknown plugin: {}", name)))?;

        plugin.lock().await.execute(input).await
    }

    /// Get plugin specifications for the LLM.
    /// Returns a list of tool definitions in a format the LLM can understand.
    pub async fn plugin_specs(&self) -> Vec<Value> {
        let mut specs = Vec::new();
        for plugin in self.plugins.values() {
            let locked_plugin = plugin.lock().await;
            specs.push(serde_json::json!({
                "name": locked_plugin.name(),
                "description": locked_plugin.description(),
                "parameters": locked_plugin.parameter_schema(),
            }));
        }
        specs
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

        fn required_permission(&self) -> Permission {
            Permission::READ_ONLY
        }

        async fn execute(&self, _input: Value) -> crate::Result<PluginOutput> {
            Ok(PluginOutput::new("test output"))
        }
    }

    #[test]
    fn test_registry_permissions() {
        let mut registry = PluginRegistry::new(Permission::READ_ONLY);
        // let plugin = TestPlugin;

        // assert!(registry.register(plugin).await);
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_registry_permission_denial() {
        let mut registry = PluginRegistry::new(Permission::NONE);
        // let plugin = TestPlugin;

        // assert!(!registry.register(plugin));
        assert!(registry.get("test").is_none());
    }
}
