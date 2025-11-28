use crate::{Permission, Plugin, PluginError, PluginOutput};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry for managing plugins.
/// 
/// The registry is responsible for:
/// - Registering plugins with permission checking
/// - Looking up plugins by name
/// - Executing plugins
/// - Providing plugin specifications to the LLM
pub struct PluginRegistry {
    plugins: HashMap<String, Arc<dyn Plugin>>,
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
    pub fn register(&mut self, plugin: Arc<dyn Plugin>) -> bool {
        let required = plugin.required_permission();
        
        if !self.granted_permissions.allows(&required) {
            return false;
        }
        
        self.plugins.insert(plugin.name().to_string(), plugin);
        true
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
            .ok_or_else(|| PluginError::Other(format!("Unknown plugin: {}", name)))?;
        
        plugin.execute(input).await
    }
    
    /// Get plugin specifications for the LLM.
    /// Returns a list of tool definitions in a format the LLM can understand.
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
        let plugin = Arc::new(TestPlugin);
        
        assert!(registry.register(plugin.clone()));
        assert!(registry.get("test").is_some());
    }
    
    #[test]
    fn test_registry_permission_denial() {
        let mut registry = PluginRegistry::new(Permission::NONE);
        let plugin = Arc::new(TestPlugin);
        
        assert!(!registry.register(plugin));
        assert!(registry.get("test").is_none());
    }
}
