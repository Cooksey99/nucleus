use crate::tool::{Plugin, PluginError, PluginOutput, Result};
use async_trait::async_trait;
use schemars::{schema_for, JsonSchema};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::HashMap, path::PathBuf};
use tokio::process::Command;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExecParams {
    /// The shell command to execute (e.g. "git status", "ls -la")
    command: String,
    /// Current working directory for command execution. Defaults to current directory if not specied.
    #[serde(default)]
    cwd: Option<PathBuf>,
    /// Additional environment variables to set for this command
    #[serde(default)]
    env: HashMap<String, String>,
}

/// Plugin that runs a shell command and returns its stdout, stderr and exit code.
pub struct ExecPlugin {}

impl Default for ExecPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecPlugin {
    /// Create the plugin.
    pub fn new() -> Self {
        Self {}
    }

    /// Run `command` with `args` in `cwd`.
    pub async fn run(
        &self,
        command: String,
        args: Vec<String>,
        cwd: PathBuf,
    ) -> Result<PluginOutput> {
        let input = serde_json::json!({
            "command": command,
            "args": args,
            "cwd": cwd
        });

        self.execute(input).await
    }
}

#[async_trait]
impl Plugin for ExecPlugin {
    fn name(&self) -> &str {
        "exec"
    }

    fn description(&self) -> &str {
        "Execute a shell command. Can run any command available in the devices shell, such as git, grep, ls, etc."
    }

    fn parameter_schema(&self) -> Value {
        let schema = schema_for!(ExecParams);
        serde_json::to_value(schema).unwrap_or_default()
    }

    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        let params: ExecParams = serde_json::from_value(input)
            .map_err(|e| PluginError::InvalidInput(format!("Invalid parameters: {}", e)))?;

        let mut command = Command::new(&params.command);
        command.envs(&params.env);
        if let Some(cwd) = &params.cwd {
            command.current_dir(cwd);
        }

        let output = match command.output().await {
            Ok(res) => {
                let stdout = String::from_utf8_lossy(&res.stdout);
                let stderr = String::from_utf8_lossy(&res.stderr);
                let exit_code = res.status.code().unwrap_or(-1);
                Ok(PluginOutput::new(format!(
                    "stdout: {}\nstderr: {}\nexit_code: {}",
                    stdout, stderr, exit_code
                )))
            }
            Err(e) => Err(PluginError::ExecutionFailed(e.to_string())),
        };

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ls_command() {
        let plugin = ExecPlugin::new();

        let input = serde_json::json!({
            "command": "ls",
            "args": ["-la"],
            "cwd": "src"
        });

        let result = plugin.execute(input).await;
        assert!(result.is_ok(), "ls with cwd succeeded")
    }
}
