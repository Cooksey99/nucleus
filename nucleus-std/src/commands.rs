use async_trait::async_trait;
use nucleus_plugin::{Permission, Plugin, PluginError, PluginOutput, Result};
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use tokio::process::Command;

#[derive(Debug, Deserialize)]
pub struct ExecParams {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    cwd: PathBuf,
}

pub struct ExecPlugin {
    /// Allow potentially dangerous or risky commands to be executed
    allow_dangerous: bool,
}

impl ExecPlugin {
    pub fn new() -> Self {
        Self {
            allow_dangerous: false,
        }
    }

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
        serde_json::json!({
            "type": "object",
            "required": ["command", "cwd"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute (e.g., 'git', 'grep', 'ls')"
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command arguments"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory"
                }
            }
        })
    }

    fn required_permission(&self) -> Permission {
        Permission::ALL
    }

    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        let params: ExecParams = serde_json::from_value(input)
            .map_err(|e| PluginError::InvalidInput(format!("Invalid parameters: {}", e)))?;

        // Do security checks

        let mut command = Command::new(&params.command);
        command.args(&params.args);
        command.current_dir(&params.cwd);

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
