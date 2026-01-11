use async_trait::async_trait;
use nucleus_core::patterns;
use nucleus_plugin::{Permission, Plugin, PluginError, PluginOutput, Result};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use walkdir::WalkDir;

pub struct SearchPlugin;

#[derive(Debug, Deserialize)]
struct SearchParams {
    query: String,
    path: Option<String>,
    #[serde(default)]
    regex: bool,
    #[serde(default)]
    case_sensitive: bool,
    #[serde(default = "default_max_results")]
    max_results: usize,
    #[serde(default = "default_exclude_patterns")]
    exclude_patterns: Vec<String>,
}

fn default_max_results() -> usize {
    100
}

fn default_exclude_patterns() -> Vec<String> {
    patterns::default_exclude_patterns()
}

impl SearchPlugin {
    pub fn new() -> Self {
        Self
    }
}
#[async_trait]
impl Plugin for SearchPlugin {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search for text patterns in files and/or directories"
    }

    fn parameter_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text or regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (defaults to current directory)"
                },
                "regex": {
                    "type": "boolean",
                    "description": "Treat query as regex pattern",
                    "default": false
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case sensitive search",
                    "default": false
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum number of results to return",
                    "default": 100
                }
            }
        })
    }

    fn required_permission(&self) -> Permission {
        Permission::READ_ONLY
    }

    async fn execute(&self, input: Value) -> Result<PluginOutput> {
        let params: SearchParams = serde_json::from_value(input)
            .map_err(|e| PluginError::InvalidInput(format!("Invalid parameters: {}", e)))?;

        let search_path = params
            .path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        let matcher = if params.regex {
            let pattern = if params.case_sensitive {
                &params.query
            } else {
                &format!("(?i){}", params.query)
            };
            Regex::new(pattern)
                .map_err(|e| PluginError::InvalidInput(format!("Invalid regex: {}", e)))?
        } else {
            let escaped = regex::escape(&params.query);
            let pattern = if params.case_sensitive {
                escaped
            } else {
                format!("(?i){}", escaped)
            };
            Regex::new(&pattern).unwrap()
        };

        let mut results = Vec::new();
        let mut count = 0;

        for entry in WalkDir::new(&search_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }

            if count >= params.max_results {
                break;
            }

            let path = entry.path();

            if should_skip(path, &params.exclude_patterns) {
                continue;
            }

            if let Ok(content) = tokio::fs::read_to_string(path).await {
                for (line_num, line) in content.lines().enumerate() {
                    if matcher.is_match(line) {
                        results.push(serde_json::json!({
                            "file": path.display().to_string(),
                            "line": line_num + 1,
                            "content": line.trim()
                        }));
                        count += 1;

                        if count >= params.max_results {
                            break;
                        }
                    }
                }
            }
        }

        let result_json = serde_json::json!({
            "summary": format!("Found {} matches", results.len()),
            "results": results
        });

        Ok(PluginOutput::new(
            serde_json::to_string_pretty(&result_json).unwrap(),
        ))
    }
}

fn should_skip(path: &std::path::Path, exclude_patterns: &[String]) -> bool {
    nucleus_core::patterns::should_exclude(path, exclude_patterns)
}
