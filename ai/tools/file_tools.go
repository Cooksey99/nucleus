package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"llm-workspace/config"

	"github.com/ollama/ollama/api"
)

type ReadFileTool struct {
	cfg *config.Config
}

func NewReadFileTool(cfg *config.Config) *ReadFileTool {
	return &ReadFileTool{cfg: cfg}
}

func (t *ReadFileTool) Name() string {
	return "read_file"
}

func (t *ReadFileTool) Description() string {
	return "Read the contents of a file"
}

func (t *ReadFileTool) Specification() api.Tool {
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"path"},
				Properties: map[string]api.ToolProperty{
					"path": {
						Type:        api.PropertyType{"string"},
						Description: "Absolute path to the file",
					},
				},
			},
		},
	}
}

func (t *ReadFileTool) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	var params struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	content, err := os.ReadFile(params.Path)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	fmt.Printf("\n📖 Read file: %s\n", params.Path)
	return string(content), nil
}

func (t *ReadFileTool) RequiresPermission() config.Permission {
	return config.Permission{Read: true, Write: false, Command: false}
}

type ListDirectoryTool struct {
	cfg *config.Config
}

func NewListDirectoryTool(cfg *config.Config) *ListDirectoryTool {
	return &ListDirectoryTool{cfg: cfg}
}

func (t *ListDirectoryTool) Name() string {
	return "list_directory"
}

func (t *ListDirectoryTool) Description() string {
	return "List files and directories in a directory"
}

func (t *ListDirectoryTool) Specification() api.Tool {
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"path"},
				Properties: map[string]api.ToolProperty{
					"path": {
						Type:        api.PropertyType{"string"},
						Description: "Absolute path to the directory",
					},
				},
			},
		},
	}
}

func (t *ListDirectoryTool) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	var params struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	entries, err := os.ReadDir(params.Path)
	if err != nil {
		return "", fmt.Errorf("failed to read directory: %w", err)
	}

	var result strings.Builder
	for _, entry := range entries {
		if entry.IsDir() {
			result.WriteString(entry.Name() + "/\n")
		} else {
			result.WriteString(entry.Name() + "\n")
		}
	}

	fmt.Printf("\n📁 Listed directory: %s\n", params.Path)
	return result.String(), nil
}

func (t *ListDirectoryTool) RequiresPermission() config.Permission {
	return config.Permission{Read: true, Write: false, Command: false}
}

type WriteFileTool struct {
	cfg *config.Config
}

func NewWriteFileTool(cfg *config.Config) *WriteFileTool {
	return &WriteFileTool{cfg: cfg}
}

func (t *WriteFileTool) Name() string {
	return "write_file"
}

func (t *WriteFileTool) Description() string {
	return "Write or update a file with new content"
}

func (t *WriteFileTool) Specification() api.Tool {
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters: api.ToolFunctionParameters{
				Type:     "object",
				Required: []string{"path", "content", "reason"},
				Properties: map[string]api.ToolProperty{
					"path": {
						Type:        api.PropertyType{"string"},
						Description: "Absolute path to the file",
					},
					"content": {
						Type:        api.PropertyType{"string"},
						Description: "Complete new content of the file",
					},
					"reason": {
						Type:        api.PropertyType{"string"},
						Description: "Explanation of why this change is being made",
					},
				},
			},
		},
	}
}

func (t *WriteFileTool) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	var params struct {
		Path    string `json:"path"`
		Content string `json:"content"`
		Reason  string `json:"reason"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	info, err := os.Stat(params.Path)
	if err == nil {
		backupPath := params.Path + ".backup"
		oldContent, _ := os.ReadFile(params.Path)
		err = os.WriteFile(backupPath, oldContent, info.Mode())
		if err == nil {
			fmt.Printf("Backup created: %s\n", backupPath)
		}
	}

	err = os.WriteFile(params.Path, []byte(params.Content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}

	fmt.Printf("File updated: %s\n", params.Path)
	fmt.Printf("Reason: %s\n", params.Reason)
	return "File successfully updated", nil
}

func (t *WriteFileTool) RequiresPermission() config.Permission {
	return config.Permission{Read: false, Write: true, Command: false}
}
