// Package fileops provides file operations and LLM tool calling functionality.
package fileops

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"llm-workspace/config"
	"llm-workspace/rag"

	"github.com/ollama/ollama/api"
)

// Handles file operations and chat with tool support.
type Manager struct {
	config     *config.Config
	client     *api.Client
	ragManager *rag.Manager
}

// Creates a new instance.
func NewManager(cfg *config.Config, client *api.Client, ragMgr *rag.Manager) *Manager {
	return &Manager{
		config:     cfg,
		client:     client,
		ragManager: ragMgr,
	}
}

// Reads and returns file contents.
func (m *Manager) ReadFile(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}
	return string(content), nil
}

// Writes content to a file, creating a backup if it exists.
func (m *Manager) WriteFile(path, content, reason string) error {
	info, err := os.Stat(path)
	if err == nil {
		backupPath := path + ".backup"
		oldContent, _ := os.ReadFile(path)
		err = os.WriteFile(backupPath, oldContent, info.Mode())
		if err == nil {
			fmt.Printf("💾 Backup created: %s\n", backupPath)
		}
	}

	err = os.WriteFile(path, []byte(content), 0644)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	fmt.Printf("File updated: %s\n", path)
	fmt.Printf("Reason: %s\n", reason)
	return nil
}

// Sends a message with file read/write tools enabled.
// The LLM can request to read or modify files as needed.
func (m *Manager) ChatWithTools(ctx context.Context, userMessage string) (string, error) {
	relevantContext, err := m.ragManager.RetrieveContext(ctx, userMessage)
	if err != nil {
		log.Printf("Warning: retrieval failed: %v", err)
	}

	userMessageWithContext := userMessage
	if relevantContext != "" {
		userMessageWithContext = userMessage + relevantContext
	}

	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "read_file",
				Description: "Read the contents of a file",
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
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "write_file",
				Description: "Write or update a file with new content",
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
		},
	}

	messages := []api.Message{
		{
			Role:    "system",
			Content: m.config.SystemPrompt + "\n\nYou have access to read_file and write_file functions. Use them to read and modify code files when requested.",
		},
		{
			Role:    "user",
			Content: userMessageWithContext,
		},
	}

	for {
		req := &api.ChatRequest{
			Model:    m.config.LLM.Model,
			Messages: messages,
			Tools:    tools,
			Options: map[string]any{
				"temperature": m.config.LLM.Temperature,
			},
		}

		var currentMsg api.Message
		err = m.client.Chat(ctx, req, func(resp api.ChatResponse) error {
			currentMsg = resp.Message
			if resp.Message.Content != "" {
				fmt.Print(resp.Message.Content)
			}
			return nil
		})

		if err != nil {
			return "", fmt.Errorf("chat failed: %w", err)
		}

		messages = append(messages, currentMsg)

		if len(currentMsg.ToolCalls) == 0 {
			return currentMsg.Content, nil
		}

		for _, toolCall := range currentMsg.ToolCalls {
			var result string

			switch toolCall.Function.Name {
			case "read_file":
				var args struct {
					Path string `json:"path"`
				}
				argsBytes, err := json.Marshal(toolCall.Function.Arguments)
				if err != nil {
					result = fmt.Sprintf("Error marshaling arguments: %v", err)
					break
				}
				if err := json.Unmarshal(argsBytes, &args); err != nil {
					result = fmt.Sprintf("Error parsing arguments: %v", err)
				} else {
					content, err := m.ReadFile(args.Path)
					if err != nil {
						result = fmt.Sprintf("Error: %v", err)
					} else {
						result = content
						fmt.Printf("\n📖 Read file: %s\n", args.Path)
					}
				}

			case "write_file":
				var args struct {
					Path    string `json:"path"`
					Content string `json:"content"`
					Reason  string `json:"reason"`
				}
				argsBytes, err := json.Marshal(toolCall.Function.Arguments)
				if err != nil {
					result = fmt.Sprintf("Error marshaling arguments: %v", err)
					break
				}
				if err := json.Unmarshal(argsBytes, &args); err != nil {
					result = fmt.Sprintf("Error parsing arguments: %v", err)
				} else {
					if err := m.WriteFile(args.Path, args.Content, args.Reason); err != nil {
						result = fmt.Sprintf("Error: %v", err)
					} else {
						result = "File successfully updated"
					}
				}

			default:
				result = fmt.Sprintf("Unknown function: %s", toolCall.Function.Name)
			}

			messages = append(messages, api.Message{
				Role:    "tool",
				Content: result,
			})
		}
	}
}

// Sends a message without tool calling enabled.
// Retrieves relevant context from RAG before generating a response.
func (m *Manager) Chat(ctx context.Context, userMessage string) (string, error) {
	relevantContext, err := m.ragManager.RetrieveContext(ctx, userMessage)
	if err != nil {
		log.Printf("Warning: retrieval failed: %v", err)
	}

	userMessageWithContext := userMessage
	if relevantContext != "" {
		userMessageWithContext = userMessage + relevantContext
	}

	messages := []api.Message{
		{
			Role:    "system",
			Content: m.config.SystemPrompt,
		},
		{
			Role:    "user",
			Content: userMessageWithContext,
		},
	}

	req := &api.ChatRequest{
		Model:    m.config.LLM.Model,
		Messages: messages,
		Options: map[string]any{
			"temperature": m.config.LLM.Temperature,
		},
	}

	var response strings.Builder
	err = m.client.Chat(ctx, req, func(resp api.ChatResponse) error {
		response.WriteString(resp.Message.Content)
		return nil
	})

	if err != nil {
		return "", fmt.Errorf("chat failed: %w", err)
	}

	return response.String(), nil
}
