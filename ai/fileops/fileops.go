// Package fileops provides file operations and LLM tool calling functionality.
package fileops

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"llm-workspace/config"
	"llm-workspace/rag"
	"llm-workspace/tools"

	"github.com/ollama/ollama/api"
)

// Handles file operations and chat with tool support.
type Manager struct {
	config       *config.Config
	client       *api.Client
	ragManager   *rag.Manager
	toolRegistry *tools.Registry
}

// Creates a new instance.
func NewManager(cfg *config.Config, client *api.Client, ragMgr *rag.Manager) *Manager {
	toolRegistry := tools.NewRegistry(cfg)
	
	toolRegistry.Register(tools.NewReadFileTool(cfg))
	toolRegistry.Register(tools.NewListDirectoryTool(cfg))
	toolRegistry.Register(tools.NewWriteFileTool(cfg))
	
	return &Manager{
		config:       cfg,
		client:       client,
		ragManager:   ragMgr,
		toolRegistry: toolRegistry,
	}
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

	toolSpecs := m.toolRegistry.GetSpecs()
	toolNames := make([]string, 0, len(toolSpecs))
	for _, spec := range toolSpecs {
		toolNames = append(toolNames, spec.Function.Name)
	}

	messages := []api.Message{
		{
			Role:    "system",
			Content: fmt.Sprintf("%s\n\nYou have access to these tools: %s", m.config.SystemPrompt, strings.Join(toolNames, ", ")),
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
			Tools:    toolSpecs,
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
			argsBytes, err := json.Marshal(toolCall.Function.Arguments)
			if err != nil {
				messages = append(messages, api.Message{
					Role:    "tool",
					Content: fmt.Sprintf("Error marshaling arguments: %v", err),
				})
				continue
			}

			result, err := m.toolRegistry.Execute(ctx, toolCall.Function.Name, argsBytes)
			if err != nil {
				result = fmt.Sprintf("Error: %v", err)
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
