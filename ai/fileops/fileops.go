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
	log.Printf("[DEBUG] Registered %d tools", len(toolSpecs))
	toolNames := make([]string, 0, len(toolSpecs))
	for _, spec := range toolSpecs {
		toolNames = append(toolNames, spec.Function.Name)
		log.Printf("[DEBUG] Tool: %s - %s", spec.Function.Name, spec.Function.Description)
	}

	systemPrompt := fmt.Sprintf(`%s

IMPORTANT: You have access to the following tools that you MUST use when appropriate:
- read_file: Use this to read file contents. You must call this tool to see file contents.
- write_file: Use this to create or modify files
- list_directory: Use this to see what files exist in a directory

When a user asks about file contents, you MUST call the appropriate tool. Do not pretend or say you will read a file - actually call the tool.
Available tools: %s`, m.config.SystemPrompt, strings.Join(toolNames, ", "))

	messages := []api.Message{
		{
			Role:    "system",
			Content: systemPrompt,
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
		var responseBuilder strings.Builder
		err = m.client.Chat(ctx, req, func(resp api.ChatResponse) error {
			currentMsg = resp.Message
			if resp.Message.Content != "" {
				fmt.Print(resp.Message.Content)
				responseBuilder.WriteString(resp.Message.Content)
			}
			return nil
		})

		if err != nil {
			return "", fmt.Errorf("chat failed: %w", err)
		}

		log.Printf("[DEBUG] Tool calls: %d, Content length: %d", len(currentMsg.ToolCalls), len(responseBuilder.String()))

		messages = append(messages, currentMsg)

		if len(currentMsg.ToolCalls) == 0 {
			finalResponse := responseBuilder.String()
			if finalResponse == "" {
				finalResponse = currentMsg.Content
			}
			log.Printf("[DEBUG] Returning response, length: %d", len(finalResponse))
			return finalResponse, nil
		}

		log.Printf("[DEBUG] Executing %d tool calls", len(currentMsg.ToolCalls))
		for _, toolCall := range currentMsg.ToolCalls {
			log.Printf("[DEBUG] Calling tool: %s", toolCall.Function.Name)
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
				log.Printf("[DEBUG] Tool execution error: %v", err)
			} else {
				log.Printf("[DEBUG] Tool result length: %d", len(result))
			}

			messages = append(messages, api.Message{
				Role:    "tool",
				Content: result,
			})
		}
		log.Printf("[DEBUG] Continuing loop to process tool results...")
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
