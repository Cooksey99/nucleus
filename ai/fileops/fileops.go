// Package fileops provides file operations and LLM tool calling functionality.
package fileops

import (
	"context"
	"fmt"
	"log"
	"regexp"
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

type ParsedToolCall struct {
	Name      string
	Arguments string
}

// Parses tool calls from text output
func parseToolCalls(text string) []ParsedToolCall {
	var calls []ParsedToolCall

	// Match <tool_call>...</tool_call> blocks
	toolCallRegex := regexp.MustCompile(`<tool_call>\s*<name>([^<]+)</name>\s*<arguments>({[^}]+})</arguments>\s*</tool_call>`)
	matches := toolCallRegex.FindAllStringSubmatch(text, -1)

	for _, match := range matches {
		if len(match) == 3 {
			calls = append(calls, ParsedToolCall{
				Name:      strings.TrimSpace(match[1]),
				Arguments: strings.TrimSpace(match[2]),
			})
		}
	}

	return calls
}

// Removes tool call XML from text
func removeToolCalls(text string) string {
	toolCallRegex := regexp.MustCompile(`<tool_call>.*?</tool_call>`)
	return strings.TrimSpace(toolCallRegex.ReplaceAllString(text, ""))
}

// Callback function type for streaming responses
type StreamCallback func(chunk string)

// Sends a message with file read/write tools enabled.
// The LLM can request to read or modify files as needed.
func (m *Manager) ChatWithTools(ctx context.Context, userMessage string) (string, error) {
	return m.ChatWithToolsStream(ctx, userMessage, nil, nil)
}

// ChatWithToolsStream allows streaming responses via callback
func (m *Manager) ChatWithToolsStream(ctx context.Context, userMessage string, history []api.Message, streamCallback StreamCallback) (string, error) {
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

	// Build tool descriptions
	var toolDescriptions strings.Builder
	for _, spec := range toolSpecs {
		toolDescriptions.WriteString(fmt.Sprintf("\n- %s: %s", spec.Function.Name, spec.Function.Description))
		if len(spec.Function.Parameters.Required) > 0 {
			toolDescriptions.WriteString(fmt.Sprintf(" (required params: %s)", strings.Join(spec.Function.Parameters.Required, ", ")))
		}
	}

	systemPrompt := fmt.Sprintf(`%s

=== CRITICAL TOOL CALLING RULES ===
You have access to these tools:%s

When you need to use a tool, IMMEDIATELY output this XML format - DO NOT THINK, JUST DO IT:
<tool_call>
<name>tool_name</name>
<arguments>{"param1": "value1"}</arguments>
</tool_call>

RULES:
1. If user asks about a file -> IMMEDIATELY call read_file or list_directory
2. Output ONLY the tool call XML - nothing before or after
3. After tool results arrive, THEN provide your final answer
4. Default working directory: /Users/cooksey/development/llm-workspace
5. Use absolute paths: /Users/cooksey/development/llm-workspace/README.md

EXAMPLES:
User: "What's in README.md?"
You: <tool_call>
<name>read_file</name>
<arguments>{"path": "/Users/cooksey/development/llm-workspace/README.md"}</arguments>
</tool_call>

User: "List files"
You: <tool_call>
<name>list_directory</name>
<arguments>{"path": "/Users/cooksey/development/llm-workspace"}</arguments>
</tool_call>`, m.config.SystemPrompt, toolDescriptions.String())

	messages := []api.Message{
		{
			Role:    "system",
			Content: systemPrompt,
		},
	}

	// Append conversation history if provided
	log.Printf("[DEBUG] Received %d history messages", len(history))
	if len(history) > 0 {
		for i, msg := range history {
			log.Printf("[DEBUG] History[%d]: %s: %s", i, msg.Role, msg.Content[:min(50, len(msg.Content))])
		}
		messages = append(messages, history...)
	}

	// Append current user message
	messages = append(messages, api.Message{
		Role:    "user",
		Content: userMessageWithContext,
	})

	for {
		req := &api.ChatRequest{
			Model:    m.config.LLM.Model,
			Messages: messages,
			Options: map[string]any{
				"temperature": m.config.LLM.Temperature,
			},
		}

		var responseBuilder strings.Builder
		err = m.client.Chat(ctx, req, func(resp api.ChatResponse) error {
			if resp.Message.Content != "" {
				// Stream to callback if provided
				if streamCallback != nil {
					streamCallback(resp.Message.Content)
				}
				responseBuilder.WriteString(resp.Message.Content)
			}
			return nil
		})

		if err != nil {
			return "", fmt.Errorf("chat failed: %w", err)
		}

		fullResponse := responseBuilder.String()

		// Parse tool calls from response text
		toolCalls := parseToolCalls(fullResponse)
		log.Printf("[DEBUG] Parsed %d tool calls from response", len(toolCalls))

		// Add assistant message to history
		messages = append(messages, api.Message{
			Role:    "assistant",
			Content: fullResponse,
		})

		// If no tool calls, return the response (minus any tool call XML)
		if len(toolCalls) == 0 {
			cleanResponse := removeToolCalls(fullResponse)
			log.Printf("[DEBUG] Returning response, length: %d", len(cleanResponse))
			return cleanResponse, nil
		}

		// Execute tool calls
		log.Printf("[DEBUG] Executing %d tool calls", len(toolCalls))
		for _, toolCall := range toolCalls {
			log.Printf("[DEBUG] Calling tool: %s with args: %s", toolCall.Name, toolCall.Arguments)

			result, err := m.toolRegistry.Execute(ctx, toolCall.Name, []byte(toolCall.Arguments))
			if err != nil {
				result = fmt.Sprintf("Error: %v", err)
				log.Printf("[DEBUG] Tool execution error: %v", err)
			} else {
				log.Printf("[DEBUG] Tool result length: %d", len(result))
			}

			// Add tool result to message history
			messages = append(messages, api.Message{
				Role:    "user",
				Content: fmt.Sprintf("Tool '%s' result:\n%s", toolCall.Name, result),
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
