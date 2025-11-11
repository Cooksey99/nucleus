package fileops

import (
	"testing"

	"llm-workspace/config"
	"llm-workspace/rag"
	"llm-workspace/tools"

	"github.com/ollama/ollama/api"
)

func TestNewManager_InitializesToolRegistry(t *testing.T) {
	cfg := &config.Config{
		Permission: config.Permission{
			Read:    true,
			Write:   true,
			Command: false,
		},
	}
	client, _ := api.ClientFromEnvironment()
	
	mockRAG, _ := rag.NewManager(cfg, client)
	
	manager := NewManager(cfg, client, mockRAG)
	
	if manager == nil {
		t.Fatal("expected manager to be non-nil")
	}
	
	if manager.toolRegistry == nil {
		t.Fatal("expected toolRegistry to be initialized")
	}
	
	if manager.config != cfg {
		t.Error("expected config to be set")
	}
	
	if manager.client != client {
		t.Error("expected client to be set")
	}
	
	if manager.ragManager != mockRAG {
		t.Error("expected ragManager to be set")
	}
}

func TestNewManager_RegistersDefaultTools(t *testing.T) {
	cfg := &config.Config{
		Permission: config.Permission{
			Read:    true,
			Write:   true,
			Command: false,
		},
	}
	client, _ := api.ClientFromEnvironment()
	mockRAG, _ := rag.NewManager(cfg, client)
	
	manager := NewManager(cfg, client, mockRAG)
	
	expectedTools := []string{"read_file", "list_directory", "write_file"}
	
	for _, toolName := range expectedTools {
		tool, exists := manager.toolRegistry.Get(toolName)
		if !exists {
			t.Errorf("expected tool %s to be registered", toolName)
		}
		if tool == nil {
			t.Errorf("expected tool %s to be non-nil", toolName)
		}
	}
}

func TestToolRegistry_RegistersAndRetrievesTools(t *testing.T) {
	cfg := &config.Config{
		Permission: config.Permission{
			Read:    true,
			Write:   true,
			Command: false,
		},
	}
	
	registry := tools.NewRegistry(cfg)
	
	readTool := tools.NewReadFileTool(cfg)
	writeTool := tools.NewWriteFileTool(cfg)
	listTool := tools.NewListDirectoryTool(cfg)
	
	registry.Register(readTool)
	registry.Register(writeTool)
	registry.Register(listTool)
	
	testCases := []struct {
		name         string
		toolName     string
		shouldExist  bool
		expectedType interface{}
	}{
		{
			name:         "ReadFileTool exists",
			toolName:     "read_file",
			shouldExist:  true,
			expectedType: (*tools.ReadFileTool)(nil),
		},
		{
			name:         "WriteFileTool exists",
			toolName:     "write_file",
			shouldExist:  true,
			expectedType: (*tools.WriteFileTool)(nil),
		},
		{
			name:         "ListDirectoryTool exists",
			toolName:     "list_directory",
			shouldExist:  true,
			expectedType: (*tools.ListDirectoryTool)(nil),
		},
		{
			name:         "Non-existent tool",
			toolName:     "non_existent_tool",
			shouldExist:  false,
			expectedType: nil,
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tool, exists := registry.Get(tc.toolName)
			
			if exists != tc.shouldExist {
				t.Errorf("expected exists=%v, got %v", tc.shouldExist, exists)
			}
			
			if tc.shouldExist && tool == nil {
				t.Error("expected tool to be non-nil")
			}
			
			if !tc.shouldExist && tool != nil {
				t.Error("expected tool to be nil for non-existent tool")
			}
		})
	}
}

