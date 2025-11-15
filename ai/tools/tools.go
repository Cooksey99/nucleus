package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"llm-workspace/config"

	"github.com/ollama/ollama/api"
)

type Tool interface {
	Name() string
	Description() string
	Specification() api.Tool
	Execute(ctx context.Context, args json.RawMessage) (string, error)
	RequiresPermission() config.Permission
}

type Registry struct {
	tools map[string]Tool
	cfg   *config.Config
}

func NewRegistry(cfg *config.Config) *Registry {
	r := &Registry{
		tools: make(map[string]Tool),
		cfg:   cfg,
	}
	
	r.Register(NewReadFileTool(cfg))
	r.Register(NewListDirectoryTool(cfg))
	r.Register(NewWriteFileTool(cfg))
	
	return r
}

func (r *Registry) Register(tool Tool) {
	perm := tool.RequiresPermission()
	
	if perm.Read && !r.cfg.Permission.Read {
		return
	}
	if perm.Write && !r.cfg.Permission.Write {
		return
	}
	if perm.Command && !r.cfg.Permission.Command {
		return
	}
	
	r.tools[tool.Name()] = tool
}

func (r *Registry) Get(name string) (Tool, bool) {
	tool, exists := r.tools[name]
	return tool, exists
}

func (r *Registry) GetAll() []Tool {
	tools := make([]Tool, 0, len(r.tools))
	for _, tool := range r.tools {
		tools = append(tools, tool)
	}
	return tools
}

func (r *Registry) GetSpecs() []api.Tool {
	specs := make([]api.Tool, 0, len(r.tools))
	for _, tool := range r.tools {
		specs = append(specs, tool.Specification())
	}
	return specs
}

func (r *Registry) Execute(ctx context.Context, name string, args json.RawMessage) (string, error) {
	tool, exists := r.Get(name)
	if !exists {
		return "", fmt.Errorf("unknown tool: %s", name)
	}
	return tool.Execute(ctx, args)
}
