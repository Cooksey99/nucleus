# Tool System Architecture

The tool system provides a modular way to add capabilities to the AI agent. Tools are self-contained, registered automatically based on permissions, and easy to extend.

## Structure

### Core Components

- **`tools.go`**: Defines the `Tool` interface, `Registry`, and automatic tool registration
- **`file_tools.go`**: Implements file operation tools (read, write, list directory)

### Tool Interface

Each tool must implement:
```go
type Tool interface {
    Name() string
    Description() string
    Specification() api.Tool
    Execute(ctx context.Context, args json.RawMessage) (string, error)
    RequiresPermission() config.Permission
}
```

### Registration

Tools are automatically registered in `tools.NewRegistry()` and filtered based on config permissions:
- `Read`: Enables `read_file` and `list_directory`
- `Write`: Enables `write_file`
- `Command`: Reserved for shell/exec tools (future)

The registry is created once and passed to the chat manager, ensuring clean separation of concerns.

## Available Tools

### read_file
- **Permission**: Read
- **Description**: Read the contents of a file
- **Parameters**: `path` (string)

### list_directory
- **Permission**: Read
- **Description**: List files and directories in a directory
- **Parameters**: `path` (string)

### write_file
- **Permission**: Write
- **Description**: Write or update a file with new content
- **Parameters**: `path`, `content`, `reason` (all strings)

## Adding New Tools

1. Create a new struct implementing the `Tool` interface in the `tools/` package
2. Register it in `tools.NewRegistry()`:
   ```go
   r.Register(NewYourTool(cfg))
   ```
3. The registry handles permission checking automatically

**Example:**
```go
// In tools/your_tool.go
type YourTool struct {
    cfg *config.Config
}

func NewYourTool(cfg *config.Config) *YourTool {
    return &YourTool{cfg: cfg}
}

// Implement Tool interface methods...

// In tools/tools.go NewRegistry() function:
r.Register(NewYourTool(cfg))
```

## Architecture

```
server/
  └─ Creates toolRegistry and chatManager
         ↓
      tools/
        └─ Self-contained tool registration
               ↓
            chat/
              └─ Uses tool registry for LLM interactions
```

## Usage

Tools are available in `/edit` mode (ChatWithTools). The AI can:
- Read files to understand code
- List directories to explore structure
- Write files to make changes (with backup)

Regular chat mode does not have tools enabled to avoid response delays.
