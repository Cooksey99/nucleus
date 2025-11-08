package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
	chromem "github.com/philippgille/chromem-go"
	"gopkg.in/yaml.v3"
)

type Config struct {
	LLM struct {
		Model         string  `yaml:"model"`
		BaseURL       string  `yaml:"base_url"`
		Temperature   float64 `yaml:"temperature"`
		ContextLength int     `yaml:"context_length"`
	} `yaml:"llm"`
	SystemPrompt     string `yaml:"system_prompt"`
	RAG              RAGConfig
	Storage          StorageConfig
	Personalization  PersonalizationConfig
}

type RAGConfig struct {
	EmbeddingModel string `yaml:"embedding_model"`
	ChunkSize      int    `yaml:"chunk_size"`
	ChunkOverlap   int    `yaml:"chunk_overlap"`
	TopK           int    `yaml:"top_k"`
}

type StorageConfig struct {
	VectorDBPath     string `yaml:"vector_db_path"`
	ChatHistoryPath  string `yaml:"chat_history_path"`
}

type PersonalizationConfig struct {
	LearnFromInteractions bool   `yaml:"learn_from_interactions"`
	SaveConversations     bool   `yaml:"save_conversations"`
	UserPreferencesPath   string `yaml:"user_preferences_path"`
}

type LLMApp struct {
	config     Config
	client     *api.Client
	db         *chromem.DB
	collection *chromem.Collection
}

func loadConfig() (Config, error) {
	var config Config
	data, err := os.ReadFile("config.yaml")
	if err != nil {
		return config, err
	}
	err = yaml.Unmarshal(data, &config)
	return config, err
}

func NewLLMApp() (*LLMApp, error) {
	config, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create Ollama client: %w", err)
	}

	os.MkdirAll(config.Storage.VectorDBPath, 0755)
	os.MkdirAll(config.Storage.ChatHistoryPath, 0755)

	app := &LLMApp{
		config: config,
		client: client,
	}

	app.db = chromem.NewDB()

	embeddingFunc := func(ctx context.Context, text string) ([]float32, error) {
		return app.generateEmbedding(ctx, text)
	}

	collection, err := app.db.GetOrCreateCollection("knowledge", nil, embeddingFunc)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}
	app.collection = collection

	return app, nil
}

func (app *LLMApp) generateEmbedding(ctx context.Context, text string) ([]float32, error) {
	req := &api.EmbedRequest{
		Model: app.config.RAG.EmbeddingModel,
		Input: text,
	}

	resp, err := app.client.Embed(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}

	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return resp.Embeddings[0], nil
}

func (app *LLMApp) retrieveRelevantContext(ctx context.Context, query string) (string, error) {
	if app.collection.Count() == 0 {
		return "", nil
	}

	results, err := app.collection.Query(ctx, query, app.config.RAG.TopK, nil, nil)
	if err != nil {
		return "", fmt.Errorf("retrieval failed: %w", err)
	}

	if len(results) == 0 {
		return "", nil
	}

	var contextBuilder strings.Builder
	contextBuilder.WriteString("\n\nRelevant context from your knowledge base:\n")
	for i, result := range results {
		contextBuilder.WriteString(fmt.Sprintf("\n[%d] %s\n", i+1, result.Content))
	}

	return contextBuilder.String(), nil
}

func (app *LLMApp) Chat(ctx context.Context, userMessage string) (string, error) {
	relevantContext, err := app.retrieveRelevantContext(ctx, userMessage)
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
			Content: app.config.SystemPrompt,
		},
		{
			Role:    "user",
			Content: userMessageWithContext,
		},
	}

	req := &api.ChatRequest{
		Model:    app.config.LLM.Model,
		Messages: messages,
		Options: map[string]any{
			"temperature": app.config.LLM.Temperature,
		},
	}

	var response strings.Builder
	err = app.client.Chat(ctx, req, func(resp api.ChatResponse) error {
		response.WriteString(resp.Message.Content)
		return nil
	})

	if err != nil {
		return "", fmt.Errorf("chat failed: %w", err)
	}

	return response.String(), nil
}

func (app *LLMApp) AddKnowledge(ctx context.Context, content, metadata string) error {
	err := app.collection.AddDocument(ctx, chromem.Document{
		ID:       fmt.Sprintf("doc_%d", app.collection.Count()),
		Content:  content,
		Metadata: map[string]string{"source": metadata},
	})

	return err
}

func (app *LLMApp) IndexDirectory(ctx context.Context, dirPath string) error {
	if _, err := os.Stat(dirPath); os.IsNotExist(err) {
		return fmt.Errorf("directory does not exist: %s", dirPath)
	}

	var indexed int
	err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() {
			return nil
		}

		ext := filepath.Ext(path)
		if ext != ".go" && ext != ".py" && ext != ".js" && ext != ".ts" && ext != ".md" {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			log.Printf("Skipping %s: %v", path, err)
			return nil
		}

		contentStr := string(content)
		chunks := chunkText(contentStr, app.config.RAG.ChunkSize, app.config.RAG.ChunkOverlap)

		for i, chunk := range chunks {
			err := app.collection.AddDocument(ctx, chromem.Document{
				ID:      fmt.Sprintf("%s_chunk_%d", path, i),
				Content: chunk,
				Metadata: map[string]string{
					"source": path,
					"chunk":  fmt.Sprintf("%d", i),
				},
			})
			if err != nil {
				return fmt.Errorf("failed to add chunk from %s: %w", path, err)
			}
		}

		indexed++
		fmt.Printf("✓ Indexed: %s (%d chunks)\n", path, len(chunks))

		return nil
	})

	if err != nil {
		return err
	}

	fmt.Printf("\nIndexed %d files\n", indexed)
	return nil
}

func chunkText(text string, chunkSize, overlap int) []string {
	if len(text) <= chunkSize {
		return []string{text}
	}

	var chunks []string
	start := 0

	for start < len(text) {
		end := start + chunkSize
		if end > len(text) {
			end = len(text)
		}

		chunks = append(chunks, text[start:end])
		start += chunkSize - overlap
	}

	return chunks
}

func main() {
	app, err := NewLLMApp()
	if err != nil {
		log.Fatalf("Failed to initialize app: %v", err)
	}

	fmt.Println("Local LLM with RAG Ready!")
	fmt.Printf("Model: %s\n", app.config.LLM.Model)
	fmt.Printf("Knowledge Base: %d documents\n", app.collection.Count())
	fmt.Println("\nCommands:")
	fmt.Println("  /add <text>       - Add knowledge to vector DB")
	fmt.Println("  /index <path>     - Index a directory (code files)")
	fmt.Println("  /stats            - Show knowledge base stats")
	fmt.Println("  /quit             - Exit")
	fmt.Println("\nType your message:")

	scanner := bufio.NewScanner(os.Stdin)
	ctx := context.Background()

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if input == "/quit" {
			fmt.Println("Goodbye!")
			break
		}

		if input == "/stats" {
			fmt.Printf("Knowledge base contains %d documents\n", app.collection.Count())
			continue
		}

		if strings.HasPrefix(input, "/add ") {
			content := strings.TrimPrefix(input, "/add ")
			err := app.AddKnowledge(ctx, content, "user_input")
			if err != nil {
				fmt.Printf("Error adding knowledge: %v\n", err)
			} else {
				fmt.Println("✅ Added to knowledge base")
			}
			continue
		}

		if strings.HasPrefix(input, "/index ") {
			dirPath := strings.TrimPrefix(input, "/index ")
			fmt.Printf("Indexing directory: %s\n", dirPath)
			err := app.IndexDirectory(ctx, dirPath)
			if err != nil {
				fmt.Printf("Error indexing: %v\n", err)
			}
			continue
		}

		response, err := app.Chat(ctx, input)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("\n%s\n", response)
	}
}
