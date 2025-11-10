// Package rag provides retrieval augmented generation functionality.
package rag

import (
	"context"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"

	"llm-workspace/config"

	"github.com/ollama/ollama/api"
	chromem "github.com/philippgille/chromem-go"
)

// Manager handles vector database operations and RAG workflows.
type Manager struct {
	config     *config.Config
	client     *api.Client
	db         *chromem.DB
	collection *chromem.Collection
}

// Creates a new instance with the given configuration and Ollama client.
func NewManager(cfg *config.Config, client *api.Client) (*Manager, error) {
	os.MkdirAll(cfg.Storage.VectorDBPath, 0755)
	os.MkdirAll(cfg.Storage.ChatHistoryPath, 0755)

	manager := &Manager{
		config: cfg,
		client: client,
		db:     chromem.NewDB(),
	}

	embeddingFunc := func(ctx context.Context, text string) ([]float32, error) {
		return manager.generateEmbedding(ctx, text)
	}

	collection, err := manager.db.GetOrCreateCollection("knowledge", nil, embeddingFunc)
	if err != nil {
		return nil, fmt.Errorf("failed to create collection: %w", err)
	}
	manager.collection = collection

	return manager, nil
}

// Converts text into a vector embedding using the configured model.
func (m *Manager) generateEmbedding(ctx context.Context, text string) ([]float32, error) {
	req := &api.EmbedRequest{
		Model: m.config.RAG.EmbeddingModel,
		Input: text,
	}

	resp, err := m.client.Embed(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}

	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return resp.Embeddings[0], nil
}

// Searches for relevant chunks matching the query.
func (m *Manager) RetrieveContext(ctx context.Context, query string) (string, error) {
	if m.collection.Count() == 0 {
		return "", nil
	}

	results, err := m.collection.Query(ctx, query, m.config.RAG.TopK, nil, nil)
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

// Adds a single piece of text to the knowledge base.
func (m *Manager) AddKnowledge(ctx context.Context, content, metadata string) error {
	err := m.collection.AddDocument(ctx, chromem.Document{
		ID:       fmt.Sprintf("doc_%d", m.collection.Count()),
		Content:  content,
		Metadata: map[string]string{"source": metadata},
	})

	return err
}

// Recursively indexes all code files in a directory.
func (m *Manager) IndexDirectory(ctx context.Context, dirPath string) error {
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
		chunks := chunkText(contentStr, m.config.RAG.ChunkSize, m.config.RAG.ChunkOverlap)

		for i, chunk := range chunks {
			err := m.collection.AddDocument(ctx, chromem.Document{
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

// Returns the total number of documents in the knowledge base.
func (m *Manager) Count() int {
	return m.collection.Count()
}

// Splits text into overlapping chunks of the specified size.
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
