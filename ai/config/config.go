package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Holds all application configuration settings.
type Config struct {
	LLM             LLMConfig             `yaml:"llm"`
	SystemPrompt    string                `yaml:"system_prompt"`
	RAG             RAGConfig             `yaml:"rag"`
	Storage         StorageConfig         `yaml:"storage"`
	Personalization PersonalizationConfig `yaml:"personalization"`
}

// Contains settings for the language model.
type LLMConfig struct {
	Model         string  `yaml:"model"`
	BaseURL       string  `yaml:"base_url"`
	Temperature   float64 `yaml:"temperature"`
	ContextLength int     `yaml:"context_length"`
}

// Contains settings for retrieval augmented generation.
type RAGConfig struct {
	EmbeddingModel string `yaml:"embedding_model"`
	ChunkSize      int    `yaml:"chunk_size"`
	ChunkOverlap   int    `yaml:"chunk_overlap"`
	TopK           int    `yaml:"top_k"`
}

// Defines paths for persistent data storage.
type StorageConfig struct {
	VectorDBPath    string `yaml:"vector_db_path"`
	ChatHistoryPath string `yaml:"chat_history_path"`
}

// Controls learning and memory features.
type PersonalizationConfig struct {
	LearnFromInteractions bool   `yaml:"learn_from_interactions"`
	SaveConversations     bool   `yaml:"save_conversations"`
	UserPreferencesPath   string `yaml:"user_preferences_path"`
}

// Reads and parses config.yaml from the current directory.
func Load() (*Config, error) {
	data, err := os.ReadFile("config.yaml")
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return &cfg, nil
}
