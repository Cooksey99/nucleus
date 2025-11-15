package interactive

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"llm-workspace/chat"
	"llm-workspace/config"
	"llm-workspace/rag"
	"llm-workspace/tools"

	"github.com/ollama/ollama/api"
)

func Run() error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("failed to create Ollama client: %w", err)
	}

	ragManager, err := rag.NewManager(cfg, client)
	if err != nil {
		return fmt.Errorf("failed to initialize RAG: %w", err)
	}

	toolRegistry := tools.NewRegistry(cfg)
	chatManager := chat.NewManager(cfg, client, ragManager, toolRegistry)

	fmt.Println("Local LLM with RAG Ready!")
	fmt.Printf("Model: %s\n", cfg.LLM.Model)
	fmt.Printf("Knowledge Base: %d documents\n", ragManager.Count())
	fmt.Println("\nCommands:")
	fmt.Println("  /add <text>       - Add knowledge to vector DB")
	fmt.Println("  /index <path>     - Index a directory (code files)")
	fmt.Println("  /edit <request>   - Enable file editing mode")
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
			fmt.Printf("Knowledge base contains %d documents\n", ragManager.Count())
			continue
		}

		if strings.HasPrefix(input, "/add ") {
			content := strings.TrimPrefix(input, "/add ")
			err := ragManager.AddKnowledge(ctx, content, "user_input")
			if err != nil {
				fmt.Printf("Error adding knowledge: %v\n", err)
			} else {
				fmt.Println("Added to knowledge base")
			}
			continue
		}

		if strings.HasPrefix(input, "/index ") {
			dirPath := strings.TrimPrefix(input, "/index ")
			fmt.Printf("Indexing directory: %s\n", dirPath)
			err := ragManager.IndexDirectory(ctx, dirPath)
			if err != nil {
				fmt.Printf("Error indexing: %v\n", err)
			}
			continue
		}

		if strings.HasPrefix(input, "/edit ") {
			request := strings.TrimPrefix(input, "/edit ")
			response, err := chatManager.ChatWithTools(ctx, request)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else if response != "" {
				fmt.Printf("\n%s\n", response)
			}
			continue
		}

		response, err := chatManager.Chat(ctx, input)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("\n%s\n", response)
	}

	return nil
}
