package main

import (
	"fmt"
	"log"
	"os"

	"llm-workspace/interactive"
	"llm-workspace/server"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "interactive" {
		if err := interactive.Run(); err != nil {
			log.Fatal(err)
		}
		return
	}

	pidFile := "/tmp/llm-workspace.pid"
	pid := fmt.Sprintf("%d", os.Getpid())
	if err := os.WriteFile(pidFile, []byte(pid), 0644); err != nil {
		log.Printf("Warning: could not write PID file: %v", err)
	}
	defer os.Remove(pidFile)

	srv, err := server.New()
	if err != nil {
		os.Remove(pidFile)
		log.Fatal(err)
	}

	if err := srv.Start(); err != nil {
		os.Remove(pidFile)
		log.Fatal(err)
	}
}
