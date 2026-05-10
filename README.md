# SmartSort 🧠📁
Intelligent, Local-First, Privacy-Preserving File Organization for macOS.

SmartSort scans messy directories and organizes them based on *content and context*, not just file extensions. It is specifically optimized for corporate Apple Silicon Macs, running a fast rules-engine and falling back to local LLMs via Ollama to ensure sensitive data never leaves your machine.

## Prerequisites (macOS)
1. Install [Ollama](https://ollama.com/download)
2. Pull the lightweight model:
   ```bash
   ollama pull qwen2.5:3b