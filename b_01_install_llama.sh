#!/bin/bash
set -e

# https://ollama.com/download/linux

curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama server in the background (it would block the script if run in the foreground)
if ! curl -s http://localhost:11434 > /dev/null; then
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    until curl -s http://localhost:11434 > /dev/null; do sleep 1; done
fi

ollama pull qwen3:0.6b
# ollama pull deepseek-r1:8b

# Optional: build the custom model defined in b_02_ModelFile, then chat with it
ollama create rag-qwen3 -f "$(dirname "$0")/b_02_ModelFile"
# ollama run rag-qwen3
