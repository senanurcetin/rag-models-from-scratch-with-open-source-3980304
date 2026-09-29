#!/bin/bash
# Starts PostgreSQL and Ollama and prepares the database and chat model.
#   post-create.sh            -> start services + one-time setup (database, model pull)
#   post-create.sh services   -> only (re)start the services
set -e

sudo service postgresql start
until pg_isready -q -h localhost; do sleep 1; done

if ! curl -s http://localhost:11434 > /dev/null; then
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    until curl -s http://localhost:11434 > /dev/null; do sleep 1; done
fi

[ "$1" = "services" ] && exit 0

# pg_hba.conf uses "trust", so no password is needed
psql -U postgres -h localhost -tc "SELECT 1 FROM pg_database WHERE datname='text_embeddings'" | grep -q 1 \
    || psql -U postgres -h localhost -c "CREATE DATABASE text_embeddings"
psql -U postgres -h localhost -d text_embeddings -f init-db/init.sql

ollama pull qwen3:0.6b
ollama create rag-qwen3 -f b_02_ModelFile
