#!/bin/bash
# Native PostgreSQL 16 + pgvector install for Ubuntu/Debian.
# Easier alternatives: `docker compose up -d` (see docker-compose.yml) or the devcontainer.
set -e

sudo apt install -y wget ca-certificates

sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/postgresql.gpg

sudo apt update
sudo apt install -y postgresql-16 postgresql-contrib postgresql-16-pgvector
sudo service postgresql start

# The scripts connect as postgres/postgres (see DATABASE_URL in rag_config.py)
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres'"
sudo -u postgres psql -c "CREATE DATABASE text_embeddings OWNER postgres"
sudo -u postgres psql -d text_embeddings -c "CREATE EXTENSION IF NOT EXISTS vector"
