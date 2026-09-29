# Base image
FROM ubuntu:22.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Set locale to avoid warnings
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install tools and dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    ca-certificates \
    lsb-release \
    software-properties-common \
    wget \
    git \
    sudo \
    python3 \
    python3-pip \
    python3-venv

# Add PostgreSQL APT repository
RUN echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    apt-get update && apt-get install -y \
    postgresql-14 \
    postgresql-contrib \
    postgresql-14-pgvector

# Disable postgres password
RUN sed -i -E 's/\b(peer|md5|scram-sha-256)\b/trust/g' /etc/postgresql/14/main/pg_hba.conf

# Set environment variables for PostgreSQL
ENV PGDATA=/var/lib/postgresql/14/main

# Clean up
RUN rm -rf /var/lib/apt/lists/*

# Install Ollama (system-level daemon)
RUN curl -fsSL https://ollama.com/install.sh -o install-ollama.sh && \
    bash install-ollama.sh && \
    rm install-ollama.sh

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Hugging Face cache (mounted as a volume, so the embedding model is downloaded only once)
RUN mkdir -p /home/$USERNAME/.cache/huggingface && chown -R $USERNAME:$USERNAME /home/$USERNAME/.cache

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install Python packages
COPY --chown=vscode:vscode requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# Expose PostgreSQL default port
EXPOSE 5432