"""Shared settings for the RAG pipeline. Everything can be overridden with environment variables."""
import os

# The driver is named explicitly: SQLAlchemy 2.1+ maps plain "postgresql://" to psycopg (v3), not psycopg2.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost/text_embeddings"
)

# Qwen3-Embedding-0.6B produces 1024-dimensional vectors.
EMBEDDING_DIM = 1024

_LOCAL_EMBEDDING_MODEL = "/models/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    _LOCAL_EMBEDDING_MODEL
    if os.path.isdir(_LOCAL_EMBEDDING_MODEL)
    else "Qwen/Qwen3-Embedding-0.6B",  # downloaded from the Hugging Face Hub on first use
)

CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3:0.6b")

NO_ANSWER = "Given the information provided, I am unable to answer your question."

SYSTEM_PROMPT = (
    "You are a bot which only responds to questions based on the content that you are provided with. "
    f"If the information that the user requests is not found within the provided content simply respond by saying '{NO_ANSWER}' "
    "The content will be provided at the beginning of the prompt between a <|content_start> and <|content_end> tag."
)


def load_embedding_model():
    """Load the sentence-transformers embedding model (imported lazily because it is heavy)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def encode_query(model, query):
    # Qwen3-Embedding is trained with an instruction prefix for queries; documents get none.
    return model.encode(query, prompt_name="query")
