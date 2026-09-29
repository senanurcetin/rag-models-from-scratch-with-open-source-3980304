from sqlalchemy import create_engine, text, Column, Integer, String, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector

import rag_config
from rag_config import EMBEDDING_DIM

Base = declarative_base()


class TextEmbedding(Base):
    __tablename__ = 'text_embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding = Column(Vector(EMBEDDING_DIM))
    content = Column(String)
    file_name = Column(String)
    sentence_number = Column(Integer)

    __table_args__ = (
        # Approximate nearest-neighbour index for cosine distance
        Index(
            'ix_text_embeddings_embedding_hnsw', 'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'},
        ),
        # Fast neighbouring-sentence lookups
        Index('ix_text_embeddings_file_sentence', 'file_name', 'sentence_number'),
    )

    def __str__(self):
        return f"{self.content} {self.id}"


# Connect to PostgreSQL
def get_psql_session():
    engine = create_engine(rag_config.DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    return Session()


def insert_embeddings(embeddings, contents, file_names, session):
    counters = {}
    for embedding, content, file_name in zip(embeddings, contents, file_names):
        counters[file_name] = counters.get(file_name, 0) + 1
        session.add(TextEmbedding(embedding=embedding, content=content, file_name=file_name,
                                  sentence_number=counters[file_name]))
    session.commit()
