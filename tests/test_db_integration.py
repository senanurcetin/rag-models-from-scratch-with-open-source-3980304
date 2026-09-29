"""SQL-level checks against a real PostgreSQL + pgvector. Skipped unless TEST_DATABASE_URL is set, e.g.

    TEST_DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost/text_embeddings_test pytest tests
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_DB = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")

import rag_config
from sqlalchemy import text
from c_03_database_connect_embeddings import get_psql_session, TextEmbedding, Base, insert_embeddings
from c_05_pull_db_content import search_embeddings
from d_02_prepare_content import get_filtered_matches, get_surrounding_sentences


@pytest.fixture()
def session(monkeypatch):
    monkeypatch.setattr(rag_config, "DATABASE_URL", TEST_DB)
    s = get_psql_session()
    Base.metadata.drop_all(s.get_bind())
    s.close()
    s = get_psql_session()
    yield s
    s.close()


def _vectors(n, seed):
    return np.random.default_rng(seed).normal(size=(n, rag_config.EMBEDDING_DIM)).astype(np.float32)


def test_search_and_windows(session):
    a, b = _vectors(30, 1), _vectors(30, 2)
    insert_embeddings(a, [f"a{i+1}" for i in range(30)], ["a.txt"] * 30, session)
    insert_embeddings(b, [f"b{i+1}" for i in range(30)], ["b.txt"] * 30, session)

    # sentence_number is assigned per file
    assert session.query(TextEmbedding.sentence_number).filter_by(file_name="b.txt").order_by(TextEmbedding.sentence_number).first()[0] == 1

    # the query equals sentence 15 of a.txt and (slightly perturbed) sentence 2 of b.txt
    query = a[14]
    results = search_embeddings(query, session, limit=55)
    assert (results[0].file_name, results[0].sentence_number) == ("a.txt", 15)

    matches = get_filtered_matches(results, num_matches=2, group_window_size=5)
    windows = get_surrounding_sentences([m[3] for m in matches], [m[1] for m in matches], 5, session)
    first = windows[0] if windows[0][0].file_name == "a.txt" else windows[1]
    numbers = [r.sentence_number for r in first]
    assert numbers == list(range(10, 21))          # +-5 around sentence 15, ordered
    assert {r.file_name for r in first} == {"a.txt"}  # never crosses into another article


def test_window_at_file_start_does_not_leak_into_previous_file(session):
    a, b = _vectors(12, 3), _vectors(12, 4)
    insert_embeddings(a, [f"a{i+1}" for i in range(12)], ["a.txt"] * 12, session)
    insert_embeddings(b, [f"b{i+1}" for i in range(12)], ["b.txt"] * 12, session)
    windows = get_surrounding_sentences(["b.txt"], [2], 5, session)
    assert [r.content for r in windows[0]] == [f"b{i}" for i in range(1, 8)]


def test_hnsw_index_exists(session):
    idx = {r[0] for r in session.execute(text("SELECT indexname FROM pg_indexes WHERE tablename='text_embeddings'"))}
    assert "ix_text_embeddings_embedding_hnsw" in idx and "ix_text_embeddings_file_sentence" in idx

