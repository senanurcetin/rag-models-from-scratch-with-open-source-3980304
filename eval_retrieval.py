"""Measure retrieval quality: for each question, is the expected article among the top-k results?

    python eval_retrieval.py [eval_set.json] [k]

eval_set.json is a list of {"question": "...", "expected_files": ["Some_article.txt", ...]}.
File names are the sanitized Wikipedia titles written by b_04_generate_corpus.py (non-alphanumerics -> "_").
"""
import json
import sys


def rank_of_first_hit(ranked_files, expected_files):
    """1-based rank of the first expected file in the ranked list, or None."""
    for rank, file_name in enumerate(ranked_files, 1):
        if file_name in expected_files:
            return rank
    return None


def summarize(ranks, k):
    """hit@k and mean reciprocal rank over a list of ranks (None = miss)."""
    n = len(ranks)
    if n == 0:
        return {"questions": 0, "hit_at_k": 0.0, "mrr": 0.0}
    hits = sum(1 for r in ranks if r is not None and r <= k)
    mrr = sum(1.0 / r for r in ranks if r is not None and r <= k) / n
    return {"questions": n, "hit_at_k": hits / n, "mrr": mrr}


def unique_files(rows):
    seen, files = set(), []
    for row in rows:
        if row.file_name not in seen:
            seen.add(row.file_name)
            files.append(row.file_name)
    return files


def main(path="eval_set.json", k=5):
    from c_03_database_connect_embeddings import get_psql_session, TextEmbedding
    from c_05_pull_db_content import search_embeddings
    from rag_config import load_embedding_model, encode_query

    with open(path, encoding="utf-8") as f:
        eval_set = json.load(f)

    session = get_psql_session()
    indexed = {r[0] for r in session.query(TextEmbedding.file_name).distinct()}
    model = load_embedding_model()

    ranks = []
    for item in eval_set:
        missing = [name for name in item["expected_files"] if name not in indexed]
        if len(missing) == len(item["expected_files"]):
            print(f"SKIP (expected article not in database): {item['question']}")
            continue
        # over-fetch sentences, then collapse to distinct articles
        rows = search_embeddings(encode_query(model, item["question"]), session, limit=k * 20)
        rank = rank_of_first_hit(unique_files(rows), set(item["expected_files"]))
        ranks.append(rank)
        print(f"{'HIT ' + str(rank) if rank else 'MISS'}: {item['question']}")

    print(json.dumps(summarize(ranks, k), indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_set.json",
         int(sys.argv[2]) if len(sys.argv) > 2 else 5)
