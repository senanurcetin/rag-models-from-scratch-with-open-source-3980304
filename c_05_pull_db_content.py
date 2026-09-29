import sys
from c_03_database_connect_embeddings import get_psql_session, TextEmbedding
from rag_config import load_embedding_model, encode_query


# Finding the content from our database which is most similar to the query
def search_embeddings(query_embedding, session, limit=5):
    return session.query(TextEmbedding.id, TextEmbedding.sentence_number, TextEmbedding.content, TextEmbedding.file_name,
        TextEmbedding.embedding.cosine_distance(query_embedding).label("distance") )\
        .order_by("distance").limit(limit).all()


def get_surrounding_sentences(entry_ids, file_names, sentence_numbers, group_window_size, session):
    # Neighbours are looked up by (file_name, sentence_number) so a window never crosses into another article.
    surrounding_sentences = []
    for file_name, sentence_number in zip(file_names, sentence_numbers):
        surrounding_sentences.append(
            session.query(TextEmbedding.id, TextEmbedding.sentence_number, TextEmbedding.content, TextEmbedding.file_name)\
            .filter(TextEmbedding.file_name == file_name)\
            .filter(TextEmbedding.sentence_number >= sentence_number - group_window_size)\
            .filter(TextEmbedding.sentence_number <= sentence_number + group_window_size)\
            .order_by(TextEmbedding.sentence_number).all()
        )

    return surrounding_sentences


if __name__ == '__main__':
    query = sys.argv[1] if len(sys.argv) > 1 else "Tell me about human rights in Germany."
    model = load_embedding_model()
    query_embedding = encode_query(model, query)
    session = get_psql_session()
    for row in search_embeddings(query_embedding=query_embedding, session=session, limit=5):
        print(row)
