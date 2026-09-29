import os
import nltk
from nltk.tokenize import sent_tokenize
from c_03_database_connect_embeddings import get_psql_session, TextEmbedding
from rag_config import load_embedding_model


def ensure_nltk_data():
    for resource in ("punkt", "punkt_tab"):
        nltk.download(resource, quiet=True)


def populate_vector_database(folder_path='all_articles', batch_size=32):

    # Check if the directory exists, if not create it
    os.makedirs(folder_path, exist_ok=True)

    ensure_nltk_data()
    session = get_psql_session()
    model = load_embedding_model()

    # Files that are already embedded are skipped, so the script can be re-run safely
    already_done = {row[0] for row in session.query(TextEmbedding.file_name).distinct()}

    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        if filename in already_done:
            print("Skipping (already in database): {}".format(file_path))
            continue

        print("Trying: {}".format(file_path))

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            sentences = [s for s in sent_tokenize(content) if s.strip()]
            if not sentences:
                continue
            embeddings = model.encode(sentences, batch_size=batch_size)

            for i, (embedding, sentence) in enumerate(zip(embeddings, sentences)):
                session.add(TextEmbedding(embedding=embedding, content=sentence,
                                          file_name=filename, sentence_number=i + 1))
            session.commit()

            print("Successfully generated embeddings for: {}".format(file_path))

        except Exception as e:
            session.rollback()
            print(f"Error processing {filename}: {str(e)}")
            continue


if __name__ == '__main__':
    populate_vector_database()
