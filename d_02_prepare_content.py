import sys
import gc
from c_03_database_connect_embeddings import get_psql_session, TextEmbedding
from c_05_pull_db_content import search_embeddings
from rag_config import load_embedding_model, encode_query

# Search results are (id, sentence_number, content, file_name, distance) rows.
SENTENCE_NUMBER = 1
FILE_NAME = 3


# Check if a match's context window overlaps with the context window of an already accepted match.
# Two windows of +-group_window_size sentences overlap when the matches are at most 2 * group_window_size apart.
def is_unique_to_window(existing_matches, current_match, group_window_size=5):

    for match in existing_matches:
        if match[FILE_NAME] != current_match[FILE_NAME]:
            continue
        if abs(match[SENTENCE_NUMBER] - current_match[SENTENCE_NUMBER]) <= 2 * group_window_size:
            return False

    return True


# Walk the ranked search results and keep the best matches whose context windows do not overlap.
def get_filtered_matches(search_results, num_matches=5, group_window_size=5):
    matches = []
    for result in search_results:
        if len(matches) >= num_matches:
            break
        if is_unique_to_window(matches, result, group_window_size):
            matches.append(result)

    return matches


# Turn matches into (file_name, first_sentence, last_sentence) windows and merge the ones that touch.
def merge_windows(file_names, sentence_numbers, group_window_size):
    windows = sorted(
        (file_name, max(1, number - group_window_size), number + group_window_size)
        for file_name, number in zip(file_names, sentence_numbers)
    )

    merged = []
    for file_name, start, end in windows:
        if merged and merged[-1][0] == file_name and start <= merged[-1][2] + 1:
            merged[-1] = (file_name, merged[-1][1], max(merged[-1][2], end))
        else:
            merged.append((file_name, start, end))

    return merged


def get_surrounding_sentences(file_names, sentence_numbers, group_window_size, session):

    surrounding_sentences = []
    for file_name, start, end in merge_windows(file_names, sentence_numbers, group_window_size):
        surrounding_sentences.append(
            session.query(TextEmbedding.id, TextEmbedding.sentence_number, TextEmbedding.content, TextEmbedding.file_name)\
            .filter(TextEmbedding.file_name == file_name)\
            .filter(TextEmbedding.sentence_number >= start)\
            .filter(TextEmbedding.sentence_number <= end)\
            .order_by(TextEmbedding.sentence_number)\
            .all()
        )

    return surrounding_sentences


def search_by_query(query, num_matches=5, group_window_size=5):

    session = get_psql_session()
    model = load_embedding_model()
    query_embedding = encode_query(model, query)
    del model
    gc.collect()

    # Fetch extra candidates so that we can still find num_matches non-overlapping windows
    search_results = search_embeddings(query_embedding, session=session, limit=num_matches * (2*group_window_size + 1) )
    filtered_matches = get_filtered_matches(search_results, num_matches, group_window_size)

    file_names = [m[FILE_NAME] for m in filtered_matches]
    sentence_numbers = [m[SENTENCE_NUMBER] for m in filtered_matches]

    return get_surrounding_sentences(file_names=file_names, sentence_numbers=sentence_numbers,
                                     group_window_size=group_window_size, session=session)


if __name__=="__main__":

    query = "Tell me about children's rights in Germany."

    if len(sys.argv) > 1:
        query = sys.argv[1]

    context = search_by_query(query)

    for window in context:
        print(window, "\n")
