import re
import sys
from rag_config import CHAT_MODEL, SYSTEM_PROMPT


def format_context(context):
    """Turn the retrieved windows (lists of DB rows) into plain text, one passage per window."""
    passages = []
    for window in context:
        if not window:
            continue
        text = " ".join(row.content.strip() for row in window)
        passages.append(f"[Source: {window[0].file_name}]\n{text}")
    return "\n\n".join(passages)


def clean_answer(text):
    """Remove the <think>...</think> reasoning block that Qwen3 models may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_prompt(context, query):
    return f"<|content_start>\n{format_context(context)}\n<|content_end>\n\n{query}"


def answer(query):
    from ollama import chat
    from d_02_prepare_content import search_by_query

    context = search_by_query(query)
    prompt = build_prompt(context, query)

    response = chat(
        model=CHAT_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        options={'temperature': 0, 'num_ctx': 4096, 'seed': 42},
    )
    return clean_answer(response.message.content)


if __name__ == '__main__':
    query = "Is the Dominican Republic a member of the United Nations?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    print(answer(query))
