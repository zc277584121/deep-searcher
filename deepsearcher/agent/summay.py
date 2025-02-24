from typing import List, Tuple

from deepsearcher import configuration

# from deepsearcher.configuration import llm
from deepsearcher.agent.prompt import get_final_answer_prompt
from deepsearcher.tools import log
from deepsearcher.vector_db.base import RetrievalResult


def generate_final_answer(
    original_query: str, all_sub_queries: List[str], all_chunks: List[RetrievalResult]
) -> Tuple[str, int]:
    llm = configuration.llm
    chunk_texts = []
    for chunk in all_chunks:
        if "wider_text" in chunk.metadata:
            chunk_texts.append(chunk.metadata["wider_text"])
        else:
            chunk_texts.append(chunk.text)
    log.color_print(
        f"<think> Summarize answer from all {len(all_chunks)} retrieved chunks... </think>\n"
    )
    summary_prompt = get_final_answer_prompt(
        question=original_query,
        mini_questions=all_sub_queries,
        mini_chuncks=chunk_texts,
    )
    chat_response = llm.chat([{"role": "user", "content": summary_prompt}])
    return chat_response.content, chat_response.total_tokens
