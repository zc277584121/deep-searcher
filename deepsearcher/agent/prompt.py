from typing import List


def get_vector_db_search_prompt(
    question: str,
    collection_info: List,
    context: List[str] = None,
):
    return f"""
I provide you with collection_name(s) and corresponding collection_description(s). Please select the collection names that may be related to the question and return a python list of str. If there is no collection related to the question, you can return an empty list.

"QUESTION": {question}
"COLLECTION_INFO": {collection_info}

When you return, you can ONLY return a python list of str, WITHOUT any other additional content. Your selected collection name list is:
"""


def get_reflect_prompt(
    question: str,
    mini_questions: List[str],
    mini_chuncks: List[str],
):
    mini_chunk_str = ""
    for i, chunk in enumerate(mini_chuncks):
        mini_chunk_str += f"""<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"""
    reflect_prompt = f"""Determine whether additional search queries are needed based on the original query, previous sub queries, and all retrieved document chunks. If further research is required, provide a Python list of up to 3 search queries. If no further research is required, return an empty list.

If the original query is to write a report, then you prefer to generate some further queries, instead return an empty list.

    Original Query: {question}
    Previous Sub Queries: {mini_questions}
    Related Chunks: 
    {mini_chunk_str}
    """

    footer = """Respond exclusively in valid List of str format without any other text."""
    return reflect_prompt + footer


def get_final_answer_prompt(
    question: str,
    mini_questions: List[str],
    mini_chuncks: List[str],
):
    mini_chunk_str = ""
    for i, chunk in enumerate(mini_chuncks):
        mini_chunk_str += f"""<chunk_{i}>\n{chunk}\n</chunk_{i}>\n"""
    summary_prompt = f"""You are a AI content analysis expert, good at summarizing content. Please summarize a specific and detailed answer or report based on the previous queries and the retrieved document chunks.

    Original Query: {question}
    Previous Sub Queries: {mini_questions}
    Related Chunks: 
    {mini_chunk_str}
    """
    return summary_prompt
