import ast
from typing import List

from deepsearcher.tools import log
# from deepsearcher.configuration import llm
from deepsearcher import configuration

PROMPT = """To answer this question more comprehensively, please break down the original question into up to four sub-questions. Return as list of str.
If this is a very simple question and no decomposition is necessary, then keep the only one original question in the list.

Original Question: {original_query}


<EXAMPLE>
Example input:
"Explain deep learning"

Example output:
[
    "What is deep learning?",
    "What is the difference between deep learning and machine learning?",
    "What is the history of deep learning?"
]
</EXAMPLE>

Provide your response in list of str format:
"""
def generate_sub_queries(original_query: str) -> List[str]:
    llm = configuration.llm
    chat_response = llm.chat(messages=[{"role": "user", "content": PROMPT.format(original_query=original_query)}])
    response_content = chat_response.content
    try:
        sub_queries = ast.literal_eval(response_content)
    except:
        print(f"Failed to parse response: {response_content}\nReturning original query as sub-query.")
        sub_queries = [original_query]
    return sub_queries