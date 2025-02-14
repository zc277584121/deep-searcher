from typing import List


def get_vector_db_search_prompt(
    question: str,
    collection_names: List[str],
    collection_descriptions: List[str],
    context: List[str] = None,
):
    sections = []
    # common prompt
    common_prompt = f"""You are an advanced AI problem analyst. Use your reasoning ability and historical conversation information, based on all the existing data sets, to get absolutely accurate answers to the following questions, and generate a suitable question for each data set according to the data set description that may be related to the question.

Question: {question}
"""
    sections.append(common_prompt)
    
    # data set prompt
    data_set = []
    for i, collection_name in enumerate(collection_names):
        data_set.append(f"{collection_name}: {collection_descriptions[i]}")
    data_set_prompt = f"""The following is all the data set information. The format of data set information is data set name: data set description.

Data Sets And Descriptions:
"""
    sections.append(data_set_prompt + "\n".join(data_set))
    
    # context prompt
    if context:
        context_prompt = f"""The following is a condensed version of the historical conversation. This information needs to be combined in this analysis to generate questions that are closer to the answer. You must not generate the same or similar questions for the same data set, nor can you regenerate questions for data sets that have been determined to be unrelated.

Historical Conversation:
"""
        sections.append(context_prompt + "\n".join(context))
    
    # response prompt
    response_prompt = f"""Based on the above, you can only select a few datasets from the following dataset list to generate appropriate related questions for the selected datasets in order to solve the above problems. The output format is json, where the key is the name of the dataset and the value is the corresponding generated question.

Data Sets:
"""
    sections.append(response_prompt + "\n".join(collection_names))
    
    footer = """Respond exclusively in valid JSON format matching exact JSON schema.

Critical Requirements:
- Include ONLY ONE action type
- Never add unsupported keys
- Exclude all non-JSON text, markdown, or explanations
- Maintain strict JSON syntax"""
    sections.append(footer)
    return "\n\n".join(sections)


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