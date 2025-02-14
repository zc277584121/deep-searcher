# Some test dataset and evaluation method are ref from https://github.com/OSU-NLP-Group/HippoRAG/tree/main/data , many thanks

################################################################################
# Note: This evaluation script will cost a lot of LLM token usage, please make sure you have enough token budget.
################################################################################


import json
import os

from tqdm import tqdm

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query, naive_rag_query, naive_retrieve, retrieve

dataset_name = "2wikimultihopqa"

current_dir = os.path.dirname(os.path.abspath(__file__))

corpus_file = os.path.join(current_dir, f'data/{dataset_name}_corpus.json')

config = Configuration()

config.set_provider_config("file_loader", "JsonFileLoader", {"text_key": "text"})
## Replace with your provider settings
config.set_provider_config("vector_db", "Milvus", {"uri": ...})
config.set_provider_config("llm", "OpenAI", {"model": "gpt-4o-mini"})
# config.set_provider_config("llm", "AzureOpenAI", {
#     "model": "zilliz-gpt-4o-mini",
#     "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT_BAK"),
#     "api_key": os.getenv("AZURE_OPENAI_API_KEY_BAK"),
#     "api_version": "2023-05-15"
# })
config.set_provider_config("embedding", "OpenAIEmbedding", {"model_name": "text-embedding-ada-002"})
init_config(config = config)

# set chunk size to a large number to avoid chunking, because the dataset was chunked already.
load_from_local_files(corpus_file, force_new_collection=True, chunk_size=999999, chunk_overlap=0)


data_with_gt_file_path = os.path.join(current_dir, f'data/{dataset_name}.json')
data_with_gt = json.load(open(data_with_gt_file_path, 'r'))

k_list = [1, 2, 5, 10, 15, 20, 30, 40, 50, 80, 100]
total_recall = {k: 0 for k in k_list}


# There are 1000 samples in total,
# for cost efficiency, we only evaluate the first 300 samples by default.
# You can change the value of pre_num to evaluate more samples.
PRE_NUM = 300

if not PRE_NUM:
    PRE_NUM = len(data_with_gt)
    
for sample_idx, sample in tqdm(enumerate(data_with_gt), total=min(PRE_NUM, len(data_with_gt)), desc='Evaluation'):  # for each sample
    question = sample['question']

    retry_num = 3
    for i in range(retry_num):
        try:
            retrieved_results, _ = retrieve(question)
            break
        except SyntaxError as e:
            print("Parse LLM's output failed, retry again...")

    # naive_retrieved_results = naive_retrieve(question)

    retrieved_titles = [retrieved_result.metadata["title"] for retrieved_result in retrieved_results]
    # naive_retrieved_titles = [retrieved_result.metadata["title"] for retrieved_result in naive_retrieved_results]

    # retrieved_titles = naive_retrieved_titles#todo

    if dataset_name in ['hotpotqa', 'hotpotqa_train']:
        gold_passages = [item for item in sample['supporting_facts']]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = retrieved_titles
    elif dataset_name in ['2wikimultihopqa']:
        gold_passages = [item for item in sample['supporting_facts']]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = retrieved_titles
    # elif dataset_name in ['musique']:
    #     gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
    #     gold_items = set(
    #         [item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
    #     retrieved_items = retrieved_passages
    # else:
    #     gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
    #     gold_items = set(
    #         [item['title'] + '\n' + item['text'] for item in gold_passages])
    #     retrieved_items = retrieved_passages




    # calculate metrics
    recall = dict()
    print(f'idx: {sample_idx + 1} ', end='')
    for k in k_list:
        recall[k] = round(
            sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items), 4)
        total_recall[k] += recall[k]
        print(f'R@{k}: {total_recall[k] / (sample_idx + 1):.4f} ', end='')
    print()