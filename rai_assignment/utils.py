from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.evaluation import generate_question_context_pairs, EmbeddingQAFinetuneDataset
from llama_index.core.llama_dataset import BaseLlamaPredictionDataset

import nest_asyncio

import pandas as pd
import os

def load_documents(documents_path):
    documents = SimpleDirectoryReader(documents_path).load_data()
    return documents

def get_nodes(documents, chunk_size=512, chunk_overlap=10):
    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = text_splitter.get_nodes_from_documents(documents)
    return nodes

def create_index(nodes, vector_store):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index

def create_retriever(index, similarity_top_k):
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    return retriever

def configure_query_engine(index, llm, retriever, similarity_cutoff=0.2, required_keywords=[], exclude_keywords=[]):
    similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    keyword_postprocessor = KeywordNodePostprocessor(required_keywords=required_keywords, exclude_keywords=exclude_keywords)
    post_processors = [similarity_postprocessor, keyword_postprocessor]
    
    response_synthesizer = get_response_synthesizer(llm=llm,response_mode=ResponseMode.COMPACT)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=post_processors,
        response_synthesizer=response_synthesizer,
    )
    
    return query_engine

def create_retrieval_qa_dataset(nodes, llm, path, num_questions_per_chunk=2):
    nest_asyncio.apply()

    if not os.path.exists(path):
        qa_dataset = generate_question_context_pairs(
            nodes, llm=llm, num_questions_per_chunk=num_questions_per_chunk
        )
        qa_dataset.save_json(path)

    return EmbeddingQAFinetuneDataset.from_json(path)

def display_results(name, eval_results):
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    columns = {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}

    metric_df = pd.DataFrame(columns)

    return metric_df

# def create_predictions_dataset(rag_dataset, path, sleep_time_in_seconds=2, batch_size=10):    
#     if not os.path.exists(path):
#         prediction_data = rag_dataset.amake_predictions_with(
#             predictor=query_engine, 
#             batch_size=batch_size, show_progress=True, 
#             sleep_time_in_seconds=sleep_time_in_seconds 
#         )
#         prediction_data.save_json(path)

#     prediction_data = BaseLlamaPredictionDataset.from_json("predictions.json")

#     return rag_dataset