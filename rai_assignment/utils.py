import os
import json
import pandas as pd
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio
from more_itertools import chunked
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset,AnswerRelevancyEvaluator, ContextRelevancyEvaluator, FaithfulnessEvaluator, SemanticSimilarityEvaluator, generate_question_context_pairs
from llama_index.core.evaluation.notebook_utils import get_eval_results_df

def load_documents(documents_path):
    documents = SimpleDirectoryReader(documents_path).load_data()
    return documents


def get_nodes(documents, chunk_size=512, chunk_overlap=10):
    text_splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = text_splitter.get_nodes_from_documents(documents)
    return nodes


def create_index(nodes, vector_store):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index


def create_retrieval_qa_dataset(nodes, llm, path, num_questions_per_chunk=2):
    if os.path.exists(path):
        os.remove(path)
        print("Previous retrieval evaluation dataset deleted successfully.")
    else:
        print("Retrieval evaluation dataset does not exist. Creating one now...")

    qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=num_questions_per_chunk
    )
    qa_dataset.save_json(path)

    return EmbeddingQAFinetuneDataset.from_json(path)


def display_retrieval_evaluation_results(name, eval_results):
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

def create_question_dataset(nodes, llm):

    NUM_QUESTIONS_GENERATED_PER_NODE = 2

    dataset_generator = RagDatasetGenerator(
        nodes,
        llm=llm,
        num_questions_per_chunk=NUM_QUESTIONS_GENERATED_PER_NODE,
    )
    rag_dataset = dataset_generator.generate_questions_from_nodes()
    return rag_dataset

async def create_prediction_dataset(rag_dataset, query_engine):
    
    PREDICTION_REQUEST_BATCH_SIZE = 10
    REQUEST_WAIT_TIME = 2

    prediction_data = await rag_dataset.amake_predictions_with(
        predictor=query_engine, 
        batch_size=PREDICTION_REQUEST_BATCH_SIZE, 
        sleep_time_in_seconds=REQUEST_WAIT_TIME,
        show_progress=True, 
    )
    return prediction_data

def create_judges(evaluation_llm):
    judges = {}

    judges["answer_relevancy"] = AnswerRelevancyEvaluator(
        llm=evaluation_llm,
    )

    judges["context_relevancy"] = ContextRelevancyEvaluator(
        llm=evaluation_llm,
    )

    judges["faithfulness"] = FaithfulnessEvaluator(
        llm=evaluation_llm,
    )

    judges["semantic_similarity"] = SemanticSimilarityEvaluator()

    return judges

def create_evaluation_tasks(rag_dataset, prediction_data, judges):
    eval_tasks = []
    for example, prediction in zip(
        rag_dataset.examples, prediction_data.predictions
    ):
        eval_tasks.append(
            judges["answer_relevancy"].aevaluate(
                query=example.query,
                response=prediction.response,
                sleep_time_in_seconds=1.5,
            )
        )
        eval_tasks.append(
            judges["context_relevancy"].aevaluate(
                query=example.query,
                contexts=prediction.contexts,
                sleep_time_in_seconds=1.5,
            )
        )
        eval_tasks.append(
            judges["faithfulness"].aevaluate(
                query=example.query,
                response=prediction.response,
                contexts=prediction.contexts,
                sleep_time_in_seconds=1.5,
            )
        )
        eval_tasks.append(
            judges["semantic_similarity"].aevaluate(
                query=example.query,
                response="\n".join(prediction.contexts),
                reference="\n".join(example.reference_contexts),
                sleep_time_in_seconds=1.5,
            )
        )
    return eval_tasks

async def evaluate_tasks(eval_tasks):
    '''Batch await tasks to avoid rate limits on Tier1 OpenAI usage'''
    nest_asyncio.apply()

    EVALUATION_BATCH_SIZE = 10
    MINIMUM_INTERVAL = 10.0
    DELAY = 5.0

    eval_results = []
    for chunk in chunked(eval_tasks, EVALUATION_BATCH_SIZE):
        eval_result = await tqdm_asyncio.gather(*chunk, mininterval=MINIMUM_INTERVAL, delay=DELAY)
        eval_results.extend(eval_result)
    return eval_results

def display_generation_evaluation_results(eval_results):

    EVALUATIONS_PATH = "./qa_datasets/evaluations_2.json"

    # Bundle eval results into different metric categories
    evals = {
        "answer_relevancy": eval_results[::4],
        "context_relevancy": eval_results[1::4],
        "faithfulness": eval_results[2::4],
        "semantic_similarity": eval_results[3::4],
    }

    # Save evaluations to JSON
    evaluations_objects = {
        "semantic_similariy": [e.dict() for e in evals["semantic_similarity"]],
        "context_relevancy": [e.dict() for e in evals["context_relevancy"]],
        "faithfulness": [e.dict() for e in evals["faithfulness"]],
        "answer_relevancy": [e.dict() for e in evals["answer_relevancy"]],
    }
    with open(EVALUATIONS_PATH, "w") as json_file:
        json.dump(evaluations_objects, json_file)

    # Viewing evaluation results
    deep_eval_df, mean_answer_relevancy_df = get_eval_results_df(
        ["mean value"] * len(evals["answer_relevancy"]),
        evals["answer_relevancy"],
        metric="answer_relevancy",
    )
    deep_eval_df, mean_context_relevancy_df = get_eval_results_df(
        ["mean value"] * len(evals["context_relevancy"]),
        evals["context_relevancy"],
        metric="context_relevancy",
    )
    _, mean_faithfulness_df = get_eval_results_df(
        ["mean value"] * len(evals["faithfulness"]),
        evals["faithfulness"],
        metric="faithfulness",
    )
    _, mean_semantic_similarity_df = get_eval_results_df(
        ["mean value"] * len(evals["semantic_similarity"]),
        evals["semantic_similarity"],
        metric="semantic_similarity",
    )

    mean_scores_df = pd.concat(
        [
            mean_answer_relevancy_df.reset_index(),
            mean_context_relevancy_df.reset_index(),
            mean_faithfulness_df.reset_index(),
            mean_semantic_similarity_df.reset_index(),
        ],
        axis=0,
        ignore_index=True,
    )
    mean_scores_df = mean_scores_df.set_index("index")
    mean_scores_df.index = mean_scores_df.index.set_names(["metrics"])
    return mean_scores_df

# def display_pairwise_eval_df(query, response1, response2, eval_result) -> None:
#     eval_df = pd.DataFrame(
#         {
#             "Query": query,
#             "Reference Response (Answer 1)": response2,
#             "Current Response (Answer 2)": response1,
#             "Score": eval_result.score,
#             "Reason": eval_result.feedback,
#         },
#         index=[0],
#     )
#     eval_df = eval_df.style.set_properties(
#         **{
#             "inline-size": "300px",
#             "overflow-wrap": "break-word",
#         },
#         subset=["Current Response (Answer 2)", "Reference Response (Answer 1)"]
#     )
#     display(eval_df)