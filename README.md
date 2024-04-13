## Project Overview

This project creates and compares two RAG pipelines. Each pipeline consists of a retrieval framework and a generation framework. The performance of these pipelines is evaluated a suite of metrics.

### Data Sources

- [NRMA Car Policy Booklet](https://www.nrma.com.au/sites/nrma/files/nrma/policy_booklets/nrma-car-pds-1023-east.pdf)
- [Allianz Policy Document](https://www.allianz.com.au/openCurrentPolicyDocument/POL011BA/$File/POL011BA.pdf)

## Technologies Used

- **Retrieval Tools:**
  - `OpenAI Embeddings`: Utilised for creating vector representations of text for vector search.
  - `Qdrant`: Hybrid database for managing and querying embeddings.
  - `Cohere`: Local vector database for managing embeddings.

- **Generation Models:**
  - `GPT-3.5`: Used for generating text in one of the RAG pipelines.
  - `BAAI BGE Model`: Another generation model used for comparative performance.

- **Other Tools:**
  - `LlamaIndex`: For chaining together different AI components into coherent pipelines.

## Evaluation Metrics

- **Retrieval Performance:**
  - `Hit Rate (HR)`: Measures the frequency of relevant document retrieval.
  - `Mean Reciprocal Rank (MRR)`: Provides an average of reciprocal ranks of the query results.

- **Generation Performance:**
  - `Faithfulness`: Assesses the accuracy of the generated text in preserving the intended facts.
  - `Relevancy`: Evaluates how relevant the generated text is to the input query.
  - `Context Score`: Measures how well the generated text fits into the larger context of the discussion.

## Setup Instructions

1. Clone the repository:

    git clone https://github.com/myz96/rai-assignment.git

2. Install dependencies:

    poetry install
    brew install --cask ollama
    "Open Ollama GUI to serve Ollama" 
    ollama pull gemma:2b 

3. Run the scripts to setup and evaluate the RAG pipelines:

    poetry run python rai_assignment/rag1.py
    poetry run python rai_assignment/rag2.py

