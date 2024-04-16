## Project Overview

This project creates and compares two RAG pipelines. Each pipeline consists of a retrieval framework and a generation framework. The performance of these pipelines is evaluated by a suite of metrics.

### Data Sources

- [NRMA Car Policy Booklet](https://www.nrma.com.au/sites/nrma/files/nrma/policy_booklets/nrma-car-pds-1023-east.pdf)
- [Allianz Policy Document](https://www.allianz.com.au/openCurrentPolicyDocument/POL011BA/$File/POL011BA.pdf)

## Technologies Used

- **Retrieval Tools:**

  - `OpenAI Embeddings`: Proprietary model by OpenAI for creating vector representations of text.
  - `BAAI BGE Model`: A open-source embedding model used for comparative performance.
  - `Chroma`: Vector search database for managing and querying embeddings.
  - `Weaviate`: Open-source hybrid search database for managing embeddings.

- **Generation Models:**

  - `GPT-3.5`: Large, proprietary generation model used for text generation by OpenAI.
  - `Gemma 2b`: Small, open-source and locally hosted generation model by Ollama.

- **Other Tools:**
  - `LlamaIndex`: For chaining together different AI components into coherent pipelines.

## Evaluation Metrics

- **Retrieval Performance:**

  - `Hit Rate (HR)`: Measures the frequency of relevant document retrieval.
  - `Mean Reciprocal Rank (MRR)`: Provides an average of reciprocal ranks of the query results.

- **Generation Performance:**
  - `Faithfulness`: Measure of hallucination, relevance of the answer to context.
  - `Relevancy`: Evaluates how relevant the generated text is to the input query.
  - `Context Score`: Measures how well the retrieved context fits to the retrieval query.
  - `Semantic Similarity`: Measures similarity between queries and response embeddings.

## Setup Instructions

1. Clone the repository:

   ```console
   git clone https://github.com/myz96/rai-assignment.git
   ```

2. Install dependencies:

   First, [install poetry](https://python-poetry.org/docs/#installation)

   Then, install dependencies

   ```console
   poetry install
   brew install --cask ollama
   ```

   Open Ollama GUI to serve Ollama model

   Then, download Gemma model

   ```console
   ollama pull gemma:2b
   ```

3. Run the scripts to setup and evaluate the RAG pipelines:

   ```console
   poetry run python rai_assignment/rag1.py
   poetry run python rai_assignment/rag2.py
   ```
