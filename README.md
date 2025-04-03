# DS4300 Retrieval-Augmented Generation System - Ruchidhiyanka

## Overview
This project implements a local Retrieval-Augmented Generation system to answer questions and summarize our group's DS4300 course notes and prepared documents for the purpose of assisting on the exam. 

The project:
1. Ingests a collection of documents, notes, and PDFs.
2. Indexes these documents using embeddings and vector databases.
3. Accepts user queries and retrieves relevant context.
4. Packages the context as a prompt for a local large language model (LLM).
5. Generates a response using the retrieved context and LLM.
6. Evaluates the models based on performance metrics like ingest and search speed, memory usage, and retrieval quality.

## Features we Tested
- 3 vector databases: Redis, Chroma, and Qdrant.
- 3 different embedding models:
  - `nomic-embed-text`
  - `all-minilm`
  - `mxbai-embed-large`
- 9 variations of chunking strategies:
  - **Sizes**: 50, 100, 200 tokens
  - **Overlaps**: 0, 10, 20 tokens
- 2 local LLMs:
  - `Mistral`
  - `Llama-2`

### Setup

1. **Start necessary containers**
- **Redis**

- **Qdrant**

- **Chroma**

2. **Download and use Ollama**

## Usage
### 1. Running All Experiments
To run all experiments with the predefined configurations:
```
python main.py
```

This will:
- Test all combinations of vector databases, embedding models, chunking strategies, and LLMs
- Process the PDF documents in the Data directory
- Measure performance metrics for each configuration
- Output results to experiment_results.csv

### 2. Individual Component Testing
#### Testing just the ingestion pipeline:
```bash
# Example for Redis with nomic-embed-text
python ingest.py
```

#### Testing search with a specific vector database:
```bash
# For Redis
python search.py
# For Qdrant
python search_qdrant.py
# For Chroma
python search_chroma.py
```

### 3. Customizing Experiments
Modify `config.py` to adjust:
- Chunking strategies
- Embedding models
- Vector databases
- LLM models
- Test queries

## Experiment Configuration
The current configuration in `config.py` specifies:

```python
# Chunking strategies
chunking_strategies = [
    {"chunk_size": 50, "overlap": 0},
    {"chunk_size": 100, "overlap": 10},
    {"chunk_size": 200, "overlap": 20}
]

# Embedding models
embedding_models = [
    {"model_name": "nomic-embed-text", "vector_dim": 768},
    {"model_name": "all-minilm", "vector_dim": 384},
    {"model_name": "mxbai-embed-large", "vector_dim": 1024},
]

# Vector databases
vector_dbs = [
    "redis",
    "chroma",
    "qdrant"
]

# LLM models
llm_models = [
    "mistral:latest",
    "llama2:latest"
]
```

## Analyzing Results
After running the experiments, you can analyze the results in `experiment_results.csv`:

```bash
# Example: visualize results with the provided code
python visualize_results.py
```

Based on our analysis, our top performing pipeline is:
1. Qdrant + llama2:latest + all-minilm (chunk size: 200, overlap: 20)

## Team
- **Team Members**: Priyanka Adhikari, Ruchira Banerjee, and Nidhi Bendre
- **DS4300 Group**: Ruchidhiyanka
