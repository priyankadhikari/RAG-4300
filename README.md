# Retrieval-Augmented Generation (RAG) System for DS4300 Notes

## Overview
This project implements a local Retrieval-Augmented Generation (RAG) system to facilitate querying and summarization of DS4300 course notes. The system:

1. **Ingests** a collection of documents (e.g., course notes, slides, and additional documentation).
2. **Indexes** these documents using embeddings and a vector database.
3. **Accepts user queries** and retrieves relevant context.
4. **Packages the context** into a prompt for a locally running Large Language Model (LLM).
5. **Generates a response** using the retrieved context and LLM.

## Features
- Supports multiple **vector databases**: Redis, Chroma, and Qdrant.
- Compares different **embedding models**:
  - `nomic-embed-text`
  - `all-minilm`
  - `mxbai-embed-large`
- Allows tuning of **chunking strategies**:
  - **Sizes**: 50, 200, 500 tokens
  - **Overlaps**: 0, 50, 100 tokens
- Compares **local LLMs**:
  - `Mistral`
  - `Llama-2`
- Evaluates indexing/querying **performance metrics** (speed, memory usage, retrieval quality).

---
## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required Python packages (install using `requirements.txt`)
- Ollama (for running local LLMs)
- Redis (for Redis Vector DB)
- Qdrant (for Qdrant Vector DB)
- Chroma (for Chroma Vector DB)

### Setup

1. **Clone the repository**
```bash
    git clone https://github.com/yourusername/DS4300-RAG-System.git
    cd DS4300-RAG-System
```

2. **Install dependencies**
```bash
    pip install -r requirements.txt
```

3. **Start necessary services**
- **Redis**
```bash
    redis-server
```
- **Qdrant** (Docker-based setup)
```bash
    docker run -p 6333:6333 -d qdrant/qdrant
```
- **Chroma**
```bash
    python -m chromadb run
```

---
## Usage

### 1. Data Ingestion & Indexing
To process and index course notes:
```bash
    python ingest.py
```
This will:
- Load and preprocess documents (PDFs, text files).
- Generate embeddings.
- Store embeddings in the selected vector database.

### 2. Querying and Searching
Run the interactive search interface:
```bash
    python search_qdrant.py
```
This allows users to enter queries, retrieve context, and generate responses from the LLM.

### 3. Experimentation
Modify parameters such as:
- `chunk_size` and `overlap` in `ingest.py`
- `embedding_model` in `search_qdrant.py`
- `vector database` choice
- `local LLM` (Mistral, LLaMA 2, etc.)

### 4. Performance Evaluation
To measure indexing/search performance:
```bash
    python utils.py
```
This records time, memory usage, and experiment results in `experiment_results.csv`.

---
## Configuration
Modify `config.py` (if applicable) or update parameters directly in scripts:
- **Chunking**:
  - Sizes: 50, 200, 500 tokens
  - Overlaps: 0, 50, 100 tokens
- **Embedding models**:
  - `Nomic-embed-text`
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `mxbai-embed-large`
- **LLMs**:
  - `Mistral`
  - `Llama-2`
- **Vector DB**:
  - Redis, Qdrant, Chroma

---
## Test Queries
The following test queries were used to evaluate retrieval effectiveness and LLM responses:
1. **Write a PyMongo query** to find documents where the Customer’s address starts with the letter "S" or higher for Customers (“customers”) in the database (“mydatabase”). Only use the documents provided. [Reference](https://www.w3schools.com/python/python_mongodb_query.asp)
2. **What is the capital of Tennessee?** Only use the documents provided.
3. **Provide a 2-sentence summary on Redis and its functionality.** Only use the documents provided.
4. **Compare and contrast Redis and Neo4j.** Only use the documents provided.

---
## Deliverables
- **GitHub Repository** (this project)
- **Slide Deck** analyzing results
- **Evaluation Metrics** (Robustness, Analysis, Pipeline Recommendation)

---
## Team
- **Team Members**: Priyanka Adhikari, Ruchira Banerjee, and Nidhi Bendre
- **DS4300 Group**: Ruchidhiyanka

