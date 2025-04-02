# Retrieval-Augmented Generation (RAG) System for DS4300 Notes

## Overview
This project implements a local Retrieval-Augmented Generation (RAG) system to facilitate querying and summarization of DS4300 course notes. The system:

1. **Ingests** a collection of documents (e.g., course notes, slides, and additional documentation).
2. **Indexes** these documents using embeddings and a vector database.
3. **Accepts user queries** and retrieves relevant context.
4. **Packages the context** into a prompt for a locally running Large Language Model (LLM).
5. **Generates a response** using the retrieved context and LLM.

## Features
- Supports multiple **vector databases**: Redis, Qdrant, and an additional database of choice.
- Compares different **embedding models** (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `InstructorXL`).
- Allows tuning of **chunking strategies** (chunk size, overlap, preprocessing).
- Compares **local LLMs** such as LLaMA 2 and Mistral.
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
- Additional vector database (as chosen by the team)

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
- **Other vector databases** should be installed as per their documentation.

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
- **Chunking**: `chunk_size=200, overlap=50`
- **Embedding models**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLMs**: `mistral:latest`, `llama2-7B`
- **Vector DB**: Redis, Qdrant

---
## Deliverables
- **GitHub Repository** (this project)
- **Slide Deck** analyzing results
- **Evaluation Metrics** (Robustness, Analysis, Pipeline Recommendation)

---
## Contributors
- **Team Members**: [Add names here]
- **Course**: DS4300
- **Instructor**: [Instructor Name]

---
## License
[Specify license, e.g., MIT License]

