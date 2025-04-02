# Chunking strategies
chunking_strategies = [
    {"chunk_size": 50, "overlap": 0},
    {"chunk_size": 100, "overlap": 10},
    {"chunk_size": 200, "overlap": 20}
]

# Embedding models
embedding_models = [
    {"model_name": "nomic-embed-text", "vector_dim" : 768},
    {"model_name": "all-minilm", "vector_dim" : 384},
    {"model_name":"mxbai-embed-large", "vector_dim" : 1024},
]

# Vector databases
vector_dbs = [
    #"redis",
    "chroma",
    "qdrant"
]

# LLM models
llm_models = [
    "mistral:latest",
    "llama2:latest"
]