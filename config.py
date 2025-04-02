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
    "redis",
    "chroma",
    "qdrant"
]

# LLM models
llm_models = [
    "mistral:latest",
    "llama2:latest"
]

# Queries to test
queries = [
    "1. Write a Pymongo query to find the documents where a Customer’s address starts with the letter "
    "'S' or higher for Customers ('customers') in a database ('mydatabase'). Only use the documents provided.",
    "2. What is the capital of Tennessee? Only use the documents provided.",
    "3. Provide a 2-sentence summary on Redis and its functionality. Only use the documents provided.",
    "4. Compare and contrast Redis and Neo4j. Only use the documents provided."
]