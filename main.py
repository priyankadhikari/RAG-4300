from config import chunking_strategies, embedding_models, vector_dbs, llm_models
from utils import measure_time_memory, write_to_csv
import ingest
import search
import search_chroma
import search_qdrant
import ingest_chroma
import ingest_qdrant

def run_ingest_pipeline(vector_db, embed_model, chunk_size, overlap, vector_dim):
    if vector_db == "redis":
        ingest.clear_redis_store()
        ingest.create_hnsw_index(vector_dim)
        ingest.process_pdfs("Data", embed_model=embed_model, chunk_size=chunk_size, overlap=overlap)
    elif vector_db == "chroma":
        ingest_chroma.clear_chroma_store(vector_dim)
        collection = ingest_chroma.get_or_create_collection(vector_dim)
        ingest_chroma.process_pdfs(collection=collection,data_dir="Data", embed_model=embed_model, chunk_size=chunk_size, overlap=overlap)
    elif vector_db == "qdrant":
        ingest_qdrant.clear_qdrant_collection(vector_dim=vector_dim)
        ingest_qdrant.create_qdrant_collection(vector_dim=vector_dim)
        ingest_qdrant.process_pdfs(vector_dim=vector_dim,data_dir="Data", embed_model=embed_model, chunk_size=chunk_size, overlap=overlap)
    print("\n---Done processing PDFs---\n")

def run_search_pipeline(vector_db, embed_model, llm_model, vector_dim, test_query="test query"):
    if vector_db == "redis":
        context_results = search.search_embeddings(test_query, embed_model)
        response = search.generate_rag_response(test_query, context_results, llm_model)
    elif vector_db == "chroma":
        collection = search_chroma.get_collection(vector_dim=vector_dim)
        context_results = search_chroma.search_embeddings(collection, test_query, embed_model)
        response = search_chroma.generate_rag_response(test_query, context_results, llm_model)
    elif vector_db == "qdrant":
        context_results = search_qdrant.search_embeddings(vector_dim, test_query, embed_model)
        response = search_qdrant.generate_rag_response(test_query, context_results, llm_model)
    return response

def run_experiments():
    """
    Iterates over all configurations, running both ingestion and search pipelines,
    measuring time and memory usage, and writing results to a CSV.
    """
    for vector_db in vector_dbs:
        for embedding_model in embedding_models:
            embed_model = embedding_model["model_name"]
            vector_dim = embedding_model["vector_dim"]
            for strategy in chunking_strategies:
                chunk_size = strategy["chunk_size"]
                overlap = strategy["overlap"]
                for llm_model in llm_models:
                    print(
                        f"Running: DB={vector_db}, Embed={embed_model}, Chunk={chunk_size}, Overlap={overlap}, LLM={llm_model}")

                    # Measure ingestion pipeline performance.
                    ingest_time, ingest_mem, _ = measure_time_memory(
                        run_ingest_pipeline, vector_db, embed_model, chunk_size, overlap, vector_dim
                    )

                    # Measure search pipeline performance using a fixed test query.
                    search_time, search_mem, search_result = measure_time_memory(
                        run_search_pipeline, vector_db, embed_model, llm_model, vector_dim, "test query"
                    )

                    # Record the results into a CSV row.
                    row = [
                        vector_db, llm_model, embed_model, chunk_size, overlap,
                        ingest_time, ingest_mem, search_time, search_mem
                    ]
                    write_to_csv(row)

                    print(
                        f"  Ingest: {ingest_time:.2f}s, {ingest_mem:.2f}MB | Search: {search_time:.2f}s, {search_mem:.2f}MB\n")

if __name__ == "__main__":
    run_experiments()