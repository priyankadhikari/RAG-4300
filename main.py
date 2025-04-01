from config import chunking_strategies, embedding_models, vector_dbs, llm_models
from utils import measure_time_memory, write_to_csv

import ingest
import search
import ingest_chroma

def run_ingest_pipeline(vector_db, embed_model, chunk_size, overlap):
    if vector_db == "redis":
        ingest.clear_redis_store()
        ingest.create_hnsw_index()
        ingest.process_pdfs("Data", embed_model=embed_model, chunk_size=chunk_size, overlap=overlap)
    elif vector_db == "chroma":
        ingest_chroma.clear_chroma_store()
        # process pdfs
    elif vector_db == "qdrant":
        #qdrant stuff
    print("\n---Done processing PDFs---\n")

def run_search_pipeline(vector_db, embed_model, llm_model, test_query="test query"):
    if vector_db == "redis":
        context_results = search.search_embeddings(test_query, embed_model)
        response = search.generate_rag_response(test_query, context_results, llm_model)
    elif vector_db == "chroma":
        # search for chroma
    elif vector_db == "qdrant":
        # search for qdrant
    return response


def run_experiments():
    """
    Iterates over all configurations, running both ingestion and search pipelines,
    measuring time and memory usage, and writing results to a CSV.
    """
    for vector_db in vector_dbs:
        for embed_model in embedding_models:
            for strategy in chunking_strategies:
                chunk_size = strategy["chunk_size"]
                overlap = strategy["overlap"]
                for llm_model in llm_models:
                    print(
                        f"Running: DB={vector_db}, Embed={embed_model}, Chunk={chunk_size}, Overlap={overlap}, LLM={llm_model}")

                    # Measure ingestion pipeline performance.
                    ingest_time, ingest_mem, _ = measure_time_memory(
                        run_ingest_pipeline, vector_db, embed_model, chunk_size, overlap
                    )

                    # Measure search pipeline performance using a fixed test query.
                    search_time, search_mem, search_result = measure_time_memory(
                        run_search_pipeline, vector_db, embed_model, llm_model, "test query"
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