import ollama
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

qdrant_client = QdrantClient(url="http://localhost:6333")

VECTOR_DIM = 768
COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = "Cosine"

def get_embedding(text: str, embed_model="nomic-embed-text"):
    response = ollama.embeddings(model=embed_model, prompt=text)
    return response["embedding"]

def search_embeddings(query, embed_model="nomic-embed-text", top_k=3):
    query_embedding = get_embedding(query, embed_model)

    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )

        top_results = [
            {
                "file": result.payload.get('file', 'Unknown file'),
                "page": result.payload.get('page', 'Unknown page'),
                "chunk": result.payload.get('chunk', 'Unknown chunk'),
                "similarity": result.score
            }
            for result in search_result
        ]

        for result in top_results:
            print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_rag_response(query, context_results, llm):
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model=llm, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search(embed_model, llm):
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query, embed_model)

        response = generate_rag_response(query, context_results, llm)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search(embed_model="nomic-embed-text", llm="mistral:latest")
