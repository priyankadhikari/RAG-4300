import chromadb
import numpy as np
import ollama

chroma_client = chromadb.HttpClient(host="localhost", port=8000)

def get_collection(vector_dim):
    collection = chroma_client.get_or_create_collection(name=f"embedding_collection_{vector_dim}")
    return collection
def get_embedding(text: str, embed_model):
    response = ollama.embeddings(model=embed_model, prompt=text)
    return response["embedding"]


def search_embeddings(collection, query, embed_model, top_k=3):
    query_embedding = get_embedding(query, embed_model)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    top_results = []

    for doc_id, metadata, distance in zip(results['ids'][0], results['metadatas'][0], results['distances'][0]):
        if metadata is not None:
            file = metadata.get('file', 'Unknown file')
            page = metadata.get('page', 'Unknown page')
            chunk = metadata.get('chunk', 'Unknown chunk')

            top_results.append({
                "file": file,
                "page": page,
                "chunk": chunk,
                "similarity": distance
            })
        else:
            print(f"Warning: Missing metadata for document ID {doc_id}")

    for result in top_results:
        print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

    return top_results


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

    # Generate response using Ollama
    response = ollama.chat(
        model=llm, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search(collection, embed_model="nomic-embed-text", llm="mistral:latest"):
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(collection, query, embed_model)

        response = generate_rag_response(query, context_results, llm)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    collection = get_collection(vector_dim=768)
    interactive_search(collection, embed_model="nomic-embed-text", llm="mistral:latest")
