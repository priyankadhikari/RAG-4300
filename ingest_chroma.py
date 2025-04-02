import ollama
import chromadb
import numpy as np
import os
from text_preprocessing import get_text, split_chunks

chroma_client = chromadb.HttpClient(host="localhost", port=8000)

def clear_chroma_store(vector_dim):
    print("Clearing existing Chroma store...")
    chroma_client.get_or_create_collection(name=f"embedding_collection_{vector_dim}")
    chroma_client.delete_collection(name=f"embedding_collection_{vector_dim}")
    print("Chroma store cleared.")

def get_or_create_collection(vector_dim):
    collection_name = f"embedding_collection_{vector_dim}"
    collection = chroma_client.get_or_create_collection(name=collection_name)
    return collection

def get_embedding(text, model):
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def store_embedding(collection, file: str, page: str, chunk: str, embedding: list):
    doc_id = f"{file}_page_{page}_chunk_{chunk}"
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
    )
    print(f"Stored embedding for: {chunk}")


def process_pdfs(collection, data_dir, embed_model, chunk_size=50, overlap=0):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = get_text(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_chunks(text, chunk_size=chunk_size, overlap=overlap)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk, embed_model)
                    store_embedding(collection=collection,
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

def query_chroma(collection, query_text: str):
    embedding = get_embedding(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=5)

    for doc_id, metadata, distance in zip(results["ids"][0], results["metadatas"][0], results["distances"][0]):
        print(f"{doc_id} \n ----> Distance: {distance}\n Metadata: {metadata}\n")


def main():
    collection = get_or_create_collection(vector_dim=348)
    process_pdfs(collection=collection,data_dir="Data", embed_model="nomic-embed-text", chunk_size=50, overlap=0)
    print("\n---Done processing PDFs---\n")


if __name__ == "__main__":
    main()

