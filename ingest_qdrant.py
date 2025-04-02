import qdrant_client
import os
import ollama
from text_preprocessing import get_text, split_chunks
from qdrant_client.models import VectorParams, Distance
import uuid
import config
qdrant_client = qdrant_client.QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = Distance.COSINE

def clear_qdrant_collection():
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting collection: {e}")

def create_qdrant_collection(vector_dim):
    try:
        vectors_config = VectorParams(
            size=vector_dim,
            distance=DISTANCE_METRIC
        )

        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_config
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print(f"Error creating collection: {e}")


def get_embedding(text: str, embed_model="nomic-embed-text") -> list:
    response = ollama.embeddings(model=embed_model, prompt=text)
    return response["embedding"]


def store_embedding(file: str, page: str, chunk: str, embedding: list):
    point_id = str(uuid.uuid4())

    payload = {
        "file": file,
        "page": page,
        "chunk": chunk
    }

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[{
            "id": point_id,
            "vector": embedding,
            "payload": payload
        }]
    )
    print(f"Stored embedding for: {chunk}")


def process_pdfs(data_dir, embed_model="nomic-embed-text", chunk_size=50, overlap=0):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = get_text(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_chunks(text, chunk_size=chunk_size, overlap=overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, embed_model)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk_index),
                        embedding=embedding,
                    )
            print(f"-----> Processed {file_name}")


def main():
    clear_qdrant_collection()
    create_qdrant_collection()
    process_pdfs("Data", embed_model="nomic-embed-text", chunk_size=50, overlap=0)
    print("\n--- Done processing PDFs ---\n")


if __name__ == "__main__":
    main()
