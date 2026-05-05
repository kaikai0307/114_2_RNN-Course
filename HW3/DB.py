from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from Chunking import load_and_chunk_data


def build_vector_db(documents, embeddings, collection_name, persist_directory):
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    print(f"Saved ChromaDB collection '{collection_name}' to {persist_directory}")
    return vector_db

EMBED_MODEL_NAME = "BAAI/bge-m3" # Or "sentence-transformers/all-MiniLM-L6-v2" for speed
print(f"Loading Embedding Model: {EMBED_MODEL_NAME} on CUDA...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': 'cuda'}, # Utilizing RTX 4090
    encode_kwargs={'normalize_embeddings': True}
)


# Build Vector DB
print("Building ChromaDB Index...")
chunk_size = 1024  # 512 or 1024
method = "both"
chunks_a, chunks_b = load_and_chunk_data(chunk_size, method)

chunks_a_vector_db = build_vector_db(
    documents=chunks_a,
    embeddings=embeddings,
    collection_name="chunks_a",
    persist_directory=f"db_{chunk_size}/chunks_a",
)

chunks_b_vector_db = build_vector_db(
    documents=chunks_b,
    embeddings=embeddings,
    collection_name="chunks_b",
    persist_directory=f"db_{chunk_size}/chunks_b",
)

print("Vector DB ready.")
