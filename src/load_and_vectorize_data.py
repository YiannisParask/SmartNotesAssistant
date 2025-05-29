from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import time

# Constants
CHUNK_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_path: str) -> list:
    """
    Load documents from the specified directory and split them into chunks.

    Args:
        data_path (str): Path to the directory containing markdown files.

    Returns:
        list: List of document chunks.
    """
    loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
        show_progress=True,
    )
    docs = loader.load()

    print(f"loaded {len(docs)} documents")

    chuck_overlap: int = np.round(CHUNK_SIZE * 0.1, 0)
    print(f"chunk_size: {CHUNK_SIZE}, chunk_overlap: {chuck_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=chuck_overlap,
    )

    return splitter.split_documents(docs)


def download_encoder() -> tuple:
    """
    Download and initialize the SentenceTransformer model.
    Returns:
        SentenceTransformer: Initialized SentenceTransformer model.
    """
    model_name: str = "BAAI/bge-large-en-v1.5"
    encoder = SentenceTransformer(model_name, device=DEVICE)

    # Get the model parameters and save for later.
    embedding_dim = encoder.get_sentence_embedding_dimension()
    max_seq_length_in_tokens = encoder.get_max_seq_length()

    print(f"model_name: {model_name}")
    print(f"EMBEDDING_DIM: {embedding_dim}")
    print(f"MAX_SEQ_LENGTH: {max_seq_length_in_tokens}")

    return encoder, embedding_dim


def save_to_milvus(embedding_dim, dict_list) -> None:
    mc = MilvusClient("local_milvus_db.db")
    collection_name = "MilvusDocs"
    mc.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        consistency_level="Eventually",
        auto_id=True,
        overwrite=True,
    )
    print("Start inserting entities")

    start_time = time.time()
    mc.insert(collection_name, data=dict_list, progress_bar=True)
    end_time = time.time()
    print(f"Milvus insert time for {len(dict_list)} vectors: ", end="")
    print(f"{round(end_time - start_time, 2)} seconds")


def main() -> None:
    data_path: str = "../../Personal-Cheat-Sheets"

    chunks = load_data(data_path)
    print(f"Total chunks created: {len(chunks)}")

    # Encoder input is doc.page_content as strings.
    list_of_strings: list = [
        doc.page_content for doc in chunks if hasattr(doc, "page_content")
    ]

    # Embedding inference using HuggingFace encoder.
    encoder, embedding_dim = download_encoder()
    embeddings = encoder.encode(list_of_strings, show_progress_bar=True)

    # Normalize the embeddings using numpy.
    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Milvus expects a list of `numpy.ndarray` of `numpy.float32` numbers.
    converted_values = list(map(np.float32, embeddings))

    # Create dict_list for Milvus insertion.
    dict_list = []
    for chunk, vector in zip(chunks, converted_values):
        # Assemble embedding vector, original text chunk, metadata.
        chunk_dict = {
            "chunk": chunk.page_content,
            "source": chunk.metadata.get("source", ""),
            "vector": vector,
        }
        dict_list.append(chunk_dict)

    # Save to Milvus.
    save_to_milvus(embedding_dim, dict_list)


if __name__ == "__main__":
    main()
