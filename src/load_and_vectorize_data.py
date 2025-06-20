from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import time
from typing import Any

# Constants
CHUNK_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLLECTION_NAME = "MilvusDocs"


def load_md_data(data_path: str) -> list:
    """Load documents from the specified directory and split them into chunks.

    Args:
        data_path (str): Path to the directory containing markdown files.

    Returns:
        list: List of document chunks.
    """
    loader: Any = DirectoryLoader(
        data_path,
        glob="**/*.md",
        show_progress=True,
    )
    docs: list = loader.load()

    print(f"loaded {len(docs)} documents")

    return docs


def load_pdf_data(data_path: str) -> list:
    """Load documents from a directory containing PDF files.

    Args:
        data_path (str): Path to the directory containing PDF files.

    Returns:
        list: List of loaded documents.
    """
    # Load documents from a directory containing PDF files
    loader: Any = PyPDFDirectoryLoader(path=data_path)
    docs: list = loader.load()

    print(f"loaded {len(docs)} documents")
    # DEBUG: Print the first 300 characters of each document
    # for doc in docs:
    #     print(f"source: {doc.metadata['source']}")
    #     print(doc.page_content[:300], "...\n")

    return docs


def load_embedding_model() -> tuple[object, int]:
    """Download and initialize the SentenceTransformer model.

    Returns:
        tuple: A tuple containing the encoder model and its embedding dimension.
    """
    model_name: str = "BAAI/bge-large-en-v1.5"
    encoder: Any = SentenceTransformer(model_name, device=DEVICE)  # type: ignore

    # Get the model parameters and save for later.
    embedding_dim: int = encoder.get_sentence_embedding_dimension()
    max_seq_length_in_tokens: int = encoder.get_max_seq_length()

    print(f"model_name: {model_name}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Max sequence length in tokens: {max_seq_length_in_tokens}")

    return encoder, embedding_dim


def encode_data(encoder: SentenceTransformer, docs: list) -> list:
    """Encode the documents into embeddings and prepare them for Milvus
    insertion.

    Args:
        encoder (SentenceTransformer): The embedding model to use.
        docs (list): List of documents to encode.

    Returns:
        list: List of dictionaries containing the encoded data ready for Milvus insertion.
    """
    chunk_size: int = 512
    chunk_overlap: float = np.round(chunk_size * 0.1, 0)
    print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")

    # Define the splitter
    child_splitter: Any = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split the documents into smaller chunks
    chunks: list = child_splitter.split_documents(docs)
    print(f"{len(docs)} docs split into {len(chunks)} child documents.")

    # Encoder input is doc.page_content as strings
    list_of_strings: list[str] = [
        doc.page_content for doc in chunks if hasattr(doc, "page_content")
    ]

    # Embedding inference using HuggingFace encoder.
    embeddings_tensor: torch.Tensor = torch.tensor(encoder.encode(list_of_strings))

    # Normalize the embeddings using PyTorch
    embeddings_tensor = embeddings_tensor / torch.linalg.norm(
        embeddings_tensor, dim=1, keepdim=True
    )
    embeddings: np.ndarray = embeddings_tensor.cpu().numpy()

    # Milvus expects a list of `numpy.ndarray` of `numpy.float32` numbers.
    converted_values = list(map(np.float32, embeddings))

    # Create dict_list for Milvus insertion.
    dict_list: list = []
    for chunk, vector in zip(chunks, converted_values):
        chunk_dict: dict = {
            "chunk": chunk.page_content,
            "source": chunk.metadata.get("source", ""),
            "vector": vector,
        }
        dict_list.append(chunk_dict)

    return dict_list


def save_to_milvus(dict_list: list, embedding_dim: int) -> None:
    """Save the vectorized data to a Milvus collection.

    Args:
        embedding_dim (int): Dimension of the embedding vectors.
        dict_list (list): List of dictionaries containing chunk data and vectors.
    """
    print("Saving to Milvus...")

    # Initialize the Milvus client.
    mc: Any = MilvusClient("../data/local_milvus_database.db")

    # Create a collection with flexible schema and AUTOINDEX.
    mc.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=embedding_dim,
        consistency_level="Eventually",
        auto_id=True,
        overwrite=True,
    )

    # Insert data into the Milvus collection.
    print("Start inserting entities into Milvus...")
    start_time: float = time.time()
    mc.insert(COLLECTION_NAME, data=dict_list, progress_bar=True)
    print(
        f"Milvus insert time for {len(dict_list)} vectors: "
        f"{time.time() - start_time:.2f} seconds"
    )


def main() -> None:
    data_path: str = "../../Personal-Cheat-Sheets"

    docs: list = load_md_data(data_path)

    encoder, embedding_dim = load_embedding_model()

    encoded_data: list = encode_data(encoder, docs)  # type: ignore

    save_to_milvus(encoded_data, embedding_dim)


if __name__ == "__main__":
    main()
