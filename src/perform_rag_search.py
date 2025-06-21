import torch
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from typing import Any

# Constants
MODELNAME = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "MilvusDocs"
MILVUS_PATH = "../data/local_milvus_database.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_query(query: str, encoder: SentenceTransformer) -> np.ndarray:
    """Encode the query using the provided SentenceTransformer model.

    Args:
        query (str): The query string to encode.
        encoder (SentenceTransformer): The SentenceTransformer model to use for encoding.

    Returns:
        np.ndarray: The normalized embedding of the query.
    """
    emb = encoder.encode([query])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype(np.float32)[0]


def search_milvus(query_emb: np.ndarray, top_k: int = 5) -> tuple[Any, Any]:
    """Search the Milvus database for the top_k most similar contexts to the
    query embedding.

    Args:
        query_emb (np.ndarray): The normalized embedding of the query.
        top_k (int): The number of top results to return.

    Returns:
        tuple: A tuple containing two lists:
            - contexts: The top_k most similar contexts.
            - sources: The sources of the top_k contexts.
    """
    mc: Any = MilvusClient(MILVUS_PATH)
    results: list = mc.search(
        collection_name=COLLECTION_NAME,
        data=[query_emb],
        limit=top_k,
        output_fields=["chunk", "source"],
    )
    hits = results[0]
    contexts = [hit["chunk"] for hit in hits]
    sources = [hit["source"] for hit in hits]
    return contexts, sources


def build_prompt(contexts, sources, question) -> str:
    """Build a prompt for the LLM using the provided contexts, sources, and
    question.

    Args:
        contexts (list): A list of context strings.
        sources (list): A list of source strings corresponding to the contexts.
        question (str): The user's question.

    Returns:
        str: The formatted prompt string.
    """
    contexts_combined = " ".join(reversed(contexts))
    source_combined = ", ".join(reversed(list(dict.fromkeys(sources))))
    prompt = f"""First, check if the provided Context is relevant to the user's
              question. Second, only if the provided Context is strongly
              relevant, answer the question using the Context.
              Otherwise, if the Context is not strongly relevant, answer the
              question without using the Context. Be clear, concise, relevant.
              Answer clearly, in fewer than 2 sentences.
            Grounding sources: {source_combined}
            Context: {contexts_combined}
            User's question: {question}
            """
    return prompt


# TODO: Add argument parsing for command line usage
def main() -> None:
    # sample question
    question: str = "What is Big Data?"

    encoder: Any = SentenceTransformer(EMBED_MODEL, device=DEVICE)  # type: ignore
    encoded_query: np.ndarray = encode_query(question, encoder)
    contexts, sources = search_milvus(encoded_query, top_k=5)
    prompt: str = build_prompt(contexts, sources, question)

    llm: Any = LLM(
        model=MODELNAME,
        device=DEVICE,
    )
    sampling_params: Any = SamplingParams(
        temperature=0.1,
        top_p=0.95,
    )

    outputs: list = llm.generate([prompt], sampling_params)

    for output in outputs:
        print(f"Question: {question!r}")
        print(f"Generated text: {output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
