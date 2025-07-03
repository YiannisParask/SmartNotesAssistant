import torch
import gc
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_milvus import Milvus
from langchain_community.llms import VLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Tuple, Dict, Any
from transformers.pipelines import pipeline

# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "MilvusDocs"
MILVUS_URI = "/home/yiannisparask/Projects/SmartNotesAssistant/data/local_milvus_database.db"  # or milvus://<usr>:<pwd>@host:port
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Download and initialize the HuggingFace embeddings model.

    Returns:
        HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings with the specified model.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        show_progress=False,
    )
    return embeddings


def get_retriever(
    uri: str,
    embeddings: HuggingFaceEmbeddings,
    collection_name: str = COLLECTION_NAME,
    k: int = TOP_K,
) -> Any:
    """Create a Milvus vectorstore retriever using LangChain and return it.

    Args:
        uri (str): URI for the Milvus database.
        embeddings (SentenceTransformerEmbeddings): Embedding function to use.
        collection_name (str): Name of the Milvus collection.
        k (int): Number of top results to return.

    Returns:
        Any: A retriever object that can be used to query the Milvus vectorstore.
    """
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={
            "uri": uri,
        },
        text_field="chunk",
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_prompt_template() -> PromptTemplate:
    """Create a PromptTemplate for RAG QA."""
    template = (
        "First, check if the provided Context is relevant to the user's question.\n"
        "Second, only if the provided Context is strongly relevant, answer the question using the Context.\n"
        "Otherwise, if the Context is not strongly relevant, answer the question without using the Context.\n"
        "Be clear, concise, in fewer than 2 sentences.\n\n"
        "### Context:\n{context}\n\n"
        "### Question:\n{question}\n\n"
        "### Answer:"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)


def get_vllm() -> VLLM:
    """Instantiate the VLLM LLM wrapper."""
    return VLLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        # max_model_len=1000,
        enforce_eager=True,
        # top_k=TOP_K,
        temperature=0.1,
        max_new_tokens=500,
    )


def get_hg_llm():
    llm_pipeline = pipeline(task="text-generation", model=MODEL_NAME)
    hf_llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return hf_llm


def get_qa_chain(llm: VLLM, retriever, prompt: PromptTemplate) -> RetrievalQA:
    """Build and return a RetrievalQA chain with custom prompt."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def perform_rag_query(
    chain: RetrievalQA, query: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the RetrievalQA chain on the given query.

    Args:
        chain (RetrievalQA): The RetrievalQA chain to use.
        query (str): The user's query string.

    Returns:
        Tuple[str, List[Dict[str, Any]]]: A tuple containing the answer string and a list of source documents.
    """
    result = chain.invoke({"query": query})
    answer = result["result"].strip()
    sources = [
        {"source": doc.metadata.get("source", "<unknown>"), "text": doc.page_content}
        for doc in result["source_documents"]
    ]
    return answer, sources


def main() -> None:
    # Prepare components
    embeddings = get_embedding_function()
    retriever = get_retriever(MILVUS_URI, embeddings)
    prompt = build_prompt_template()

    # Clear cache and collect garbage
    gc.collect()
    torch.cuda.empty_cache()

    llm = get_hg_llm()
    # llm = get_vllm()

    # Build chain
    qa_chain: RetrievalQA = get_qa_chain(llm, retriever, prompt)

    # Sample query
    question = "What is Big Data?"
    answer, sources = perform_rag_query(qa_chain, question)

    # Print output
    print("📝 Answer:\n", answer)
    print("\n📚 Sources:")
    for src in sources:
        print(f" • {src['source']}: {src['text'][:200]!r}...")


if __name__ == "__main__":
    main()
