import torch
from src.load_and_vectorize_data import LoadAndVectorizeData
from src.perform_rag_search import RagSearch

# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "MilvusDocs"
MILVUS_URI = "/home/yiannisparask/Projects/SmartNotesAssistant/data/local_milvus_database.db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_vectorize_data(data_path: str) -> None:

    # Initialize the data loader
    loader = LoadAndVectorizeData(
        data_path=data_path,
        collection_name=COLLECTION_NAME,
        device=DEVICE,
        milvus_uri=MILVUS_URI
    )

    # Load and process documents
    docs = loader.load_md_data()
    chunks = loader.slit_docs(docs)
    embeddings_model = loader.get_embeddings_model(embeddings_model=EMBED_MODEL)

    # Save to Milvus
    loader.save_to_milvus(chunks, embeddings_model)


def perform_rag_search() -> None:
    torch.cuda.empty_cache()

    # Example RAG search
    rag_search = RagSearch(
        milvus_uri=MILVUS_URI,
        device=DEVICE,
        collection_name=COLLECTION_NAME,
    )

    rag_search.get_embeddings_model(EMBED_MODEL)
    rag_search.get_hg_llm(MODEL_NAME)

    # Build QA chain
    retriever = rag_search.get_retriever()
    prompt = rag_search.build_prompt_template()
    qa_chain = rag_search.get_qa_chain(retriever, prompt)

    # Test query
    question = "What is Big Data?"
    answer, sources = rag_search.perform_rag_query(qa_chain, question)

    print("📝 Answer:\n", answer)
    print("\n📚 Sources:")
    for src in sources:
        print(f" • {src['source']}: {src['text'][:200]}...")


def main() -> None:
    data_path: str = "/home/yiannisparask/Projects/Personal-Cheat-Sheets"
    #load_and_vectorize_data(data_path)
    perform_rag_search()


if __name__ == "__main__":
    main()
