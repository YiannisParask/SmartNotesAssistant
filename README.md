# SmartNotesAssistant

The goal of this personal project is to develop a simple AI chat web app that
runs locally. Through a user-friendly `React` interface, you'll be able to
create a vector database (using `Milvus` or `Chroma` as the backend) to store
your notes and interact with an LLM (powered by the `vLLM` engine), which will
search your notes to provide answers. The backend, built with `FastAPI`, will
handle communication between the frontend and the AI services.

## Acknowledgements

Inspirations:

- [Building RAG Applications with Milvus, Qwen, and
  vLLM](https://zilliz.com/blog/build-rag-app-with-milvus-qwen-and-vllm)
- [Building RAG with Milvus, vLLM, and Llama
  3.1](https://milvus.io/docs/milvus_rag_with_vllm.md)
- [Retrieval-Augmented Generation (RAG) with Milvus and
  LangChain](https://milvus.io/docs/integrate_with_langchain.md)
