import torch
from textual.app import App, ComposeResult
from textual.widgets import Input, Static
from textual.reactive import reactive
from src.perform_rag_search import RagSearch
from textual.containers import VerticalScroll
import asyncio


# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "MilvusDocs"
MILVUS_URI = "/home/yiannisparask/Projects/SmartNotesAssistant/data/local_milvus_database.db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Message(Static):
    def __init__(self, user: str, message: str, **kwargs):
        style = "bold blue" if user == "You" else "magenta"
        super().__init__(f"[{style}]{user}:[/] {message}", **kwargs)

class ChatApp(App):
    CSS = """
    Screen { align: center middle; }
    #chat { width: 80vw; height: 60vh; border: round yellow; padding: 1; }
    #input { width: 80vw; margin-top: 1; }
    """
    messages = reactive([])

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat_container")
        yield Input(placeholder="Type your message and press Enter...", id="input")


    def on_mount(self):
        # Initialize your RAG pipeline once!
        self.console.log("Loading models and vector store (this may take a minute)...")
        try:
            self.rag = RagSearch(
                milvus_uri=MILVUS_URI,
                device=DEVICE,
                collection_name=COLLECTION_NAME,
            )
            # Setup embeddings, retriever, LLM, prompt, chain
            self.rag.get_embeddings_model(EMBED_MODEL)
            retriever = self.rag.get_retriever()
            prompt = self.rag.build_prompt_template()
            self.rag.get_hg_llm(MODEL_NAME)
            self.qa_chain = self.rag.get_qa_chain(retriever, prompt)
            self.querying = False

            chat = self.query_one("#chat_container", VerticalScroll)
            chat.mount(Message("System", "Welcome! Type your question and press Enter."))
            self.console.log("RAG system initialized successfully!")

        except Exception as e:
            self.console.log(f"Error initializing RAG system: {e}")
            self.rag = None
            self.qa_chain = None
            self.querying = True  # Disable querying if initialization failed

            chat = self.query_one("#chat_container", VerticalScroll)
            chat.mount(Message("System", f"Error: Failed to initialize RAG system. {e}"))


    async def on_input_submitted(self, event: Input.Submitted):
        user_message = event.value.strip()
        if not user_message or self.querying:
            return
        self.querying = True
        chat = self.query_one("#chat_container", VerticalScroll)
        chat.mount(Message("You", user_message))
        event.input.value = ""
        await self.perform_llm_query(user_message, chat)


    async def perform_llm_query(self, prompt, chat):
        # Check if RAG system is properly initialized
        if self.rag is None or self.qa_chain is None:
            chat.mount(Message("System", "Error: RAG system not initialized. Please restart the application."))
            self.querying = False
            return

        torch.cuda.empty_cache()
        try:
            answer = await asyncio.to_thread(
                self.rag.perform_rag_query, self.qa_chain, prompt
            )
            chat.mount(Message("LLM", answer))
        except Exception as e:
            chat.mount(Message("System", f"Error processing query: {e}"))

        self.querying = False

        def scroll_to_end():
            chat.scroll_y = chat.virtual_size.height

        self.call_after_refresh(scroll_to_end)

    def on_unmount(self):
        """Clean up memory when the app is closed."""
        self.cleanup_memory()

    def cleanup_memory(self):
        """Clean up GPU memory and Python objects."""
        try:
            # Clear RAG components
            if hasattr(self, 'rag') and self.rag is not None:
                if hasattr(self.rag, 'text_generator') and self.rag.text_generator is not None:
                    self.rag.text_generator = None
                if hasattr(self.rag, 'embeddings_generator') and self.rag.embeddings_generator is not None:
                    self.rag.embeddings_generator = None
                self.rag = None

            if hasattr(self, 'qa_chain'):
                self.qa_chain = None

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Force garbage collection
            import gc
            gc.collect()

            self.console.log("Memory cleaned up successfully.")
        except Exception as e:
            self.console.log(f"Error during cleanup: {e}")


if __name__ == "__main__":
    ChatApp().run()
