import os
from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from textual.app import App
from textual.widgets import Input, LoadingIndicator
from textual.reactive import reactive
from textual.binding import Binding
from src.perform_rag_search import RagSearch
from src.load_and_vectorize_data import LoadAndVectorizeData
from textual.containers import VerticalScroll
import asyncio
import gc
import logging
from src.views.chat_screen import ChatScreen
from src.views.setup_screen import SetupScreen
from src.widgets.chat_message import ChatMessage

# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "MilvusDocs"
CWD: str = os.getcwd()
MILVUS_URI: str = os.path.join(CWD, "data", "local_milvus_database.db")
LAST_PATH_FILE: str = os.path.join(CWD, "data", "last_path.txt")
DEVICE: Any = "cuda" if torch.cuda.is_available() else "cpu"


class ChatApp(App[None]):
    """Central coordinator for the Smart Notes Assistant Textual UI."""
    messages: reactive[list[str]] = reactive([])
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit the app"),
        Binding(key="d", action="toggle_dark", description="Toggle dark mode"),
        Binding(key="s", action="toggle_settings", description="Settings")
    ]

    milvus_uri: str = MILVUS_URI


    async def on_mount(self) -> None:
        """ `Textual` method. Initialize the RAG system and set up the UI based
        on whether the MilvusLiteDB file exists.
        """
        # Configure logging
        logging.basicConfig(
            filename="assistant.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        self.querying = False
        self.rag = None
        self.retriever = None

        # Show setup if the Milvus Lite DB file isn't present yet.
        needs_setup: bool = not os.path.exists(MILVUS_URI)

        if needs_setup:
            await self.push_screen(ChatScreen())
            await self.push_screen(SetupScreen())
        else:
            await self.push_screen(ChatScreen())


    async def initialize_rag(self) -> None:
        """ Initialize the RAG pipeline components. This should only be called
        once after the index is built.
        """
        # Initialize RAG pipeline once!
        chat = self.screen.query_one("#chat_container", VerticalScroll)
        init_loader: LoadingIndicator = self.screen.query_one("#init_loader", LoadingIndicator)
        chat_input: Input = self.screen.query_one("#input", Input)

        init_loader.display = True
        chat.display = False
        chat_input.display = False

        logging.info("Loading models and vector store (this may take a minute)...")

        try:
            def _load():
                loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)

                    self.rag = RagSearch(
                        milvus_uri=MILVUS_URI,
                        device=DEVICE,
                        collection_name=COLLECTION_NAME,
                    )
                    # Setup embeddings, retriever, LLM, prompt, chain
                    self.rag.get_embeddings_model(EMBED_MODEL)
                    self.retriever = self.rag.get_retriever()
                    self.rag.setup_agent(model_name="qwen2.5:1.5b")
                finally:
                    try:
                        loop.close()
                    finally:
                        asyncio.set_event_loop(None)

            # Offload heavy model loading to a background thread so the loader spins
            await asyncio.to_thread(_load)

            self.querying = False
            chat.mount(
                ChatMessage("System", "Welcome! Type your question and press Enter.")
            )
            logging.info("RAG system initialized successfully!")

        except Exception as e:
            logging.error(f"Error initializing RAG system: {e}")
            self.rag = None
            self.retriever = None
            self.querying = True  # Disable querying if initialization failed
            chat.mount(
                ChatMessage("System", f"Error: Failed to initialize RAG system. {e}")
            )

        finally:
            init_loader.display = False
            chat.display = True
            chat_input.display = True
            self.set_focus(chat_input)


    def build_index(self, data_dir: str) -> None:
        """ Build the Milvus Lite index from the specified data directory.
            Args:
                data_dir (str): Path to the directory containing markdown files.
            Returns:
                None
        """
        # Ensure the Milvus Lite data directory exists
        os.makedirs(os.path.dirname(MILVUS_URI), exist_ok=True)

        # Create an event loop for this worker thread (needed by libs calling get_event_loop)
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)

            lvd = LoadAndVectorizeData(
                data_path=data_dir,
                collection_name=COLLECTION_NAME,
                device=DEVICE,
                milvus_uri=MILVUS_URI,
            )
            md_docs: list[Any] = lvd.load_md_data()

            chunks: list[Any] = lvd.split_docs(md_docs)
            embeddings: HuggingFaceEmbeddings = lvd.get_embeddings_model(EMBED_MODEL)
            lvd.save_to_milvus(chunks, embeddings)
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)


    def get_last_data_path(self) -> str:
        """Get the last used data folder path from a file, if it exists.

        Returns:
            str: The last used data folder path.
        """
        if os.path.exists(LAST_PATH_FILE):
            try:
                with open(LAST_PATH_FILE, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                logging.error(f"Error reading last path file: {e}")
                pass
        return ""


    def save_last_data_path(self, path: str) -> None:
        """Save the last used data folder path to a file.

        Args:
            path (str): The data folder path to save.
        """
        os.makedirs(os.path.dirname(LAST_PATH_FILE), exist_ok=True)
        try:
            with open(LAST_PATH_FILE, "w", encoding="utf-8") as f:
                f.write(path)
        except Exception as e:
            logging.error(f"Error saving last path file: {e}")


    def reset_rag(self) -> None:
        """Reset and release RAG and Milvus connections."""
        self.rag = None
        self.retriever = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def delete_db(self) -> None:
        """Close connections and delete the Milvus Lite database file."""
        self.reset_rag()
        if os.path.exists(self.milvus_uri):
            try:
                os.remove(self.milvus_uri)
                logging.info(f"Deleted Milvus Lite database file: {self.milvus_uri}")
            except Exception as e:
                logging.error(f"Error deleting Milvus Lite database file: {e}")
                raise e


    def on_unmount(self) -> None:
        """Clean up memory when the app is closed."""
        try:
            self.reset_rag()
            logging.info("Memory cleaned up successfully.")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")


    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )


    def action_toggle_settings(self) -> None:
        """Keyboard shortcut: show/hide setup (data folder + build)."""
        if getattr(self, 'querying', False):
            return

        if isinstance(self.screen, SetupScreen):
            self.pop_screen()
        else:
            self.push_screen(SetupScreen())


if __name__ == "__main__":
    ChatApp().run()
