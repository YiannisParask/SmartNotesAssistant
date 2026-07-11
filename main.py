import os
from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from textual.app import App, ComposeResult
from textual.widgets import Input, LoadingIndicator, Static, Placeholder, Footer, Button
from textual.reactive import reactive
from textual.binding import Binding
from src.perform_rag_search import RagSearch
from src.load_and_vectorize_data import LoadAndVectorizeData
from textual.containers import VerticalScroll, Horizontal
import asyncio
from rich.markup import escape
import traceback
import gc
import logging


# Constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
COLLECTION_NAME = "MilvusDocs"
CWD: str = os.getcwd()
MILVUS_URI: str = os.path.join(CWD, "data", "local_milvus_database.db")
DEVICE: Any = "cuda" if torch.cuda.is_available() else "cpu"


class Header(Placeholder):
    DEFAULT_CSS = """
    Header {
        height: 3;
        dock: top;
    }
    """


class Message(Static):
    def __init__(self, user: str, message: str, **kwargs):
        style = "bold blue" if user == "You" else "magenta"
        super().__init__(f"[{style}]{user}:[/] {message}", **kwargs)


class ChatApp(App):
    CSS = """
    Screen {
        layout: vertical;
        align: center middle;
    }

    #chat_container {
        height: 1fr;
        width: 100%;
        border: round blue;
        padding: 1;
    }

    #input {
        width: 100%;
        min-height: 3;
    }

    .hidden {
        display: none;
    }
    """

    messages: reactive[list[str]] = reactive([])
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit the app"),
        Binding(key="d", action="toggle_dark", description="Toggle dark mode"),
        Binding(key="s", action="toggle_settings", description="Settings")
    ]


    def compose(self) -> ComposeResult:
        """ `Textual` method that renders all UI elements.
        Defines the applications UI layout and components.
        """
        yield Header("Smart Notes Assistant")

        # Initialization loader
        yield LoadingIndicator(id="init_loader", classes="hidden")

        # Setup panel (shown only if DB/collection missing)
        setup = Static(id="setup")
        setup.update(
            "[b]Setup Page[/b]\n"
            "Provide the path to your data folder. Click 'Build Index' to load and vectorize.\n\n"
        )
        yield setup

        # Controls inside setup
        yield Input(placeholder="Enter your data folder path...", id="data_path")
        # Buttons row (Build + Back)
        yield Horizontal(
            Button("Build Index", id="build_index"),
            Button("Back", id="back_setup"),
            id="setup_buttons",
        )
        yield Static("", id="setup_status")

        # Chat UI (shown after setup is complete)
        yield VerticalScroll(id="chat_container")
        yield Input(placeholder="Type your message and press Enter...", id="input")

        # Show Footer
        yield Footer()


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
        self._toggle_setup(needs_setup)

        # If the DB file exists, we can initialize the RAG system immediately.
        if not needs_setup:
            await self.initialize_rag()


    def _toggle_setup(self, show_setup: bool) -> None:
        """
        Toggle the visibility of the setup page.
        """
        # Setup widgets
        setup: Static = self.query_one("#setup", Static)
        data_path: Input = self.query_one("#data_path", Input)
        setup_buttons: Horizontal = self.query_one("#setup_buttons", Horizontal)
        setup_status: Static = self.query_one("#setup_status", Static)
        chat: VerticalScroll = self.query_one("#chat_container", VerticalScroll)
        chat_input: Input = self.query_one("#input", Input)

        init_loader: LoadingIndicator = self.query_one("#init_loader", LoadingIndicator)

        for w in (setup, data_path, setup_buttons, setup_status):
            w.display = show_setup
        chat.display = not show_setup
        chat_input.display = not show_setup

        if show_setup:
            init_loader.display = False
            self.set_focus(data_path)
        else:
            self.set_focus(chat_input)

        # Enable Back only if an index already exists or RAG loaded
        back_button: Button = self.query_one("#back_setup", Button)
        back_button.disabled = not (os.path.exists(MILVUS_URI) or self.rag)


    async def initialize_rag(self) -> None:
        """ Initialize the RAG pipeline components. This should only be called
        once after the index is built.
        """
        # Initialize RAG pipeline once!
        chat = self.query_one("#chat_container", VerticalScroll)
        init_loader: LoadingIndicator = self.query_one("#init_loader", LoadingIndicator)
        chat_input: Input = self.query_one("#input", Input)

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
            chat.mount(Message("System", "Welcome! Type your question and press Enter."))
            logging.info("RAG system initialized successfully!")

        except Exception as e:
            logging.error(f"Error initializing RAG system: {e}")
            self.rag = None
            self.retriever = None
            self.querying = True  # Disable querying if initialization failed
            chat.mount(Message("System", f"Error: Failed to initialize RAG system. {e}"))

        finally:
            init_loader.display = False
            chat.display = True
            chat_input.display = True
            self.set_focus(chat_input)


    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "build_index":
            await self._handle_build_index()
        elif event.button.id == "back_setup":
            self._attempt_back_from_setup()


    def _attempt_back_from_setup(self) -> None:
        """Attempt to go back from setup to chat."""
        status: Static = self.query_one("#setup_status", Static)
        if os.path.exists(MILVUS_URI) or self.rag:
            self._toggle_setup(False)
        else:
            status.update("[yellow]Cannot go back: index not built yet.[/yellow]")


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


    async def _handle_build_index(self) -> None:
        """ Handle the build index button press event;
        Disables UI controls & shows status message.
        """
        data_path: Input = self.query_one("#data_path", Input)
        status: Static = self.query_one("#setup_status", Static)
        build_btn: Button = self.query_one("#build_index", Button)

        path: str = (data_path.value or "").strip()
        if not path:
            status.update("[red]Please provide a data folder path.[/red]")
            return
        if not os.path.isdir(path):
            status.update(f"[red]Not a directory:[/red] {path}")
            return

        # Disable inputs during build
        data_path.disabled = True
        build_btn.disabled = True
        status.update(
            "[yellow]Building index (loading + vectorizing)... This may take a while.[/yellow]"
        )

        try:
            await asyncio.to_thread(self.build_index, path)
            status.update(
                "[green]Index built successfully! Initializing chat...[/green]"
            )
            self._toggle_setup(False)
            await self.initialize_rag()
        except Exception as e:
            logging.error(traceback.format_exc())
            status.update(
                f"[red]Failed to build index:[/red] {escape(str(e))}"
            )
        finally:
            data_path.disabled = False
            build_btn.disabled = False


    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_message: str = event.value.strip()
        if not user_message or self.querying:
            return
        self.querying = True

        # Disable input while processing
        input_widget: Input = self.query_one("#input", Input)
        input_widget.disabled = True
        input_widget.placeholder = "Processing... Please wait"

        chat: VerticalScroll = self.query_one("#chat_container", VerticalScroll)
        chat.mount(Message("You", user_message))
        event.input.value = ""
        await self.perform_llm_query(user_message, chat)


    async def perform_llm_query(self, prompt, chat) -> None:
        # Check if RAG system is properly initialized
        if self.rag is None or self.retriever is None or self.rag.agent is None:
            chat.mount(Message("System", "Error: RAG system not initialized. Please restart the application."))
            self.querying = False
            self.re_enable_input()
            return

        torch.cuda.empty_cache()
        try:
            answer: str = await self.rag.perform_rag_query(self.retriever, prompt)
            chat.mount(Message("LLM", answer))
        except Exception as e:
            chat.mount(Message("System", f"Error processing query: {escape(str(e))}"))

        self.querying = False
        self.re_enable_input()

        def scroll_to_end() -> None:
            chat.scroll_y = chat.virtual_size.height

        self.call_after_refresh(scroll_to_end)


    def re_enable_input(self) -> None:
        """Re-enable the input field after processing."""
        input_widget: Input = self.query_one("#input", Input)
        input_widget.disabled = False
        input_widget.placeholder = "Type your message and press Enter..."
        input_widget.value = ""


    def on_unmount(self) -> None:
        """Clean up memory when the app is closed."""
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
            gc.collect()

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
        setup: Static = self.query_one("#setup", Static)
        self._toggle_setup(not setup.display)


if __name__ == "__main__":
    ChatApp().run()
