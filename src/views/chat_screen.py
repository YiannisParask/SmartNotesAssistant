from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import os
from textual.screen import Screen
from textual.widgets import Input, LoadingIndicator, Footer
from textual.app import ComposeResult
from src.widgets.app_header import AppHeader
from src.widgets.chat_message import ChatMessage
from textual.containers import VerticalScroll

if TYPE_CHECKING:
    from main import ChatApp

class ChatScreen(Screen[None]):
    """Screen containing the main conversation interface."""

    DEFAULT_CSS = """
    ChatScreen {
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
    """

    def compose(self) -> ComposeResult:
        """Compose the chat screen widgets.

        Returns:
            ComposeResult: The composed widgets.
        """
        yield AppHeader("Smart Notes Assistant")
        yield LoadingIndicator(id="init_loader", classes="hidden")
        yield VerticalScroll(id="chat_container")
        yield Input(placeholder="Type your message and press Enter...", id="input")
        yield Footer()

    def on_mount(self) -> None:
        """Handle the mounting event for ChatScreen."""
        chat_input: Input = self.query_one("#input", Input)
        self.set_focus(chat_input)

        # If RAG is not initialized yet and the database exists, initialize it
        app: ChatApp = self.app  # type: ignore[assignment]
        if app.rag is None and os.path.exists(app.milvus_uri):
            app.run_worker(app.initialize_rag())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submissions.

        Args:
            event (Input.Submitted): The input submission event.
        """
        app: ChatApp = self.app  # type: ignore[assignment]
        user_message: str = event.value.strip()
        if not user_message or app.querying:
            return
        app.querying = True

        # Disable input while processing
        input_widget: Input = self.query_one("#input", Input)
        input_widget.disabled = True
        input_widget.placeholder = "Processing... Please wait"

        chat: VerticalScroll = self.query_one("#chat_container", VerticalScroll)
        chat.mount(ChatMessage("You", user_message))
        event.input.value = ""
        await self.perform_llm_query(user_message, chat)

    async def perform_llm_query(self, prompt: str, chat: VerticalScroll) -> None:
        """Query the LLM agent via the app coordinator and display results.

        Args:
            prompt (str): The prompt message to send to the agent.
            chat (VerticalScroll): The chat display container.
        """
        app: ChatApp = self.app  # type: ignore[assignment]
        if app.rag is None or app.retriever is None or app.rag.agent is None:
            chat.mount(ChatMessage("System", "Error: RAG system not initialized. Please restart the application."))
            app.querying = False
            self.re_enable_input()
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            answer: str = await app.rag.perform_rag_query(app.retriever, prompt)
            chat.mount(ChatMessage("LLM", answer))
        except Exception as e:
            chat.mount(ChatMessage("System", f"Error processing query: {str(e)}"))

        app.querying = False
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