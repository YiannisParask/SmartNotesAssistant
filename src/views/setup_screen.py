from __future__ import annotations
import os
import asyncio
import logging
import traceback
from typing import TYPE_CHECKING
from rich.markup import escape
from textual.screen import Screen
from textual.widgets import Input, Static, Button, Checkbox
from textual.containers import Horizontal
from textual.app import ComposeResult

if TYPE_CHECKING:
    from main import ChatApp

class SetupScreen(Screen[None]):
    """Screen for database initialization and indexing."""

    DEFAULT_CSS = """
    SetupScreen {
        layout: vertical;
        align: center middle;
        padding: 2;
    }

    #setup {
        width: 100%;
        margin-bottom: 1;
    }

    #data_path {
        width: 100%;
        margin-bottom: 1;
    }

    #setup_buttons {
        height: auto;
        margin-bottom: 1;
    }

    #setup_buttons Button {
        margin-right: 2;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the setup screen widgets.

        Returns:
            ComposeResult: The composed widgets.
        """
        setup = Static(id="setup")
        setup.update(
            "[b]Setup Page[/b]\n"
            "Provide the path to your data folder. Click 'Build Index' to load and vectorize.\n\n"
        )
        yield setup

        # Pre-fill the data path input with a default value if it exists
        app = self.app  # type: ignore[assignment]
        last_path: str = app.get_last_data_path()
        yield Input(value=last_path, placeholder="Enter data folder path...", id="data_path")
        yield Checkbox("Clean build (recreate database file)", value= True, id="clean_build")

        # Buttons row (Build + Back)
        yield Horizontal(
            Button("Build Index", id="build_index"),
            Button("Back", id="back_setup"),
            id="setup_buttons",
        )
        yield Static("", id="setup_status")

    def on_mount(self) -> None:
        """Handle the mounting event for SetupScreen."""
        data_path: Input = self.query_one("#data_path", Input)
        self.set_focus(data_path)
        self._update_back_button_state()

    def _update_back_button_state(self) -> None:
       back_button: Button = self.query_one("#back_setup", Button)
       app: ChatApp = self.app
       back_button.disabled = not (os.path.exists(app.milvus_uri) or app.rag)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events.

        Args:
            event (Button.Pressed): The button press event.
        """
        if event.button.id == "build_index":
            await self._handle_build_index()
        elif event.button.id == "back_setup":
            self._attempt_back_from_setup()

    def _attempt_back_from_setup(self) -> None:
        """Attempt to navigate back from the setup screen."""
        status: Static = self.query_one("#setup_status", Static)
        app: ChatApp = self.app
        if os.path.exists(app.milvus_uri) or app.rag:
            self.app.pop_screen()
        else:
            status.update("[yellow]Cannot go back. Please build the index first.[/yellow]")


    async def _handle_build_index(self) -> None:
        """Handle building the vector index in a background thread."""
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

        # Disable controls during indexing
        data_path.disabled = True
        build_btn.disabled = True
        status.update(
            "[yellow]Building index (loading + vectorizing)... This may take a while.[/yellow]"
        )

        clean_build: bool = self.query_one("#clean_build", Checkbox).value

        app: ChatApp = self.app  # type: ignore[assignment]
        if clean_build:
            try:
                app.delete_db()
            except Exception as e:
                logging.error(traceback.format_exc())
                status.update(
                    f"[red]Failed to delete database:[/red] {escape(str(e))}"
                )
                return

        app.save_last_data_path(path)

        try:
            await asyncio.to_thread(app.build_index, path)
            status.update(
                "[green]Index built successfully! Initializing chat...[/green]"
            )
            self.app.pop_screen()
            await app.initialize_rag()
        except Exception as e:
            logging.error(traceback.format_exc())
            status.update(
                f"[red]Failed to build index:[/red] {escape(str(e))}"
            )
        finally:
            data_path.disabled = False
            build_btn.disabled = False
