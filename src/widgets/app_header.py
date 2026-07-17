from textual.widgets import Placeholder

class AppHeader(Placeholder):
    """Custom header widget for the application."""

    DEFAULT_CSS = """
    AppHeader {
        height: 3;
        dock: top;
    }
    """