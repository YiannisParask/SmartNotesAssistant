from textual.widgets import Static

class ChatMessage(Static):

    def __init__(self, message: str, user: str, **kwargs) -> None:
        """Initialize the ChatMessage widget.

        Args:
            message (str): The chat message content.
            user (str): The user who sent the message.
            **kwargs: Additional keyword arguments for the Static widget.
        """
        style: str = "bold blue" if user == "You" else "magenta"
        super().__init__(f"[{style}]{user}:[/] {message}", **kwargs)