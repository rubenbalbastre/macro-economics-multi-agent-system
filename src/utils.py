from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json

console = Console()

def format_message_content(message):
    """Convert message content to displayable string"""
    parts = []
    tool_calls_processed = False
    
    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'tool_use':
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))
    
    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if not tool_calls_processed and hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2)}")
            parts.append(f"   ID: {tool_call['id']}")
    
    return "\n".join(parts)


def format_messages(messages):
    """Format and display a list of messages with Rich formatting"""
    for m in messages:
        msg_type = m.__class__.__name__.replace('Message', '')
        content = format_message_content(m)

        if msg_type == 'Human':
            console.print(Panel(content, title="🧑 Human", border_style="blue"))
        elif msg_type == 'Ai':
            console.print(Panel(content, title="🤖 Assistant", border_style="green"))
        elif msg_type == 'Tool':
            console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))


def format_message(messages):
    """Alias for format_messages for backward compatibility"""
    return format_messages(messages)


def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "blue"):
    """
    Display a prompt with rich formatting and XML tag highlighting.
    
    Args:
        prompt_text: The prompt string to display
        title: Title for the panel (default: "Prompt")
        border_style: Border color style (default: "blue")
    """
    # Create a formatted display of the prompt
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r'<[^>]+>', style="bold blue")  # Highlight XML tags
    formatted_text.highlight_regex(r'##[^#\n]+', style="bold magenta")  # Highlight headers
    formatted_text.highlight_regex(r'###[^#\n]+', style="bold cyan")  # Highlight sub-headers

    # Display in a panel for better presentation
    console.print(Panel(
        formatted_text, 
        title=f"[bold green]{title}[/bold green]",
        border_style=border_style,
        padding=(1, 2)
    ))


from collections.abc import Sequence
from langchain_core.messages import BaseMessage, HumanMessage, ChatMessage, FunctionMessage, AIMessage, ToolMessage, SystemMessage


def get_buffer_string(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    r"""Convert a sequence of messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of `HumanMessage`s.
        ai_prefix: The prefix to prepend to contents of `AIMessage`.

    Returns:
        A single string concatenation of all input messages.

    Raises:
        ValueError: If an unsupported message type is encountered.

    Example:
        ```python
        from langchain_core import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="Hi, how are you?"),
            AIMessage(content="Good, how are you?"),
        ]
        get_buffer_string(messages)
        # -> "Human: Hi, how are you?\nAI: Good, how are you?"
        ```
    """
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ToolMessage):
            role = "Tool"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            msg = f"Got unsupported message type: {m}"
            raise ValueError(msg)  # noqa: TRY004
        message = f"{role}: {m.text}"
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls"):
            message += f"{m.tool_calls}"
        elif isinstance(m, AIMessage) and "tool_calls" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return "\n".join(string_messages)
