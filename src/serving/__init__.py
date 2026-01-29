"""
Serving Module - OpenAI-compatible LLM Server.

Provides:
- FastAPI Server
- Integration with ContinuousBatching
"""

from src.serving.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from src.serving.server import create_server

__all__ = [
    "create_server",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
]
