"""LLM integration via Mellea-partials stream_with_chunking."""

import asyncio
import logging
import os

from mellea.backends.openai import OpenAIBackend
from mellea.backends.model_options import ModelOption
from mellea.stdlib.components.instruction import Instruction

from stream_with_chunking import ChunkingMode, stream_with_chunking

logger = logging.getLogger(__name__)

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "local-model")

# System prompt for the voice assistant
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Respond naturally and conversationally. "
    "Keep responses concise — typically 2-4 sentences. "
    "Avoid markdown, bullet points, or numbered lists; use plain prose only."
)


def _make_backend() -> OpenAIBackend:
    return OpenAIBackend(
        model_id=LM_STUDIO_MODEL,
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model_options={ModelOption.STREAM: True},
    )


async def generate_response(user_text: str, sentence_queue: asyncio.Queue[str | None]) -> None:
    """Stream LLM response sentence-by-sentence into sentence_queue.

    Puts each sentence string onto the queue, then puts None as sentinel.
    """
    logger.info("LLM input: %r", user_text)
    backend = _make_backend()

    instruction = Instruction(
        description=f"{user_text}",
    )

    result = await stream_with_chunking(
        instruction,
        backend,
        chunking_mode=ChunkingMode.SENTENCE,
    )

    async for sentence in result.astream():
        sentence = sentence.strip()
        if sentence:
            logger.info("LLM sentence: %r", sentence)
            await sentence_queue.put(sentence)

    logger.info("LLM complete. Full text: %r", result.full_text)
