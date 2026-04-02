"""LLM integration via Mellea-partials stream_with_chunking."""

import asyncio
import logging
import os
import re

from mellea.backends.openai import OpenAIBackend
from mellea.backends.model_options import ModelOption
from mellea.stdlib.components.instruction import Instruction
from mellea.core.requirement import Requirement, ValidationResult
from mellea.stdlib.context import SimpleContext

from mellea_partial import ChunkingMode, StreamChunkingResult, stream_with_chunking

logger = logging.getLogger(__name__)

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "granite-4.0-micro@q8_0")
GUARDIAN_MODEL = os.environ.get("GUARDIAN_MODEL", "granite-guardian-3.3-8b")

# System prompt for the voice assistant
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Respond naturally and conversationally. "
    "Keep responses concise — typically 2-4 sentences. "
    "Avoid markdown, bullet points, or numbered lists; use plain prose only."
)


class MarkdownFreeRequirement(Requirement):
    """Rejects chunks that contain markdown formatting (bad for TTS)."""

    def __init__(self):
        super().__init__(
            description="The response must not contain markdown formatting.",
            check_only=True,
        )

    async def validate(self, backend, ctx, *, format=None, model_options=None):
        text = ctx.last_output().value or ""
        has_markdown = bool(re.search(r"(\*\*|\*|#+|`|\[.+\]\(.+\)|^\s*[-*]\s)", text, re.MULTILINE))
        return ValidationResult(result=not has_markdown)


class GuardianRequirement(Requirement):
    """Quick-check requirement that calls Granite Guardian to detect harmful content."""

    def __init__(self):
        super().__init__(
            description="The response must not contain harmful, toxic, or unsafe content.",
            check_only=True,
            output_to_bool=(lambda x: "yes" not in str(x).lower())
        )
        self._guardian_backend = OpenAIBackend(
            model_id=GUARDIAN_MODEL,
            base_url=LM_STUDIO_URL,
            api_key="lm-studio",
        )

    async def validate(self, backend, ctx, *, format=None, model_options=None):
        last_output = ctx.last_output()
        chunk_text = last_output.value
        if not chunk_text or not chunk_text.strip():
            return ValidationResult(result=True)
        try:
            test_ctx = SimpleContext()
            test_ctx = test_ctx.add(last_output)
            return await super().validate(
                self._guardian_backend, test_ctx, format=format, model_options=model_options
            )
        except Exception:
            logger.warning("Guardian check failed, allowing chunk: %r", chunk_text, exc_info=True)
            return ValidationResult(result=True)


def _make_backend() -> OpenAIBackend:
    return OpenAIBackend(
        model_id=LM_STUDIO_MODEL,
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model_options={ModelOption.STREAM: True,
                       SYSTEM_PROMPT: "You are a helpful chat assistant. Keep answers short."},
    )


async def generate_response(user_text: str, sentence_queue: asyncio.Queue[str | None]) -> StreamChunkingResult:
    """Stream LLM response sentence-by-sentence into sentence_queue.

    Puts each sentence string onto the queue, then puts None as sentinel.
    Returns the StreamChunkingResult so callers can cancel _task if needed.
    """
    logger.debug("LLM input: %r", user_text)
    backend = _make_backend()

    instruction = Instruction(
        description=f"{user_text}",
    )

    async def on_failure(chunk: str, ctx, requirements, results) -> tuple[bool, str]:
        return True, "Sorry, some things are not right."

    ctx = SimpleContext()
    result = await stream_with_chunking(
        instruction,
        backend,
        ctx,
        chunking=ChunkingMode.SENTENCE,
        quick_check_requirements=[GuardianRequirement(), MarkdownFreeRequirement()],
        quick_repair=on_failure,
    )

    async for sentence in result.astream():
        sentence = sentence.strip()
        if sentence:
            logger.debug("LLM sentence: %r", sentence)
            await sentence_queue.put(sentence)

    if result.failed_chunk:
        logger.warning("Guardian stopped streaming at chunk: %r", result.failed_chunk)

    logger.debug("LLM complete. Full text: %r", result.full_text)
    return result
