"""LLM Gateway using litellm for multi-provider support."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import litellm
import structlog

logger = structlog.get_logger()

# Pattern for removing <think>...</think> tags from LLM responses (extended thinking mode)
THINK_TAG_PATTERN = re.compile(r'<think>.*?</think>\s*', re.DOTALL | re.IGNORECASE)

# Drop unsupported params to avoid "unexpected keyword argument" errors
# This is needed for OpenRouter and other providers that don't support all params
# See: https://github.com/BerriAI/litellm/issues/14137
litellm.drop_params = True

from dumont_core.llm.providers.factory import create_provider
from dumont_core.resilience import CircuitBreaker, RetryPolicy
from lca.core.cache.semantic_cache import SemanticCache


# Compression statistics for monitoring
@dataclass
class CompressionStats:
    """Statistics for prompt compression."""

    total_requests: int = 0
    compressed_requests: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    total_savings_tokens: int = 0
    total_compression_time: float = 0.0

    @property
    def compression_ratio(self) -> float:
        """Average compression ratio."""
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def savings_percent(self) -> float:
        """Average token savings percentage."""
        return (1 - self.compression_ratio) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "compressed_requests": self.compressed_requests,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "total_savings_tokens": self.total_savings_tokens,
            "compression_ratio": self.compression_ratio,
            "savings_percent": self.savings_percent,
            "total_compression_time": self.total_compression_time,
        }


# Global compression stats
_compression_stats = CompressionStats()

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@dataclass
class ToolCall:
    """Represents a function call from LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Response from LLM chat."""

    content: str | None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str = "stop"

    def __post_init__(self):
        """Clean up <think>...</think> tags from content (extended thinking mode artifacts)."""
        if self.content:
            self.content = THINK_TAG_PATTERN.sub('', self.content).strip()


class LLMGateway:
    """
    Gateway for LLM providers via litellm.

    Implements LLMProtocol.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        provider: str = "openai",
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        fallback_models: list[str] | None = None,
    ):
        """
        Initialize LLM Gateway.

        Args:
            model: Model name
            api_key: API key for the provider
            provider: Provider name (default: "openai")
            retry_policy: Optional retry policy for resilient calls
            circuit_breaker: Optional circuit breaker for failure protection
            fallback_models: List of fallback models to try if primary fails
        """
        self.provider = provider
        self.api_key = api_key  # Store API key for explicit passing to LiteLLM
        self.fallback_models = fallback_models or []

        # Create provider strategy
        try:
            self.provider_strategy = create_provider(provider)
        except ValueError:
            # Fallback: try to detect provider from model name
            if model.startswith(("openrouter/", "qwen/")):
                self.provider_strategy = create_provider("openrouter")
                self.provider = "openrouter"
            else:
                # Default to OpenAI
                self.provider_strategy = create_provider("openai")
                self.provider = "openai"

        # Configure provider
        self.provider_strategy.configure(model, api_key)

        # Get formatted model name
        self.model = self.provider_strategy.get_model_name(model)

        # Check function calling support
        self._supports_function_calling = self.provider_strategy.supports_function_calling(
            self.model
        )

        # Resilience patterns
        self.retry_policy = retry_policy
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        # Initialize cache
        self.cache = SemanticCache()

        # Initialize compression (lazy loaded)
        self._compression_enabled = os.environ.get("LLMLINGUA_ENABLED", "").lower() == "true"
        self._compression_use_remote = os.environ.get("LLMLINGUA_USE_REMOTE", "").lower() == "true"

        # Safe type casting from environment variables
        try:
            self._compression_target_ratio = float(os.environ.get("LLMLINGUA_TARGET_RATIO", "0.5"))
        except (ValueError, TypeError):
            logger.warning("Invalid LLMLINGUA_TARGET_RATIO, using default 0.5")
            self._compression_target_ratio = 0.5

        try:
            self._compression_min_size = int(os.environ.get("LLMLINGUA_MIN_PROMPT_SIZE", "500"))
        except (ValueError, TypeError):
            logger.warning("Invalid LLMLINGUA_MIN_PROMPT_SIZE, using default 500")
            self._compression_min_size = 500
        self._compressor = None
        self._remote_compressor = None

        logger.info(
            "LLMGateway initialized",
            model=self.model,
            compression_enabled=self._compression_enabled,
            compression_remote=self._compression_use_remote,
        )

    def _extract_valid_json(self, json_str: str) -> dict:
        """
        Extract valid JSON from potentially malformed string.
        
        Handles cases where JSON has extra text before/after or is incomplete.
        """
        import re
        
        # Try to find JSON object
        if json_str.strip().startswith('{'):
            # Find matching closing brace (handles nested objects)
            brace_count = 0
            end_pos = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(json_str):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > 0:
                try:
                    return json.loads(json_str[:end_pos])
                except json.JSONDecodeError:
                    pass
        
        # Fallback: try to extract from markdown code block
        code_block_match = re.search(r"```(?:json)?\n?([\s\S]*?)\n?```", json_str)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Final fallback: return empty dict
        return {}

    @property
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling."""
        return self._supports_function_calling

    def _format_model_for_openrouter(self, model: str) -> str:
        """Format model name for OpenRouter."""
        if model.startswith("openrouter/"):
            return model
        if "/" in model:
            return f"openrouter/{model}"
        return f"openrouter/{model}"

    async def _compress_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Compress messages using LLMLingua-2.

        Only compresses the content of messages that exceed min_prompt_size.
        System and short messages are preserved as-is.
        """
        global _compression_stats

        if not self._compression_enabled:
            return messages

        _compression_stats.total_requests += 1

        compressed_messages = []
        total_original = 0
        total_compressed = 0

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # NEVER compress system messages - they contain critical instructions
            if role == "system":
                compressed_messages.append(msg)
                continue

            # Skip compression for small messages
            if len(content) < self._compression_min_size:
                compressed_messages.append(msg)
                continue

            try:
                if self._compression_use_remote:
                    # Use Vast.ai remote compression
                    compressed_content = await self._compress_remote(content)
                else:
                    # Use local compression
                    compressed_content = await self._compress_local(content)

                original_tokens = len(content.split())
                compressed_tokens = len(compressed_content.split())
                total_original += original_tokens
                total_compressed += compressed_tokens

                logger.debug(
                    "Compressed message",
                    role=role,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    ratio=f"{compressed_tokens/original_tokens*100:.1f}%",
                )

                compressed_messages.append({
                    **msg,
                    "content": compressed_content,
                })

            except Exception as e:
                logger.warning(f"Compression failed, using original: {e}")
                compressed_messages.append(msg)

        # Update stats
        if total_original > 0:
            _compression_stats.compressed_requests += 1
            _compression_stats.original_tokens += total_original
            _compression_stats.compressed_tokens += total_compressed
            _compression_stats.total_savings_tokens += total_original - total_compressed

            logger.info(
                "Compression complete",
                original_tokens=total_original,
                compressed_tokens=total_compressed,
                savings=f"{(1-total_compressed/total_original)*100:.1f}%",
            )

        return compressed_messages

    async def _compress_local(self, text: str) -> str:
        """Compress text using local LLMLingua-2."""
        if self._compressor is None:
            from lca.compression import get_compressor, CompressionConfig

            config = CompressionConfig(
                enabled=True,
                target_ratio=self._compression_target_ratio,
                min_prompt_size=self._compression_min_size,
            )
            self._compressor = get_compressor(config)

        import asyncio
        result = await asyncio.to_thread(
            self._compressor.compress,
            text,
            self._compression_target_ratio,
        )
        return result.compressed_text

    async def _compress_remote(self, text: str) -> str:
        """Compress text using remote Vast.ai LLMLingua-2 service."""
        if self._remote_compressor is None:
            from lca.compression.remote import get_vastai_client

            self._remote_compressor = get_vastai_client()

        result = await self._remote_compressor.compress(
            text,
            target_ratio=self._compression_target_ratio,
        )
        return result.compressed_text

    def get_compression_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        global _compression_stats
        return _compression_stats.to_dict()

    async def _execute_chat_with_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        """Execute chat with a specific model."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # Ensure API key is passed correctly for OpenRouter
        if self.provider == "openrouter":
            kwargs["api_key"] = self.api_key
            kwargs["model"] = self._format_model_for_openrouter(model)

        # Add function calling if supported and tools provided
        if tools and self._supports_function_calling:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = await litellm.acompletion(**kwargs)

        # Validate response has choices before accessing
        if not response.choices or len(response.choices) == 0:
            raise ValueError("LLM response has no choices")

        message = response.choices[0].message

        # Parse tool calls if present
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                # Handle different response formats
                if hasattr(tc, "function"):
                    func = tc.function
                    args_str = (
                        func.arguments
                        if isinstance(func.arguments, str)
                        else json.dumps(func.arguments)
                    )
                    # Parse JSON with error handling
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = self._extract_valid_json(args_str)
                    tool_calls.append(
                        ToolCall(
                            id=getattr(tc, "id", ""),
                            name=func.name,
                            arguments=args,
                        )
                    )
                elif isinstance(tc, dict):
                    # Handle dict format
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    if isinstance(args_str, dict):
                        args = args_str
                    else:
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            args = self._extract_valid_json(args_str)
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            arguments=args,
                        )
                    )

        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=getattr(response.choices[0], "finish_reason", "stop"),
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.7, # Added for caching key
        max_tokens: int | None = None, # Added for caching key
        **kwargs: Any # Added for caching key
    ) -> ChatResponse:
        """Send chat messages and get response with optional function calling."""
        # Apply compression if enabled (before cache check)
        if self._compression_enabled:
            messages = await self._compress_messages(messages)

        # Check cache
        cache_key_kwargs = {"messages": messages, "temperature": temperature, "tools": tools, "tool_choice": tool_choice, "max_tokens": max_tokens}
        cached_response_str = self.cache.get(str(messages), self.model, **cache_key_kwargs)
        if cached_response_str:
            logger.info("Returning cached response for chat", model=self.model)
            try:
                # Deserialize
                data = json.loads(cached_response_str)
                # Reconstruct ChatResponse (simplified)
                tool_calls = None
                if data.get("tool_calls"):
                    tool_calls = [ToolCall(**tc) for tc in data["tool_calls"]]
                
                return ChatResponse(
                    content=data.get("content"),
                    tool_calls=tool_calls,
                    finish_reason=data.get("finish_reason", "stop")
                )
            except Exception as e:
                logger.warning("Failed to deserialize cached response", error=str(e))
                # Fallthrough to normal execution

        # Build list of models to try: primary + fallbacks
        models_to_try = [self.model] + self.fallback_models
        last_error: Exception | None = None

        for current_model in models_to_try:
            try:
                logger.info(f"Trying model: {current_model}")
                # Try this model directly (simpler approach, no closure issues)
                result = await self._execute_chat_with_model(current_model, messages, tools, tool_choice)
                
                # Cache response if successful
                try:
                    # Serialize
                    response_data = {
                        "content": result.content,
                        "finish_reason": result.finish_reason,
                        "tool_calls": [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in (result.tool_calls or [])
                        ] if result.tool_calls else None
                    }
                    self.cache.set(str(messages), self.model, json.dumps(response_data), **cache_key_kwargs)
                except Exception as e:
                    logger.warning("Failed to cache response", error=str(e))
                
                return result
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                logger.warning(f"Model {current_model} failed: {e}")
                # Check if it's a rate limit or temporary error - try fallback
                if "429" in str(e) or "rate" in error_str or "limit" in error_str or "temporarily" in error_str:
                    logger.info(f"Rate limited, trying next fallback model...")
                    continue
                # For other errors, also try fallback
                logger.info(f"Trying next fallback model...")
                continue

        # All models failed
        raise RuntimeError(f"LLM request failed (all models exhausted): {last_error}") from last_error

    # LEGACY: Manter para compatibilidade
    async def chat_text(self, messages: list[dict[str, str]]) -> str:
        """Legacy text-only chat (backward compatible)."""
        response = await self.chat(messages)
        return response.content or ""

    async def stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        async def _execute_stream() -> AsyncGenerator[str, None]:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                stream=True,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        # Apply circuit breaker (note: circuit breaker doesn't work well with generators)
        # For streaming, we'll just wrap the initial call
        try:
            async for chunk in _execute_stream():
                yield chunk
        except Exception as e:
            # Check circuit breaker state
            if self.circuit_breaker.state.value == "open":
                raise RuntimeError("Circuit breaker is OPEN") from e
            raise RuntimeError(f"LLM stream failed: {e}") from e
