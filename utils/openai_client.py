"""
LLM calls go through AWS Bedrock (not direct OpenAI).

This module provides client factories that use OPENAI_API_KEY and OPENAI_BASE_URL
so every stage talks to your Bedrock endpoint. The OpenAI SDK is used only as the
client; all requests are sent to the base_url (Bedrock Mantle), not to api.openai.com.

Required for Bedrock:
    export OPENAI_API_KEY="bedrock-api-key-..."
    export OPENAI_BASE_URL="https://bedrock-mantle.us-west-2.api.aws/v1"

Then all create_openai_client() / create_async_openai_client() calls use Bedrock.
Use Bedrock model IDs (e.g. openai.gpt-oss-120b) in config.yaml / model params.

Example:
    from utils.openai_client import create_openai_client
    client = create_openai_client()
    # Chat Completions (Bedrock Mantle supports this)
    r = client.chat.completions.create(model="openai.gpt-oss-120b", messages=[...])
    # Or Responses API
    r = client.responses.create(model="openai.gpt-oss-120b", input=[{"role": "user", "content": "..."}])
    print(r.output_text)
"""

import os
from typing import Any, Dict, Optional

from openai import AsyncOpenAI, OpenAI


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    openai_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> OpenAI:
    """
    Create a client that sends LLM requests to AWS Bedrock (or the configured endpoint).

    With OPENAI_BASE_URL set to your Bedrock endpoint, all requests go to Bedrock,
    not to OpenAI. Uses OPENAI_API_KEY and OPENAI_BASE_URL from environment when
    not provided; openai_config can supply api_key/base_url (env overrides config).
    """
    if api_key is None and openai_config:
        api_key = openai_config.get("api_key")
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if base_url is None and openai_config:
        base_url = openai_config.get("base_url")
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL")
    build_kwargs: Dict[str, Any] = {}
    if api_key is not None:
        build_kwargs["api_key"] = api_key
    if base_url is not None:
        build_kwargs["base_url"] = base_url.rstrip("/")
    build_kwargs.update(kwargs)
    return OpenAI(**build_kwargs)


def create_async_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    openai_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client with same env/config behavior as create_openai_client."""
    if api_key is None and openai_config:
        api_key = openai_config.get("api_key")
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if base_url is None and openai_config:
        base_url = openai_config.get("base_url")
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL")
    build_kwargs: Dict[str, Any] = {}
    if api_key is not None:
        build_kwargs["api_key"] = api_key
    if base_url is not None:
        build_kwargs["base_url"] = base_url.rstrip("/")
    build_kwargs.update(kwargs)
    return AsyncOpenAI(**build_kwargs)


def get_chat_completions_url(base_url: Optional[str] = None) -> str:
    """
    Return the full URL for Chat Completions (e.g. for raw HTTP).
    Uses OPENAI_BASE_URL from env when base_url is None so requests go to Bedrock,
    not to OpenAI. Set OPENAI_BASE_URL for Bedrock (e.g. https://bedrock-mantle.us-west-2.api.aws/v1).
    """
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return base_url.rstrip("/") + "/chat/completions"
    # Fallback only when OPENAI_BASE_URL not set (e.g. local dev against OpenAI)
    return "https://api.openai.com/v1/chat/completions"
