"""
Call Amazon Bedrock Converse API for Claude (and other native Bedrock models).

Use this when the model is Claude (anthropic.*): the OpenAI-compatible endpoint
(Mantle / bedrock-runtime/openai/v1) only supports open-weight models (gpt-oss-*).
Converse is the native Bedrock API that supports Claude.

Requires: OPENAI_API_KEY (Bedrock API key) and BEDROCK_REGION (default us-west-2).
Optional: AWS_BEARER_TOKEN_BEDROCK (if set, used instead of OPENAI_API_KEY for boto3).
"""

import os
from typing import Any, Dict, List, Optional

# Lazy boto3 import so callers without boto3 only fail when using Converse
_client: Optional[Any] = None


def _get_region() -> str:
    region = os.environ.get("BEDROCK_REGION", "").strip()
    if region:
        return region
    base_url = os.environ.get("OPENAI_BASE_URL", "") or ""
    if "us-west-2" in base_url:
        return "us-west-2"
    if "us-east-1" in base_url:
        return "us-east-1"
    return "us-west-2"


def get_bedrock_converse_client():
    """Return a boto3 bedrock-runtime client. Uses OPENAI_API_KEY as bearer token when set."""
    global _client
    if _client is not None:
        return _client
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise ImportError("boto3 is required for Bedrock Converse (Claude). Install with: pip install boto3") from e

    region = _get_region()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    bearer = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
    if api_key and not bearer:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
    config = Config(connect_timeout=60, read_timeout=60)
    _client = boto3.client("bedrock-runtime", region_name=region, config=config)
    return _client


def _messages_to_converse(messages: List[Dict[str, Any]]) -> tuple:
    """Convert OpenAI-style messages to Converse format. Returns (converse_messages, system_text or None)."""
    system_parts = []
    converse_messages = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        if role == "system":
            content = m.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            else:
                system_parts.append(str(content))
            continue
        if role not in ("user", "assistant"):
            role = "user"
        content = m.get("content")
        if isinstance(content, str):
            content_blocks = [{"text": content}]
        else:
            content_blocks = [{"text": str(c) if not isinstance(c, dict) else c.get("text", str(c))} for c in (content or [])]
        converse_messages.append({"role": role, "content": content_blocks})
    system_text = "\n".join(system_parts).strip() if system_parts else None
    return converse_messages, system_text


def converse(
    model_id: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call Bedrock Converse API. Returns a dict shaped like an OpenAI completion for drop-in use:
      - "content": str (assistant text)
      - "usage": {"prompt_tokens": int, "completion_tokens": int}
      - "logprobs": None (Converse does not expose logprobs in same format; callers can use JSON fallback)
    """
    client = get_bedrock_converse_client()
    converse_messages, system_text = _messages_to_converse(messages)
    if not converse_messages:
        raise ValueError("converse requires at least one non-system message")

    request: Dict[str, Any] = {
        "modelId": model_id,
        "messages": converse_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system_text:
        request["system"] = [{"text": system_text}]

    response = client.converse(**request)
    output = response.get("output", {})
    msg = output.get("message", {})
    content_blocks = msg.get("content", [])
    content_parts = []
    for block in content_blocks:
        if "text" in block:
            content_parts.append(block["text"])
    content = "".join(content_parts)

    usage = response.get("usage", {})
    return {
        "content": content,
        "usage": {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
        },
        "logprobs": None,
    }


class ConverseCompletionAdapter:
    """Object that mimics an OpenAI completion so existing code can use completion.choices[0].message.content etc."""

    def __init__(self, result: Dict[str, Any]):
        self._result = result
        content = result.get("content", "")
        usage = result.get("usage", {})
        self.choices = [
            type(
                "Choice",
                (),
                {
                    "message": type("Message", (), {"content": content})(),
                    "logprobs": result.get("logprobs"),
                    "index": 0,
                },
            )()
        ]
        self.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )()
