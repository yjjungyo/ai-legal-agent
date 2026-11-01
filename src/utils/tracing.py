"""Langfuse callbacks for the existing LangGraph invocation."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from src.config import (
    LANGFUSE_ENABLED,
    LANGFUSE_CAPTURE_CONTENT,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_SYSTEM_NAME,
)

logger = logging.getLogger("ai-legal-agent")


def _mask_content(*, data):
    return "[REDACTED]"


@lru_cache(maxsize=1)
def _get_langfuse_client():
    if not LANGFUSE_ENABLED:
        return None
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        logger.warning("[Langfuse] Tracing disabled: public/secret key is missing.")
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            mask=None if LANGFUSE_CAPTURE_CONTENT else _mask_content,
        )
    except Exception as exc:
        logger.warning("[Langfuse] Client initialization failed (%s).", type(exc).__name__)
        return None


@contextmanager
def langfuse_config(thread_id: str) -> Iterator[dict]:
    """Keep checkpoint config intact and use a separate callback for each request."""
    config = {"configurable": {"thread_id": thread_id}}
    client = _get_langfuse_client()
    if client is not None:
        try:
            from langfuse.callback import CallbackHandler

            tags = [LANGFUSE_SYSTEM_NAME]
            handler = CallbackHandler(
                stateful_client=client.trace(
                    name=LANGFUSE_SYSTEM_NAME,
                    session_id=thread_id,
                    tags=tags,
                ),
                update_stateful_client=True,
                trace_name=LANGFUSE_SYSTEM_NAME,
                session_id=thread_id,
                tags=tags,
            )
            # Langfuse Generations 목록에 모델 클래스명(AzureChatOpenAI) 대신 LegalAgent 표기 보장
            _orig_run_name = handler.get_langchain_run_name
            def _resolved_run_name(serialized=None, **kwargs):
                run_name = _orig_run_name(serialized, **kwargs)
                if run_name in ("AzureChatOpenAI", "ChatOpenAI", "<unknown>"):
                    return LANGFUSE_SYSTEM_NAME
                return run_name
            handler.get_langchain_run_name = _resolved_run_name
            config.update(callbacks=[handler], run_name=LANGFUSE_SYSTEM_NAME, tags=tags)
        except Exception as exc:
            logger.warning("[Langfuse] Callback initialization failed (%s).", type(exc).__name__)
    try:
        yield config
    finally:
        if client is not None:
            try:
                client.flush()
            except Exception as exc:
                logger.warning("[Langfuse] Trace flush failed (%s).", type(exc).__name__)
