# src/logging_utils.py
# -*- coding: utf-8 -*-
"""
logging_utils.py
- 공통 로깅 초기화와 함수 트레이싱 데코레이터 제공
- 요구사항: LOG_LEVEL 다음에 '메소드명:라인번호'를 항상 출력
  예) 2025-09-12 18:00:01,123 DEBUG route_categories:40 [ai-legal-agent] 메시지
"""

from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
import functools
from typing import Any, Callable, TypeVar, cast

from src.config import LOG_PATH

T = TypeVar("T")

LOGGER_NAME = "ai-legal-agent"

# ✅ 핵심: LOG_LEVEL 다음에 %(funcName)s:%(lineno)d 를 배치
# _LOG_FMT = "%(asctime)s,%(msecs)03d %(levelname)s %(funcName)s:%(lineno)d [%(name)s] %(message)s"
_LOG_FMT = "%(asctime)s,%(msecs)03d %(levelname)s [ %(filename)-20.20s %(funcName)-20.20s :%(lineno)4d ] %(message)s"

_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _desired_level() -> int:
    """환경변수 LOGLEVEL 로 레벨 제어 (기본 INFO)."""
    level = os.getenv("LOGLEVEL", "DEBUG").upper()
    return getattr(logging, level, logging.DEBUG)


def setup_logging(level: int | None = None) -> logging.Logger:
    """
    스트림핸들러 1개만 장착된 로거를 반환.
    - 포맷: <시간> <레벨> <메서드:라인> [로거명] <메시지>
    - 중복 핸들 방지, 전파 금지
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False

    # 레벨 설정
    logger.setLevel(level if level is not None else _desired_level())

    if not logger.handlers:
        # fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        fmt = logging.Formatter(fmt=_LOG_FMT, datefmt=_LOG_DATEFMT)

        fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


    # 외부 라이브러리 노이즈 억제 (원하면 조정)
    for noisy in ("httpx", "urllib3", "chromadb", "langchain", "openai"):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass

    # 파이썬 warnings → logging
    logging.captureWarnings(True)

    logger.debug("로깅 초기화 완료 (level=%s)", logging.getLevelName(logger.level))
    return logger


# 전역 로거(다른 모듈에서 import 후 바로 사용)
LOGGER = setup_logging()


def _shorten(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else (s[:n] + f"...(+{len(s)-n})")


def trace(name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    함수 호출을 [ENTER]/[EXIT]/[RAISE]로 추적하는 데코레이터.
    - 실행시간(ms)과 반환 타입을 함께 출력
    - 로깅 포맷이 '레벨 다음 메서드:라인'이므로, 메시지에는 함수명을 유지 (가독성)
    사용:
        @trace()
        def my_func(...): ...
        @trace("route_node")
        def route(...): ...
    """
    def _decorator(func: Callable[..., T]) -> Callable[..., T]:
        disp = name or func.__name__

        @functools.wraps(func)
        def _wrapped(*args: Any, **kwargs: Any) -> T:
            # ENTER
            try:
                arg_preview = ""
                if args:
                    arg_preview = _shorten(str(args[0]), 96) if len(args) == 1 else _shorten(str(args), 96)
                LOGGER.debug("[ENTER] %s(%s)", disp, arg_preview)
            except Exception:
                LOGGER.debug("[ENTER] %s", disp)

            # RUN
            t0 = time.perf_counter()
            try:
                out = func(*args, **kwargs)
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000.0
                LOGGER.exception("[RAISE] %s (%.1f ms): %s", disp, dt, e)
                raise
            dt = (time.perf_counter() - t0) * 1000.0

            # EXIT
            try:
                out_repr = type(out).__name__
                # 메시지 길이 긴 경우는 타입명만 출력
                if isinstance(out, (str, bytes)):
                    out_repr = f"str({len(out)})"
                LOGGER.debug("[EXIT ] %s -> %s (%.1f ms)", disp, out_repr, dt)
            except Exception:
                LOGGER.debug("[EXIT ] %s (%.1f ms)", disp, dt)

            return cast(T, out)

        return _wrapped
    return _decorator
