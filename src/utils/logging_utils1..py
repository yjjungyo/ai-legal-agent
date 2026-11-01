
import logging, os
from logging.handlers import RotatingFileHandler
from typing import Literal
from src.config import LOG_LEVEL, LOG_PATH
import functools, time, logging
from typing import Any, Callable, Tuple, Dict

def _short(v: Any, maxlen: int = 120) -> str:
    """값을 한 줄 요약 문자열로 변환(민감도 낮춤)."""
    s = str(v)
    return s if len(s) <= maxlen else s[:maxlen] + f"...(+{len(s)-maxlen})"

def trace(name: str | None = None, level: int = logging.DEBUG):
    """
    함수 진입/종료/예외를 표준화해 로그로 남기는 데코레이터.
    - name: 로거에 표시할 별칭(기본은 func.__qualname__)
    - level: DEBUG(기본) ~ INFO 등
    사용:
        @trace()
        def my_func(a,b): ...
    """
    def _decorator(func: Callable):
        log = logging.getLogger("ai-legal-agent")
        qual = func.__qualname__ #+ name

        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            t0 = time.time()
            try:
                # 인자 요약(길이 제한)
                args_s = ", ".join(_short(a) for a in args[:4])
                kwargs_s = ", ".join(f"{k}={_short(v)}" for k, v in list(kwargs.items())[:6])
                log.log(level, f"[ENTER] {qual}({args_s}{', ' if args_s and kwargs_s else ''}{kwargs_s})")
                out = func(*args, **kwargs)
                dt = (time.time() - t0) * 1000
                # 결과 요약(길이 제한)
                out_s = _short(out)
                log.log(level, f"[EXIT ] {qual} -> {_short(type(out).__name__)} ({dt:.1f} ms): {out_s}")
                return out
            except Exception as e:
                dt = (time.time() - t0) * 1000
                log.exception(f"[RAISE] {qual} ({dt:.1f} ms): {e}")
                raise
        return _wrapped
    return _decorator

def setup_logging() -> logging.Logger:
    """
    Streamlit 재실행(rerun)을 고려해 중복 핸들러 추가를 방지.
    - RotatingFileHandler: ./storage/app.log
    - StreamHandler: 콘솔 (개발 편의)
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    logger = logging.getLogger("ai-legal-agent")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        # fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        fmt = logging.Formatter("%(asctime)s %(levelname)s ] %(message)s")

        fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger

def set_log_level(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("ai-legal-agent")
    root.setLevel(lvl)
    for h in root.handlers:
        h.setLevel(lvl)

def tail_log(path: str, lines: int = 200) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            buf = f.readlines()
        return "".join(buf[-lines:])
    except FileNotFoundError:
        return "(아직 로그가 없습니다)"

def clear_log(path: str = LOG_PATH):
    try:
        open(path, "w", encoding="utf-8").close()
    except FileNotFoundError:
        pass