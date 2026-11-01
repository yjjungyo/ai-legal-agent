from __future__ import annotations
from langchain.agents import AgentExecutor
from src.utils.logging_utils import setup_logging, trace
from src.agents.base_agent import build_react_agent
from src.agents.tools.administrative_retrieval_tool import get_tool

logger = setup_logging()
_AGENT: AgentExecutor | None = None

@trace()
def get_agent() -> AgentExecutor:
    global _AGENT
    if _AGENT is None:
        tools = [get_tool()]
        system = "당신은 행정 분야를 담당합니다. 행정행위/처분사유/쟁송절차(행정심판·소송) 등을 구조화하여 안내하세요."
        _AGENT = build_react_agent("행정", tools, system_override=system)
        logger.debug("[AGENT] 행정 agent built")
    return _AGENT
