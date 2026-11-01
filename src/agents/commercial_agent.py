from __future__ import annotations
from langchain.agents import AgentExecutor
from src.utils.logging_utils import setup_logging, trace
from src.agents.base_agent import build_react_agent
from src.agents.tools.commercial_retrieval_tool import get_tool

logger = setup_logging()
_AGENT: AgentExecutor | None = None

@trace()
def get_agent() -> AgentExecutor:
    global _AGENT
    if _AGENT is None:
        tools = [get_tool()]
        system = "당신은 상법 분야를 담당합니다. 회사/어음·수표/보험 등 각 파트를 구분하고 실무상 쟁점을 정리하세요."
        _AGENT = build_react_agent("상법", tools, system_override=system)
        logger.debug("[AGENT] 상법 agent built")
    return _AGENT
