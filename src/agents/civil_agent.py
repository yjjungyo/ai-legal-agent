from __future__ import annotations
from langchain.agents import AgentExecutor
from src.utils.logging_utils import setup_logging, trace
from src.agents.base_agent import build_react_agent
from src.agents.tools.civil_retrieval_tool import get_tool

logger = setup_logging()
_AGENT: AgentExecutor | None = None

@trace()
def get_agent() -> AgentExecutor:
    global _AGENT
    if _AGENT is None:
        tools = [get_tool()]
        system = "당신은 민법 분야를 담당합니다. 계약/불법행위/물권/가족/상속 쟁점을 식별하고 요건사실과 입증책임을 정리하세요."
        _AGENT = build_react_agent("민법", tools, system_override=system)
        logger.debug("[AGENT] 민법 agent built")
    return _AGENT
