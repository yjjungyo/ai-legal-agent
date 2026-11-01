from __future__ import annotations
from langchain.agents import AgentExecutor
from src.utils.logging_utils import setup_logging, trace
from src.agents.base_agent import build_react_agent
from src.agents.tools.criminal_retrieval_tool import get_tool

logger = setup_logging()
_AGENT: AgentExecutor | None = None

@trace()
def get_agent() -> AgentExecutor:
    global _AGENT
    if _AGENT is None:
        tools = [get_tool()]
        system = "당신은 형법 분야를 담당합니다. 구성요건 해당성/위법성/책임을 단계적으로 검토하고 판례 태도를 반영하세요."
        _AGENT = build_react_agent("형법", tools, system_override=system)
        logger.debug("[AGENT] 형법 agent built")
    return _AGENT
