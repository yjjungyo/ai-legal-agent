from __future__ import annotations
from langchain.agents import AgentExecutor
from src.utils.logging_utils import setup_logging, trace
from src.agents.base_agent import build_react_agent
from src.agents.tools.labor_retrieval_tool import get_tool

logger = setup_logging()
_AGENT: AgentExecutor | None = None

@trace()
def get_agent() -> AgentExecutor:
    global _AGENT
    if _AGENT is None:
        tools = [get_tool()]
        system = "당신은 노동 분야를 담당합니다. 근로기준법/노동3권/징계·해고 요건/퇴직금 산정 등을 근거와 함께 설명하세요."
        _AGENT = build_react_agent("노동", tools, system_override=system)
        logger.debug("[AGENT] 노동 agent built")
    return _AGENT
