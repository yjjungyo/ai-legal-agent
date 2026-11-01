from __future__ import annotations
from langchain.agents import AgentExecutor
from src.utils.logging_utils import setup_logging, trace
from src.agents.base_agent import build_react_agent
from src.agents.tools.constitution_retrieval_tool import get_tool

logger = setup_logging()
_AGENT: AgentExecutor | None = None

@trace()
def get_agent() -> AgentExecutor:
    global _AGENT
    if _AGENT is None:
        tools = [get_tool()]
        system = "당신은 헌법 분야를 담당합니다. 국가기관 권한, 기본권, 헌법소송 절차 등을 명확히 구분해 설명하세요."
        _AGENT = build_react_agent("헌법", tools, system_override=system)
        logger.debug("[AGENT] 헌법 agent built")
    return _AGENT
