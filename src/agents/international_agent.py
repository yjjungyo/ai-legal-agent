from __future__ import annotations
from langchain.agents import AgentExecutor
from src.utils.logging_utils import setup_logging, trace
from src.agents.base_agent import build_react_agent
from src.agents.tools.international_retrieval_tool import get_tool

logger = setup_logging()
_AGENT: AgentExecutor | None = None

@trace()
def get_agent() -> AgentExecutor:
    global _AGENT
    if _AGENT is None:
        tools = [get_tool()]
        system = "당신은 국제법 분야를 담당합니다. 조약/관습국제법/영사·외교관계 및 분쟁해결 절차 등을 구분해 설명하세요."
        _AGENT = build_react_agent("국제", tools, system_override=system)
        logger.debug("[AGENT] 국제 agent built")
    return _AGENT
