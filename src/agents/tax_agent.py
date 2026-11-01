from __future__ import annotations
from langchain.agents import AgentExecutor
from src.utils.logging_utils import setup_logging, trace
from src.agents.base_agent import build_react_agent
from src.agents.tools.tax_retrieval_tool import get_tool

logger = setup_logging()
_AGENT: AgentExecutor | None = None

@trace()
def get_agent() -> AgentExecutor:
    global _AGENT
    if _AGENT is None:
        tools = [get_tool()]
        system = "당신은 조세 분야를 담당합니다. 과세표준/세율/불복절차(이의신청·심판·소송)와 신고기한 등을 정확히 명시하세요."
        _AGENT = build_react_agent("조세", tools, system_override=system)
        logger.debug("[AGENT] 조세 agent built")
    return _AGENT
