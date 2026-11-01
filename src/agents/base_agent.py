# -*- coding: utf-8 -*-
"""Tool Calling 에이전트 공통 빌더(프롬프트/LLM/툴 조합)."""
from __future__ import annotations
from typing import List, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.config import get_temp_llm
from src.utils.logging_utils import setup_logging, trace

logger = setup_logging()
# 카테고리 공통 시스템 지침 (툴 사용 및 근거 표기)
BASE_SYSTEM = """\
당신은 대한민국 {category} 분야 법률 전문가입니다.
- 검색 도구를 활용하여 관련 법령 문서를 검색하고, 검색 결과를 토대로 사용자의 질문에 정확히 답변하세요.
- 검색 도구는 필요한 경우 1~2회 호출하여 관련 조문과 근거를 확보하세요.
- 답변에는 인용한 **법령명, 조문 번호, 파일명과 페이지(또는 범위)** 등의 출처/근거를 명확히 포함하세요.
- 한국어로 명확하고 구조화하여 간결하게 답하세요.
"""

def _primary_tool_name(tools: List) -> str:
    try:
        return getattr(tools[0], "name", "") or "retrieval_tool"
    except Exception:
        return "retrieval_tool"

@trace()
def build_react_agent(
    category: str,
    tools: List,
    system_override: Optional[str] = None,
) -> AgentExecutor:
    """
    카테고리 전담 Tool Calling Agent 생성.
    - system_override: 분야 특화 지침을 추가하고 싶을 때 전달 (없으면 기본 사용)
    """
    llm = get_temp_llm(0.0)
    primary_tool = _primary_tool_name(tools)

    sys_prompt = BASE_SYSTEM.format(category=category)
    if system_override:
        sys_prompt = f"{sys_prompt}\n{system_override.strip()}"

    sys_prompt = f"{sys_prompt}\n\n[이전 대화 요약]\n{{chat_brief}}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]).partial(chat_brief="(이전 대화 없음)")

    logger.debug("prompt=%s", prompt)
    agent = create_tool_calling_agent(llm, tools, prompt)
    exec_ = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=3,
        max_execution_time=60,
        early_stopping_method="force",
        return_intermediate_steps=False,  # ← 선택: 중간 도구 호출 로그 확인용
    )
    logger.debug("[AGENT] built: category=%s tools=%d primary_tool=%s", category, len(tools), primary_tool)
    return exec_

