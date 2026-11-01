# -*- coding: utf-8 -*-
"""
LangGraph 워크플로우 (라우팅→플랜→병렬 실행→합성)
- MVP: MemorySaver 사용 (이미 적용된 버전 유지)
- 라우팅/플랜은 한글 라벨 기준
- 실행은 정적 매핑으로 각 파일의 get_agent 호출
"""

from __future__ import annotations
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AnyMessage

from src.utils.logging_utils import setup_logging, trace
from src.agents.router import route_categories
from src.agents.planner import plan_tools
from src.agents.synthesizer import synthesize

# 8개 정적 import
from src.agents.constitution_agent import get_agent as get_constitution
from src.agents.civil_agent import get_agent as get_civil
from src.agents.criminal_agent import get_agent as get_criminal
from src.agents.commercial_agent import get_agent as get_commercial
from src.agents.labor_agent import get_agent as get_labor
from src.agents.administrative_agent import get_agent as get_administrative
from src.agents.tax_agent import get_agent as get_tax
from src.agents.international_agent import get_agent as get_international
import re
logger = setup_logging()

class LegalState(MessagesState):
    question: str
    categories: List[str]
    plan: List[str]
    results: Dict[str, str]

def _brief_history(messages: List[AnyMessage], max_chars: int = 1000) -> str:
    if not messages:
        return "(이전 대화 없음)"
    parts: List[str] = []
    for m in messages[:-1][-5:]: # 전체 메시지 리스트에서 마지막 메시지를 제외한 최근 10개
        role = "사용자" if isinstance(m, HumanMessage) else "도우미"
        text = m.content if isinstance(m.content, str) else str(m.content)
        text = text.replace("\n", " ")
        if len(text) > 160:
            text = text[:160] + f"...(+{len(text)-160})"
        parts.append(f"{role}: {text}")
    s = " / ".join(parts)
    return s[:max_chars] + ("... (trunc)" if len(s) > max_chars else "")

# 한글 라벨 → get_agent 함수 매핑 (정적)
_GET_AGENT = {
    "헌법": get_constitution,
    "민법": get_civil,
    "형법": get_criminal,
    "상법": get_commercial,
    "노동": get_labor,
    "행정": get_administrative,
    "조세": get_tax,
    "국제": get_international,
}

# @trace()
# def _extract_citations(intermediate_steps) -> list[tuple[str, str]]:
#     """
#     AgentExecutor(return_intermediate_steps=True)로 받은 중간 스텝에서
#     [CITE] file=... page=... 패턴을 추출해 (file, page) 목록으로 반환.
#     """
#     cites: list[tuple[str, str]] = []
#     try:
#         for step in intermediate_steps or []:
#             # step: (AgentAction, observation)
#             obs = None
#             if isinstance(step, tuple) and len(step) > 1:
#                 obs = step[1]
#             elif isinstance(step, dict) and "observation" in step:
#                 obs = step["observation"]
#             if not isinstance(obs, str):
#                 continue

#             for line in obs.splitlines():
#                 if line.strip().startswith("[metadata]"):
#                     m = re.search(r"file=(.+?)\s+page=(\d+)", line)
#                     if m:
#                         file_ = m.group(1).strip()
#                         page_ = m.group(2).strip()
#                         cites.append((file_, page_))
#     except Exception:
#         pass

#     # 중복 제거 + 상위 5개 제한
#     seen, out = set(), []
#     for fp in cites:
#         if fp not in seen:
#             seen.add(fp)
#             out.append(fp)
#         if len(out) >= 5:
#             break
#     return out

# @trace()
# def _attach_citations(text: str, citations: list[tuple[str, str]]) -> str:
#     """기존 답변에 '출처:' 섹션이 없다면 붙여준다."""
#     if not citations:
#         return text
#     if "출처:" in text:
#         return text  # 이미 붙였으면 그대로 둠
#     lines = ["\n출처:"]
#     lines += [f"- {f} p.{p}" for f, p in citations]
#     return (text.rstrip() + "\n" + "\n".join(lines)).strip()

@trace()
def route_node(state: LegalState) -> LegalState:
    q = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            q = m.content if isinstance(m.content, str) else str(m.content)
            break
    q = q or "(질문 미확인)"
    cats = route_categories(q)  # ['민법', ...]
    logger.debug("[NODE:route] q_len=%d cats=%s", len(q), cats)
    state["question"] = q
    state["categories"] = cats
    return state

@trace()
def plan_node(state: LegalState) -> LegalState:
    cats = state.get("categories", []) or []
    plan = plan_tools(cats) #category 목록에 해당하는 카테고리별 Agent 목록을 찾기
    state["plan"] = plan
    logger.debug("[NODE:plan] plan=%s", plan)
    return state

@trace()
def _run_one(category_ko: str, question: str, chat_brief: str) -> tuple[str, str]:
    get = _GET_AGENT.get(category_ko)
    if not get:
        logger.warning("[RUN] 지원하지 않는 카테고리: %s", category_ko)
        return (category_ko, "(지원하지 않는 카테고리)")
    agent = get()
    out = agent.invoke({"input": question, "chat_brief": chat_brief})
    text = (out.get("output", "") or "").strip()
    logger.debug("[%s Agent Result]-> ans_len=%d Answer : \n%s" , category_ko, len(text), text)
    return (category_ko, text or "(응답 없음)")

# @trace()
# def _run_one(category_ko: str, question: str, chat_brief: str) -> tuple[str, str]:
#     get = _GET_AGENT.get(category_ko)
#     if not get:
#         logger.warning("[RUN] 지원하지 않는 카테고리: %s", category_ko)
#         return (category_ko, "(지원하지 않는 카테고리)")
#     agent = get()
#     out = agent.invoke({"input": question, "chat_brief": chat_brief})

#     # 본문
#     text = (out.get("output", "") or "").strip()

#     # ✅ 중간 스텝에서 [CITE]를 추출하여 '출처:' 강제 부착
#     citations = _extract_citations(out.get("intermediate_steps"))
#     if citations:
#         text = _attach_citations(text, citations)
#     else:
#         # 도구 호출이 없었다면, 추측 방지를 위해 경고 텍스트를 추가 (옵션)
#         logger.warning("[RUN] %s: 도구 인용을 찾지 못함(무인용 답변)", category_ko)
#         text = (
#             "자료 부족: 해당 카테고리 벡터스토어의 문서 근거를 찾지 못했습니다. "
#             "질문을 더 구체화하거나 다른 키워드로 다시 시도해 주세요."
#         )

#     logger.debug("[RUN] %s -> ans_len=%d cites=%d", category_ko, len(text), len(citations))
#     return (category_ko, text or "(응답 없음)")



@trace()
def run_agents_node(state: LegalState) -> LegalState:
    cats = state.get("categories", []) or []
    q = state.get("question", "")
    brief = _brief_history(state.get("messages", []))

    results: Dict[str, str] = {}
    if not cats:
        state["results"] = results
        return state

    max_workers = min(4, len(cats))
    logger.debug("[NODE:run_agents] cats=%s workers=%d", cats, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Each worker inherits the graph's callbacks and parent run context.
        futs = [ex.submit(copy_context().run, _run_one, c, q, brief) for c in cats]
        for f in as_completed(futs):
            cat, text = f.result()
            results[cat] = text

    state["results"] = results
    logger.debug("[NODE:run_agents] done: %s", list(results.keys()))
    logger.debug("-"*50)
    logger.debug("Category Agent Results : \n %s", state["results"])
    return state

@trace()
def synthesize_node(state: LegalState) -> LegalState:
    msgs = synthesize(state["messages"], state.get("results", {}))
    state["messages"] = msgs
    return state

@trace()
def build_legal_graph():
    graph = StateGraph(LegalState)
    graph.add_node("route", route_node)
    graph.add_node("plan", plan_node)
    graph.add_node("run_agents", run_agents_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "route")
    graph.add_edge("route", "plan")
    graph.add_edge("plan", "run_agents")
    graph.add_edge("run_agents", "synthesize")
    graph.add_edge("synthesize", END)

    app = graph.compile(checkpointer=MemorySaver())
    logger.debug("[GRAPH] build complete (MemorySaver, static agent mapping)")
    return app
