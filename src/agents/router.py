# -*- coding: utf-8 -*-
"""사용자 질의를 8개 카테고리 중 하나 이상으로 분류하는 라우터."""
from __future__ import annotations
import json
from langchain_core.prompts import PromptTemplate
from src.config import get_temp_llm
from src.utils.logging_utils import setup_logging, trace
from src.agents.constants import CATEGORIES

logger = setup_logging()

ROUTER_PROMPT = PromptTemplate.from_template(
    # ⚠️ 포맷 변수 {categories}, {question}만 단일 중괄호 사용
    # ✅ JSON 예시는 반드시 {{ ... }} 로 이스케이프!
    """당신은 한국 법률 질의를 **카테고리**로 분류하는 라우터입니다.
가능한 카테고리: {categories}

규칙:
- 반드시 위 목록 중 하나 이상을 반환하세요.
- 애매하면 2~3개를 고르세요.
- JSON 한 줄만 출력하세요. 예: {{"categories": ["민법","부동산"], "reason": "임대차 보증금 분쟁이므로 민법/부동산"}}

[질문]
{question}

# FEW-SHOT
입력: "퇴직금 산정 기준이 궁금해요"
출력: {{"categories": ["노동"], "reason": "근로기준법/노동법"}}

입력: "법인 간 주식 양수도 분쟁"
출력: {{"categories": ["상법","민법"], "reason": "회사/계약 분쟁"}}

입력: "{question}"
출력:
"""
)

@trace()
def route_categories(question: str) -> list[str]:
    """LLM으로 카테고리 배열을 분류. 실패 시 최소한 하나(민법)로 폴백."""
    llm = get_temp_llm(0.0)
    prompt = ROUTER_PROMPT.format(categories=", ".join(CATEGORIES), question=question)
    res = llm.invoke(prompt)
    text = (getattr(res, "content", str(res)) or "").strip()
    try:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1:
            text = text[i:j+1]
        data = json.loads(text)
        out = [c for c in data.get("categories", []) if c in CATEGORIES]
        if out:
            return out
    except Exception as e:
        logger.warning("[ROUTER] parse failed: %s", e)
    # 보수적 폴백(민법)
    return ["민법"]
