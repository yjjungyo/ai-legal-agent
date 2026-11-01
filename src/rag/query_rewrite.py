# -*- coding: utf-8 -*-
"""
카테고리 전용 질의 재작성 (심플 버전)
- 목적: 벡터 검색에 유리하도록 사용자 질문을 짧은 키워드로 보강
- 구현: LLM 한 번 호출해 5~8개 핵심 용어/조문만 JSON으로 받아 결합
- 폴백: LLM 실패 시, 질문에서 2글자 이상 한글/숫자 토큰 몇 개만 추려서 사용
"""

from __future__ import annotations
from typing import List
import json
import re

from langchain_core.prompts import PromptTemplate
from src.utils.logging_utils import setup_logging, trace
from src.config import get_temp_llm

logger = setup_logging()

# ---------------------------------------------------------------------
# LLM 프롬프트: 카테고리 고정, JSON 한 줄
# ---------------------------------------------------------------------
LLM_REWRITE_PROMPT = PromptTemplate.from_template(
    # ⚠️ {category}, {question}만 단일 중괄호 사용 / JSON 예시는 {{ }} 이스케이프
    """당신은 한국 법률 **{category}** 분야 검색 전문가입니다.
다음 사용자 질문을 이 분야의 핵심 **조문/법률용어 중심**으로 간결하게 보강하세요.

[출력]
- **JSON 한 줄만**: {{"terms": ["키워드1","키워드2", ...]}}
- 2~3개, 중복/일반어/말끝 조사 제외, 한국어 위주, 법조문 번호 허용(예: "민법 제447조")

[질문]
{question}

# FEW-SHOT
입력: (민법) "연대보증인데 주채무자 도피, 내 책임 범위?"
# 출력: {{"terms":["연대보증","연대채무","보증채무","구상권","분담","면책항변","대위","민법 제447조"]}}
출력: {{"terms":["연대보증","연대채무","보증채무"]}}

입력: (형법) "사기죄로 고발, 대비할 점?"
출력: {{"terms":["사기죄","고소","고발"]}}

입력: ({category}) "{question}"
출력:
"""
)

def _parse_terms(text: str) -> List[str]:
    """모델 출력에서 JSON 한 줄을 파싱해 terms 배열을 반환. 실패 시 []."""
    if not text:
        return []
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return []
    try:
        data = json.loads(text[i:j+1])
        terms = data.get("terms") or []
        return [t.strip() for t in terms if isinstance(t, str) and t.strip()]
    except Exception:
        return []

def _cheap_fallback_terms(q: str, k: int = 3) -> List[str]:
    """
    LLM 실패 시 아주 단순한 폴백:
    - 2글자 이상 한글/숫자 토큰을 뽑아 앞쪽 몇 개만 사용
    - 조사/불용어는 제거하지 않지만, 벡터검색에는 큰 문제 없음
    """
    q = (q or "").strip()
    tokens = re.findall(r"[가-힣0-9A-Za-z]{2,}", q)
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out

@trace()
def rewrite_query(question: str, category_ko: str, *, max_terms: int = 3) -> str:
    """
    심플 리라이트:
      1) LLM으로 1~2개 핵심 용어 생성
      2) 실패 시 폴백 토큰 몇 개 사용
      3) 최종 쿼리 = "원질문 + 한 줄 키워드"
    반환 예:
      "친구랑 연대보증... 어떻게 해야 하지?\n키워드: 연대보증, 연대채무, 구상권, ... "
    """
    q = (question or "").strip().replace("\n", " ")

    # 1) LLM 시도
    terms: List[str] = []
    try:
        llm = get_temp_llm(0)  # 엄격 또는, 약간의 다양성만 허용 #변수화 필요
        prompt = LLM_REWRITE_PROMPT.format(category=category_ko, question=q)
        raw = llm.invoke(prompt)
        text = (getattr(raw, "content", str(raw)) or "").strip()
        terms = _parse_terms(text)[:max_terms]
        logger.debug("[REWRITE] llm_terms=%s", terms)
    except Exception as e:
        logger.warning("[REWRITE] LLM 실패: %s", e)

    # 2) 폴백
    if not terms:
        fb = _cheap_fallback_terms(q, k=min(3, max_terms))
        logger.debug("[REWRITE] fallback_terms=%s", fb)
        terms = fb

    # 3) 결합
    final = q
    if terms:
        final = f"{q}\n키워드: {', '.join(terms)}"
    logger.debug("[REWRITE] final_len=%d final=[%s]", len(final), final)
    return final
