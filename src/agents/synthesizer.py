# -*- coding: utf-8 -*-
"""
카테고리별 에이전트 결과 LLM 합성 모듈
- 각 카테고리 에이전트의 답변(results)을 받아 LLM으로 종합 요약을 생성
- 최종 메시지: [종합 답변] + 구분선 + [카테고리별 원문 답변] 순서로 출력
- 내부 사고(Chain-of-Thought)는 노출하지 않도록 시스템 프롬프트에서 금지
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain_core.prompts import PromptTemplate

from src.utils.logging_utils import setup_logging, trace
from src.config import get_temp_llm  # ✅ AzureChatOpenAI 팩토리

logger = setup_logging()

__all__ = ["synthesize"]


# -------------------------------
# 헬퍼: 대화 요약 / 최신 질문 추출
# -------------------------------
def _brief_history(messages: List[AnyMessage], max_chars: int = 800) -> str:
    """대화 히스토리를 간단 요약(LLM 합성 컨텍스트로 전달)."""
    if not messages:
        return "(이전 대화 없음)"
    parts: List[str] = []
    for m in messages[-10:]:
        role = "사용자" if isinstance(m, HumanMessage) else "도우미"
        text = m.content if isinstance(m.content, str) else str(m.content)
        text = text.replace("\n", " ")
        if len(text) > 160:
            text = text[:160] + f"...(+{len(text)-160})"
        parts.append(f"{role}: {text}")
    s = " / ".join(parts)
    return s[:max_chars] + ("... (trunc)" if len(s) > max_chars else "")


def _extract_last_question(messages: List[AnyMessage]) -> str:
    """히스토리에서 가장 최근 사용자 메시지(질문)를 가져온다."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _format_findings(results: Dict[str, str], per_cat_limit: int = 2400) -> str:
    """
    LLM에 전달할 카테고리별 요약 입력을 만든다.
    - 너무 긴 답변은 per_cat_limit 글자 기준으로 자른다.
    """
    lines: List[str] = []
    for cat, text in results.items():
        t = (text or "").strip()
        if len(t) > per_cat_limit:
            t = t[:per_cat_limit] + f"...(+{len(text)-per_cat_limit})"
        lines.append(f"### {cat}\n{t}\n")
    return "\n".join(lines).strip()


def _render_originals(results: Dict[str, str]) -> str:
    """최종 답변 아래에 붙일 '카테고리별 원문 답변' 섹션을 만든다."""
    lines: List[str] = ["### 참고 : 카테고리별 원문 답변"]
    for cat, text in results.items():
        lines.append(f"\n#### {cat} 분야 전문가(Agent) 답변")
        lines.append(text or "(응답 없음)")
    return "\n".join(lines)


# -------------------------------
# 프롬프트 (비공개 CoT 금지 + Few-shot)
# -------------------------------
# SYNTH_SYSTEM = """당신은 대한민국 법률 전문가 AI입니다.
# - 사용자의 질문과 카테고리별 다른 에이전트 답변을 바탕으로, 하나의 일관된 최종 답변을 한국어로 작성하세요.
# - 내부 사고(Chain-of-Thought)는 출력하지 마세요. 최종 결론/근거 요약/주의사항만 명확하게 제시하세요.
# - 중복 내용은 합치고, 모순되는 부분은 조정하거나 '쟁점'으로 정리하세요.
# - 법률 명칭/조문/판례 등은 가능한 범위에서 출처(카테고리/문서 메타 등)를 간단히 표기하세요.
# - 구체적 사건 대리나 서면 작성은 안내를 거절하고 변호사 상담을 권유하세요.
# - 출력은 Markdown 형식으로 간결하고 구조화(번호/불릿/소제목)해 주세요.
# """
SYNTH_SYSTEM = """당신은 대한민국 법률 전문가입니다.
- 사용자의 질문과 카테고리별 다른 에이전트 답변을 바탕으로, 하나의 일관된 최종 답변을 한국어로 작성하세요.
- 내부 사고(Chain-of-Thought)는 출력하지 마세요. 최종 결론/근거 요약/주의사항만 명확하게 제시하세요.
- 중복 내용은 합치고, 모순되는 부분은 조정하거나 '쟁점'으로 정리하세요.
- 법률 명칭/조문/판례 등은 다른 에이전트에서 답변한 출처(카테고리/문서/메타데이터)를 반드시 표기하세요.
- 출력은 Markdown 형식으로 간결하고 구조화(번호/불릿/소제목)해 주세요.
"""

SYNTH_FEWSHOT = """
예시)
[질문] 전세보증금 반환 절차가 궁금합니다.
[에이전트별 답변 요약]
- 민법: 임대차 종료/실거주/내용증명/소송 절차…
- 행정: 주택도시보증 공사 보증 제도…
- 조세: 원천징수/소득세 이슈 없음…

[좋은 종합 답변의 특징]
1) 핵심 결론: 반환 요구 절차 → 증거 확보 → 내용증명 → 소송/집행
2) 선택지/분기: 보증보험 가입 시 경로, 미가입 시 경로
3) 주의: 임대차 기간/대항력/확정일자 등 체크리스트
4) 참고: 민법 임대차 규정, 관련 제도 링크(가능시)
"""


SYNTH_PROMPT = PromptTemplate.from_template(
    """{system}

# 대화 요약
{chat_brief}

# 사용자 질문
{question}

# 카테고리별 에이전트 답변(요약 입력)
{findings}

{fewshot}

# 지시
위 정보를 종합하여, 하나의 일관된 **최종 답변**을 작성하세요.
- 불확실한 부분은 가정/한계를 명시하세요.
- 핵심 조치/절차/기준/요건을 항목화하세요.
- 마지막에 간단한 면책 문구를 포함하세요.
"""
)


# -------------------------------
# 공개 API
# -------------------------------
@trace()
def synthesize(messages: List[AnyMessage], results: Dict[str, str]) -> List[AnyMessage]:
    """
    카테고리별 결과를 LLM으로 종합해 최종 메시지를 생성하고,
    그 아래에 카테고리별 원문 답변을 붙여 반환한다.
    """
    # 예외/빈 결과 처리
    if not results:
        final = "관련 카테고리에서 유의미한 답변이 생성되지 않았습니다. 질문을 구체화해 주세요."
        logger.debug("[SYN] empty results -> fallback")
        return messages + [AIMessage(content=final)]

    # 합성 컨텍스트 준비
    chat_brief = _brief_history(messages)
    question = _extract_last_question(messages)
    findings = _format_findings(results, per_cat_limit=2400)

    # LLM 호출
    try:
        llm = get_temp_llm(0)  # 약간의 창의성 허용
        prompt = SYNTH_PROMPT.format(
            system=SYNTH_SYSTEM,
            chat_brief=chat_brief,
            question=question or "(질문 없음)",
            findings=findings,
            fewshot=SYNTH_FEWSHOT,
        )
        logger.debug("[SYN] prompt_len=%d, cats=%d", len(prompt), len(results))
        resp = llm.invoke(prompt)
        summary = (getattr(resp, "content", str(resp)) or "").strip()
        logger.debug("[SYN] summary_len=%d", len(summary))
    except Exception as e:
        logger.exception("[SYN] LLM 합성 실패, 원문만 반환")
        # LLM 실패 시, 원문만이라도 반환
        summary = "종합 요약 생성 중 오류가 발생했습니다. 아래 카테고리별 답변을 참고해 주세요."

    # 최종 메시지 구성: [종합 답변] + 구분선 + [카테고리별 원문 답변]
    originals = _render_originals(results)
    final_md = f"{summary}\n\n---\n{originals}"

    return messages + [AIMessage(content=final_md)]
