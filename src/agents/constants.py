# -*- coding: utf-8 -*-
"""
카테고리/영문 키 상수 정의 (정적 매핑)
- 컬렉션명 규칙: law_docs_<english>
- UI/라우팅은 한글 라벨 사용, Vectorstore는 영문 키 사용
- get_agent_name(): 한글/영문 입력을 받아 '<english>_agent'로 반환
- get_tool_name(): 한글/영문 입력을 받아 '<english>_retrieval_tool'로 반환
"""

from __future__ import annotations
from typing import Dict, List

# (선택) 디버그 로깅: 상수 모듈 초기화 시 로거가 없을 수도 있어 try 처리
try:
    from src.utils.logging_utils import setup_logging
    _logger = setup_logging()
except Exception:
    _logger = None

# ---------------------------------------------------------------------
# 카테고리 라벨/영문 키
# ---------------------------------------------------------------------
CATEGORIES: List[str] = [
    "헌법", "민법", "형법", "상법", "노동", "행정", "조세", "국제",
]

CATEGORY_KO_TO_ENG: Dict[str, str] = {
    "헌법": "constitution",
    "민법": "civil",
    "형법": "criminal",
    "상법": "commercial",
    "노동": "labor",
    "행정": "administrative",
    "조세": "tax",
    "국제": "international",
}

CATEGORY_ENG_TO_KO: Dict[str, str] = {v: k for k, v in CATEGORY_KO_TO_ENG.items()}

# 영문 키 허용 집합 (검증용)
_ALLOWED_ENG_KEYS = set(CATEGORY_KO_TO_ENG.values())


def to_eng(category: str) -> str:
    """
    한글 라벨 → 영문 라벨. 이미 영문이면 그대로 반환.
    - 허용되지 않은 임의 문자열을 넣으면 그대로 돌아오므로,
      호출부에서 _ALLOWED_ENG_KEYS로 검증하는 것이 안전하다.
    """
    if not category:
        return ""
    return CATEGORY_KO_TO_ENG.get(category, category)


def collection_name_for(category: str) -> str:
    """컬렉션명 생성: law_docs_<english>"""
    eng = to_eng(category)
    return f"law_docs_{eng}"


# ---------------------------------------------------------------------
# 에이전트/툴 이름 유틸 (to_eng 기반으로 단순 조합)
# ---------------------------------------------------------------------
def get_agent_name(category: str) -> str:
    """
    한글/영문 카테고리를 받아 '<english>_agent'를 반환.
    예) '민법' → 'civil_agent', 'civil' → 'civil_agent'

    미지원 카테고리인 경우 KeyError를 발생시켜 조기에 버그를 발견한다.
    """
    eng = to_eng(category)
    if eng not in _ALLOWED_ENG_KEYS:
        if _logger:
            _logger.warning(
                "[CONST] get_agent_name: 미지원 카테고리 입력='%s' (허용: ko=%s, eng=%s)",
                category, CATEGORIES, sorted(_ALLOWED_ENG_KEYS),
            )
        raise KeyError(
            f"Unsupported category: '{category}'. "
            f"Allowed KO: {CATEGORIES} / ENG: {sorted(_ALLOWED_ENG_KEYS)}"
        )
    name = f"{eng}_agent"
    if _logger:
        _logger.debug("[CONST] get_agent_name('%s') -> '%s'", category, name)
    return name


def get_tool_name(category: str) -> str:
    """
    한글/영문 카테고리를 받아 '<english>_retrieval_tool'을 반환.
    예) '행정' → 'administrative_retrieval_tool'
    """
    eng = to_eng(category)
    if eng not in _ALLOWED_ENG_KEYS:
        if _logger:
            _logger.warning(
                "[CONST] get_tool_name: 미지원 카테고리 입력='%s' (허용: ko=%s, eng=%s)",
                category, CATEGORIES, sorted(_ALLOWED_ENG_KEYS),
            )
        raise KeyError(
            f"Unsupported category: '{category}'. "
            f"Allowed KO: {CATEGORIES} / ENG: {sorted(_ALLOWED_ENG_KEYS)}"
        )
    name = f"{eng}_retrieval_tool"
    if _logger:
        _logger.debug("[CONST] get_tool_name('%s') -> '%s'", category, name)
    return name


# ---------------------------------------------------------------------
# 모듈 내보내기
# ---------------------------------------------------------------------
__all__ = [
    "CATEGORIES",
    "CATEGORY_KO_TO_ENG",
    "CATEGORY_ENG_TO_KO",
    "to_eng",
    "collection_name_for",
    "get_agent_name",
    "get_tool_name",
]
