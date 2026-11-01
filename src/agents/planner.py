# -*- coding: utf-8 -*-
"""라우팅된 카테고리 → 실행 에이전트 목록으로 변환(계획)."""
from __future__ import annotations
from typing import List
from src.utils.logging_utils import setup_logging, trace
from src.agents.constants import get_agent_name

logger = setup_logging()

@trace()
def plan_tools(categories: List[str]) -> List[str]:
    """
    분류된 카테고리 목록을 받아 병렬 실행할 에이전트 이름 목록으로 변환.
    - 규칙: agent__<slug(category)>
    """
    plan = [get_agent_name(c) for c in categories]
    logger.debug("[PLAN] %s -> %s", categories, plan)
    return plan
