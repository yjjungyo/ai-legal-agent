# -*- coding: utf-8 -*-
"""
고정 이름 Retrieval Tool 팩토리
- 예: make_fixed_retrieval_tool(tool_name="civil_retrieval_tool", category_eng="civil", label_ko="민법")
"""
from __future__ import annotations
from langchain_core.tools import Tool

from src.utils.logging_utils import setup_logging, trace
from src.rag.retrieval import retrieve as _retrieve  # 검색 핵심 API (카테고리별 컬렉션 조회)
from src.rag.query_rewrite import rewrite_query #vector 검색용 쿼리 재작성

logger = setup_logging()

MAX_DOCS = 2 #tool retrieve 문서 개수를 3개로 한정
SNIP = 240  #tool retrieve 결과를 240자 내로 한정

@trace()
def make_fixed_retrieval_tool(tool_name: str, category_eng: str, label_ko: str) -> Tool:
    """
    고정 이름 Retrieval Tool 생성. 동적 생성 툴로 했다가, 헷갈려서 수정
    - tool_name: "civil_retrieval_tool" 같이 완전 고정된 툴 이름
    - category_eng: "civil" 같이 Vectorstore 컬렉션 키(영문)
    - label_ko: 한글 라벨(로깅/설명용)
    """
    @trace(name=f"Tool.{tool_name}")
    def _run(query: str) -> str:
        logger.debug("[TOOL:%s] query_len=%d label=%s eng=%s",
                     tool_name, len(query or ""), label_ko, category_eng)
        
        # 벡터 검색 최적화를 위해 키워드 추출
        rewritten_query = rewrite_query(query or "", label_ko, max_terms=2)
        logger.debug("벡터 검색 최적화 최종 키워드 추출 결과 : ")
        logger.debug("query : %s", query)
        logger.debug("rewritten_query : %s", rewritten_query)

        # retrieve()는 category에 '영문 키'를 넘기면 컬렉션 law_docs_<eng> 를 검색
        # docs = _retrieve(query=query, k=3, category=category_eng)
        docs = _retrieve(query=rewritten_query, k=MAX_DOCS, category=category_eng)
        if not docs:
            return f"[검색결과 없음] '[{label_ko}]-[{rewritten_query}]' 관련 문서를 찾지 못했습니다."

        lines = []
        # for d in docs[:]:
        #     m = d.metadata or {}
        #     lines.append(
        #         f"- {m.get('file_name','?')} p.{m.get('page','?')} "
        #         f"({m.get('law_name','')}/{m.get('doc_type','')}/{m.get('law_number','')}/{m.get('effective_date','')})\n"
        #         f"  {d.page_content}"
        #     )
        for d in docs[:MAX_DOCS]:
            m = d.metadata or {}
            fn = m.get("file_name", "?")
            p1 = m.get("page", "?")
            p2 = m.get("page_end", p1)
            page_disp = str(p1) if p1 == p2 or p2 in (None, "?", "") else f"{p1}-{p2}"

            snip = (d.page_content or "").strip().replace("\n", " ")
            if len(snip) > SNIP:
                snip = f"{snip[:SNIP]}...(+{len(d.page_content)-SNIP})"

            lines.append(f"[CITE] file={fn} page={page_disp}\n{snip}")

        lines.append("[GUIDE] 위 1~3건을 근거로 즉시 최종 답변을 작성하세요. 추가 검색 금지.")
        return "\n".join(lines)

    return Tool(
        name=tool_name,
        description=f"'{label_ko}' 전용 RAG 검색 도구(컬렉션=law_docs_{category_eng}).",
        func=_run,
    )
