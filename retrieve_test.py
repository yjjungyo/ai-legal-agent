# -*- coding: utf-8 -*-
"""
retrieve_test.py
- RAG 검색이 정상 동작 확인
- 사용법 예:
    python retrieve_test.py -q "전세 보증금" -c 민법 -k 5
    python retrieve_test.py -q "주권" -c 헌법 --show-collections
    python retrieve_test.py -q "근로계약 해지"          # 카테고리 미지정: 전체 컬렉션 병합 검색


"""

import os
import sys
import argparse
from typing import Optional

# --------------------------------------------------------------------
# 1) 모듈 경로 보정 (Windows/IDE에서 import 경로 꼬임 방지)
# --------------------------------------------------------------------
ROOT = os.path.abspath(".")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --------------------------------------------------------------------
# 2) 프로젝트 모듈 임포트
# --------------------------------------------------------------------
from src.utils.logging_utils import setup_logging  # 디버그 로그 초기화
from src.rag.retrieval import retrieve as rag_retrieve
from src.agents.constants import (
    CATEGORIES, to_eng, collection_name_for,
)
from src.config import CHROMA_PERSIST

# 컬렉션 존재 여부 확인용(테스트 스크립트에서만 사용)
try:
    import chromadb
except Exception:
    chromadb = None

logger = setup_logging()


def list_collections() -> list[str]:
    """Chroma에 존재하는 모든 컬렉션 이름 리스트(테스트용)."""
    names: list[str] = []
    if chromadb is None:
        logger.warning("[TEST] chromadb 미설치 또는 임포트 실패. 컬렉션 목록을 건너뜁니다.")
        return names
    try:
        os.makedirs(CHROMA_PERSIST, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_PERSIST)
        cols = client.list_collections()
        names = [getattr(c, "name", "") for c in cols]
    except Exception as e:
        logger.exception("[TEST] 컬렉션 목록 조회 실패: %s", e)
    return names


def print_doc(i, d, full_opt) -> None:
    """검색 결과 1건을 보기 좋게 출력."""
    m = d.metadata or {}
    snippet = (d.page_content or "").strip().replace("\n", " ")
    if len(snippet) > 240:
        snippet = snippet[:240] + f"...(+{len(d.page_content)-240})"
    print("-"*60)
    print(f"[{i:02d}] {m.get('file_name','?')} p.{m.get('page','?')}")
    print(f"     - law_name={m.get('law_name','')}, type={m.get('doc_type','')}, "
          f"no={m.get('law_number','')}, date={m.get('effective_date','')}")
    if full_opt:
        print(f"     - content: {d.page_content}\n")
    else:
        print(f"     - snippet: {snippet}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG 검색 테스트 스크립트")
    parser.add_argument("-q", "--query", required=True, help="검색 질의문")
    parser.add_argument("-c", "--category", default=None,
                        help="카테고리(한글 또는 영문: 헌법/민법/... 또는 constitution/civil/...)")
    parser.add_argument("-k", "--topk", type=int, default=8, help="상위 반환 개수 (기본 8)")
    parser.add_argument("--show-collections", action="store_true", help="컬렉션 목록 출력")
    parser.add_argument("--verbose", action="store_true", help="디버그 로그 활성화")
    args = parser.parse_args()

    full_opt = False
    if args.verbose:
        os.environ["LOGLEVEL"] = "DEBUG"
        logger.setLevel("DEBUG")
        full_opt = True

    print("=== RAG Retrieve Test ===")
    print(f"- CHROMA_PERSIST : {CHROMA_PERSIST}")
    print(f"- Query          : {args.query}")
    print(f"- Category (in)  : {args.category}")

    # 0) 컬렉션 목록(옵션)
    if args.show_collections:
        names = list_collections()
        print(f"- Collections    : {names}")

    # 1) 카테고리 처리(영문 키로 정규화 + 컬렉션명 안내)
    cat_eng: Optional[str] = None
    coll_name: Optional[str] = None
    if args.category:
        cat_eng = to_eng(args.category)
        coll_name = collection_name_for(args.category)
        print(f"- Category (eng) : {cat_eng}")
        print(f"- Collection     : {coll_name}")

        # 존재 확인 힌트
        names = list_collections()
        if names:
            exists = coll_name in names
            print(f"- Collection exists? {exists}")
            if not exists:
                print("  ! 컬렉션이 보이지 않습니다. 다음을 확인하세요:")
                print("    1) data/ 폴더의 PDF 인덱싱을 수행했는지 (앱 사이드바의 'PDF 인덱싱')")
                print("    2) CATEGORY와 컬렉션 규칙이 일치하는지 (law_docs_<english>)")
                print("    3) 임베딩/환경변수(Azure OpenAI)가 정상 설정되었는지")

    print("\n--- Retrieve ---")
    try:
        docs = rag_retrieve(query=args.query, k=args.topk, category=cat_eng)
        print(f"Total hits: {len(docs)}")
        if not docs:
            print("! 결과가 없습니다. 점검 사항:")
            print("  - 올바른 카테고리/컬렉션인지 확인")
            print("  - 인덱싱이 완료되었는지 확인 (파일 개수/진행 로그)")
            print("  - Azure OpenAI Embeddings 환경변수 확인")
            return

        for i, d in enumerate(docs, 1):
            print_doc(i, d, full_opt)

    except Exception as e:
        print("\n[ERROR] retrieve() 호출 중 예외 발생")
        print("  - 원인(추정): 임베딩 설정 오류, 컬렉션 미존재, Chroma 파일 권한 문제 등")
        print(f"  - 예외: {type(e).__name__}: {e}")
        logger.exception("[TEST] retrieve 실패")


if __name__ == "__main__":
    main()
