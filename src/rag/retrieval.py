# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import List, Optional, Tuple

import chromadb
from langchain_chroma import Chroma

from src.config import CHROMA_PERSIST, get_embeddings
from src.utils.logging_utils import setup_logging, trace
from src.agents.constants import collection_name_for  # ✅ 영문 컬렉션명 생성

logger = setup_logging()

_client: chromadb.PersistentClient | None = None

@trace()
def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_PERSIST, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST)
        logger.debug("[RET] PersistentClient init: %s", CHROMA_PERSIST)
    return _client

@trace()
def _get_vs(coll_name: str) -> Chroma:
    return Chroma(client=_get_client(), collection_name=coll_name, embedding_function=get_embeddings())

@trace()
def _list_collections() -> List[str]:
    cols = _get_client().list_collections()
    names = [getattr(c, "name", "") for c in cols]
    logger.debug("[RET] collections=%s", names)
    return names

@trace()
def retrieve(query: str, k: int = 3, category: Optional[str] = None) -> List:
    """
    category: 영문 키 권장('civil' 등). 한글 라벨을 넘겨도 constants.to_eng()을 통해 매핑됨.
    - 지정 시: law_docs_<eng> 딱 1개 컬렉션에서 검색 (단순/명확)
    - 미지정 시: 모든 law_docs_* 병합 검색
    """
    if not query:
        return []

    if category:
        coll = collection_name_for(category)  # ex) "civil" or "민법" -> "law_docs_civil"
        if coll not in _list_collections():
            logger.warning("[RET] 컬렉션 없음: %s", coll)
            return []
        vs = _get_vs(coll)
        docs = vs.similarity_search(query, k=k)
        logger.debug("[RET] hits=%d coll=%s", len(docs), coll)
        for d in docs:
            logger.debug("-"*50)
            logger.debug(d)

        return docs

    # all collections
    colls = [n for n in _list_collections() if n.startswith("law_docs_")]
    if not colls:
        return []
    per = max(1, k // max(1, len(colls)) + 1)
    hits: List[Tuple] = []
    for coll in colls:
        try:
            vs = _get_vs(coll)
            for doc, score in vs.similarity_search_with_relevance_scores(query, k=per):
                hits.append((doc, score))
        except Exception as e:
            logger.warning("[RET] skip coll=%s: %s", coll, e)
    hits.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in hits[:k]]
