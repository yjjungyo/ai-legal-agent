# -*- coding: utf-8 -*-
"""
RAG 인덱싱 모듈 (src/rag/indexing.py)
------------------------------------
- data/ 아래 PDF를 재귀 탐색 → PyMuPDFLoader 로드 → RecursiveCharacterTextSplitter로 청크
- 파일명에서 LLM으로 메타데이터(법령명/유형/번호/시행일) 추출(실패 시 정규식 폴백)
- '법령명'을 카테고리로 보고, 컬렉션명 law_docs_{category_slug} 으로 분리 저장
- chromadb.PersistentClient(path=CHROMA_PERSIST) + langchain_chroma.Chroma (자동 영속)
- 풍부한 주석과 DEBUG 로그, @trace 데코레이터로 실행 흐름 관찰 가능

사용 전제:
- Azure OpenAI 설정과 팩토리: src.config.get_embeddings, get_temp_llm
- 로깅 유틸: src.utils.logging_utils.setup_logging, trace
"""

from __future__ import annotations
from dataclasses import dataclass
import os
import re
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from src.config import CHROMA_PERSIST, get_embeddings, get_temp_llm
from src.utils.logging_utils import setup_logging, trace

from src.agents.constants import CATEGORIES, collection_name_for
from typing import Tuple

import fitz  # PyMuPDF
from langchain_core.documents import Document  # 수동 생성용


# 전역 로거 초기화 
logger = setup_logging()


# =========================
# 공통 유틸
# =========================
def _slug(s: str) -> str:
    """컬렉션명/키로 쓰기 안전한 슬러그 문자열."""
    return re.sub(r"[^\w\-가-힣]+", "_", (s or "").strip()) or "uncategorized"

def _fmt_date(date_str: str) -> str:
    """다양한 형식의 날짜를 YYYY-MM-DD 로 정규화."""
    if not date_str:
        return ""
    s = re.sub(r"[^\d]", "", date_str)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else date_str

def _collection_name(category_slug: str) -> str:
    """법령명 슬러그로 컬렉션명 생성."""
    return f"law_docs_{category_slug}"

def _notify(cb, msg: str) -> None:
    """on_progress 콜백을 안전하게 호출(콜백 예외는 경고만 찍고 무시)."""
    if cb is None:
        return
    try:
        cb(msg)
    except Exception as e:
        logger.warning("[INDEX] on_progress callback error: %s", e)
        
# =========================
# PDF 텍스트 추출 (페이지별)
# =========================
@trace()
def _extract_pages_text(pdf_path: str) -> List[str]:
    """
    PyMuPDF로 페이지별 텍스트를 추출 (순수 텍스트).
    - 반환: 페이지 텍스트 리스트 (인덱스0=1페이지)
    """
    texts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for p in doc:
            # 레이아웃 영향 줄이려면 "text" 모드가 가장 안정적
            t = p.get_text("text") or ""
            texts.append(t)
    logger.debug("[IDX] pages=%d extracted from %s", len(texts), pdf_path)
    return texts


# =========================
# 한국 법령 구조 정규식 (편/장/절/조문/부칙)
# =========================
_PART_RE    = re.compile(r"^\s*제\s*(\d+)\s*편\b")
_CHAPTER_RE = re.compile(r"^\s*제\s*(\d+)\s*장\b")
_SECTION_RE = re.compile(r"^\s*제\s*(\d+)\s*절\b")
_ADDENDA_RE = re.compile(r"^\s*부\s*칙\b")
# 조문: 제12조(제목) / 제12조의2(제목)
_ARTICLE_RE = re.compile(r"^\s*제\s*(\d+)\s*(?:조의\s*(\d+)|조)\s*(?:\(([^)]+)\))?")

def _norm_line(s: str) -> str:
    """라인 단위 whitespace 정규화."""
    return re.sub(r"\s+", " ", (s or "").strip())

@dataclass
class _Seg:
    part: str = ""
    chapter: str = ""
    section: str = ""
    article_no: str = ""       # "제12조", "제12조의2"
    article_title: str = ""    # "(목적)"의 제목 부분
    title_line: str = ""       # 라인 전체 (제12조(목적))
    start_page: int = 1        # 1-based
    end_page: int = 1          # 1-based
    lines: List[str] = None    # 본문 누적

    def to_text(self) -> str:
        body = "\n".join(self.lines or [])
        return (self.title_line + "\n" + body).strip()


# =========================
# 법령 구조 파서: 페이지 텍스트 → 조문 단위 세그먼트
# =========================
@trace()
def _parse_korean_law_structure(pages: List[str]) -> List[_Seg]:
    """
    페이지별 텍스트를 받아, (편/장/절/조문) 단위로 세그먼트를 생성.
    - 조문을 경계로 세그먼트 쪼갬
    - 세그먼트는 시작/끝 페이지(1-based)를 보유
    """
    segs: List[_Seg] = []
    cur_part, cur_chapter, cur_section = "", "", ""
    cur: Optional[_Seg] = None
    last_seen_page = 1

    for idx, page_text in enumerate(pages):
        page_no = idx + 1  # 1-based
        lines = page_text.splitlines()

        for raw in lines:
            line = _norm_line(raw)
            if not line:
                continue

            # 상위 구조 갱신
            if _ADDENDA_RE.match(line):
                cur_part, cur_chapter, cur_section = "부칙", "", ""
            else:
                m = _PART_RE.match(line)
                if m:
                    cur_part = f"제{m.group(1)}편"
                m = _CHAPTER_RE.match(line)
                if m:
                    cur_chapter = f"제{m.group(1)}장"
                m = _SECTION_RE.match(line)
                if m:
                    cur_section = f"제{m.group(1)}절"

            # 조문 경계?
            ma = _ARTICLE_RE.match(line)
            if ma:
                # 이전 조문 flush
                if cur is not None:
                    cur.end_page = last_seen_page
                    segs.append(cur)

                num = ma.group(1) or ""
                sub = ma.group(2)  # '의2' 등
                title = ma.group(3) or ""
                if sub:
                    article_no = f"제{num}조의{sub}"
                else:
                    article_no = f"제{num}조"

                cur = _Seg(
                    part=cur_part,
                    chapter=cur_chapter,
                    section=cur_section,
                    article_no=article_no,
                    article_title=title,
                    title_line=line,
                    start_page=page_no,
                    end_page=page_no,
                    lines=[],
                )
            else:
                # 본문 누적
                if cur is not None:
                    cur.lines.append(line)

            last_seen_page = page_no

    # 마지막 세그 flush
    if cur is not None:
        cur.end_page = last_seen_page
        segs.append(cur)

    logger.info("[IDX] parsed segments: %d (articles)", len(segs))
    return segs


# =========================
# 세그먼트 → Document 변환 (+ 긴 조문은 보조 분할)
# =========================
@trace()
def _segments_to_documents(
    segs: List[_Seg],
    base_meta: Dict[str, str],
    splitter: Optional[RecursiveCharacterTextSplitter] = None,
    long_threshold: int = 2400,
) -> List[Document]:
    """
    세그먼트를 LangChain Document로 변환.
    - 기본은 '조문 단위'로 1문서
    - 매우 긴 조문은 splitter로 2차 분할(메타에 subchunk 및 page_range 유지)
    """
    out: List[Document] = []
    for s in segs:
        text = s.to_text()
        meta = {
            **base_meta,
            "unit": "article",
            "part": s.part,
            "chapter": s.chapter,
            "section": s.section,
            "article_no": s.article_no,
            "article_title": s.article_title,
            "page": s.start_page,          # 출처 표기를 위해 시작페이지를 기본 page로
            "page_end": s.end_page,        # 범위가 있을 수 있음
            "page_range": f"{s.start_page}-{s.end_page}" if s.end_page != s.start_page else f"{s.start_page}",
        }

        if splitter and len(text) > long_threshold:
            # 긴 조문 → 보조 분할 (문맥 보존을 위해 overlap 유지)
            chunks = splitter.split_text(text)
            for i, ch in enumerate(chunks, 1):
                m = {**meta, "subchunk": i, "subchunk_total": len(chunks)}
                out.append(Document(page_content=ch, metadata=m))
        else:
            out.append(Document(page_content=text, metadata=meta))
    logger.debug("[IDX] built %d Documents from segments", len(out))
    return out

# =========================
# LLM 파일명 메타추출
# =========================
LLM_META_PROMPT = PromptTemplate.from_template(
    """다음 한국 법령 PDF 파일명에서 메타데이터를 추출해 JSON 한 줄로만 출력하세요.
필드: law_name, doc_type, law_number, effective_date(YYYY-MM-DD)
모르는 값은 "" (빈 문자열).
코드블록 없이 JSON만.

파일명: {filename}"""
)


# =========================
# (신규) 법령명 → 카테고리 단일 분류 프롬프트
# =========================
LAW_CATEGORY_PROMPT = PromptTemplate.from_template(
    # ⚠️ {categories}, {law_name} 만 단일 중괄호 사용
    # ✅ JSON 예시는 {{ ... }} 로 이스케이프!
    """당신은 한국 법령의 **분야 카테고리**를 판정하는 분류기입니다.
가능한 카테고리(이 집합에서만 선택): {categories}

[출력 형식]
- **JSON 한 줄만** 출력: {{"category":"<카테고리>","reason":"<간단한 근거>"}}
- 추가 텍스트/코드블록/개행 금지

[결정 규칙]
1) 반드시 **정확히 1개**를 선택합니다.
2) 한국 법학 교육/실무의 일반적 분류를 따릅니다.
   - 헌법: 헌법 자체·기본권·통치구조·헌법재판
   - 민법: 계약·물권·불법행위·가족·상속 및 **민사 특별법**
   - 형법: 범죄와 형벌 및 **형사 특별법**
   - 상법: 회사·어음/수표·보험·금융/증권 등 **상사 특별법**
   - 노동: 근로관계·노동3권·산안·임금·퇴직금
   - 행정: 국가/지자체의 행정작용·인허가·절차·공공서비스/사회보장 (**국민건강보험법** 등)
   - 조세: 국세/지방세의 과세·징수·불복 절차
   - 국제: 조약·관습국제법·국제기구·국제재판, 국제사법
3) 불명확할 경우 가장 일반적으로 분류되는 상위 카테고리를 고르세요.

[입력]
법령명: {law_name}

# FEW-SHOT
입력: "국민건강보험법"
출력: {{"category":"행정","reason":"사회보장/공공보험 제도 운영을 규율하는 공법 영역"}}

입력: "소득세법"
출력: {{"category":"조세","reason":"국세 과세요건·과세표준·징수 규율"}}

입력: "주택임대차보호법"
출력: {{"category":"민법","reason":"민사(임대차) 특별법"}}

입력: "{law_name}"
출력:
"""
)

def _parse_json_line(text: str) -> dict:
    """모델 출력에서 첫 {..} 블록만 잘라 JSON 파싱. 실패 시 {}."""
    if not text:
        return {}
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return {}
    try:
        return json.loads(text[i:j+1])
    except Exception:
        return {}

@trace()
def classify_law_category_with_llm(law_name: str) -> str:
    """
    LLM을 사용해 법령명을 8개 카테고리 중 1개로 분류.
    - 입력: law_name (예: '국민건강보험법')
    - 출력: 한글 카테고리 라벨 (예: '행정')
    - 실패/예외시 보수적 폴백: '민법'
    """
    try:
        llm = get_temp_llm(0.0)  # 결정적(classification) 작업이라 temperature 0
        prompt = LAW_CATEGORY_PROMPT.format(
            categories=", ".join(CATEGORIES),
            law_name=law_name or "(미상)"
        )
        logger.debug("[IDX] category LLM prompt len=%d, law_name=%s", len(prompt), law_name)
        res = llm.invoke(prompt)
        text = (getattr(res, "content", str(res)) or "").strip()
        data = _parse_json_line(text)
        cat = (data.get("category") or "").strip()
        if cat in CATEGORIES:
            logger.info("[IDX] '%s' -> category='%s' (LLM)", law_name, cat)
            return cat
        logger.warning("[IDX] 분류값이 카테고리 집합에 없음: raw=%s", cat)
    except Exception as e:
        logger.exception("[IDX] LLM 카테고리 분류 실패: %s", e)

    # ✅ 폴백: 민법
    fallback = "민법"
    logger.info("[IDX] fallback category '%s' for law_name='%s'", fallback, law_name)
    return fallback



# 규칙형 예: 민법(법률)(제20432호)(20250131).pdf
_FILE_RE = re.compile(
    r"""^\s*(?P<law>[^()]+)\((?P<type>[^)]+)\)\((?P<number>[^)]+)\)\((?P<date>\d{8})\)\.pdf\s*$""",
    re.X,
)

def _parse_with_regex(name: str) -> Dict[str, str]:
    """
    정규식 폴백 파서:
    - 예측 가능한 포맷이면 정확히 추출
    - 그렇지 않으면 괄호 단위로 느슨하게 추출
    """
    m = _FILE_RE.match(name)
    if m:
        return {
            "law_name": m.group("law").strip(),
            "doc_type": m.group("type").strip(),
            "law_number": m.group("number").strip(),
            "effective_date": _fmt_date(m.group("date").strip()),
        }
    base = Path(name).stem
    parts = re.findall(r"\(([^)]+)\)", base)
    law = base.split("(")[0].strip()
    return {
        "law_name": law or "",
        "doc_type": (parts[0].strip() if len(parts) > 0 else ""),
        "law_number": (parts[1].strip() if len(parts) > 1 else ""),
        "effective_date": _fmt_date(parts[2]) if len(parts) > 2 else "",
    }

@trace()
def classify_filename_with_llm(filename: str) -> Dict[str, str]:
    """
    파일명에서 LLM으로 메타데이터를 추출(JSON)하고, 실패 시 정규식으로 폴백.
    반환: {'law_name','doc_type','law_number','effective_date'}
    """
    try:
        llm = get_temp_llm(0.0)  # 결정적 추출
        raw = llm.invoke(LLM_META_PROMPT.format(filename=filename))
        text = (getattr(raw, "content", str(raw)) or "").strip()

        # 코드블록/주석 제거: 첫 '{' ~ 마지막 '}' 범위만 파싱
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1:
            text = text[i : j + 1]

        data = json.loads(text)
        out = {
            "law_name": (data.get("law_name") or "").strip(),
            "doc_type": (data.get("doc_type") or "").strip(),
            "law_number": (data.get("law_number") or "").strip(),
            "effective_date": _fmt_date(data.get("effective_date") or ""),
        }
        logger.debug("[IDX/LLM] %s -> %s", filename, out)
        return out
    except Exception as e:
        logger.warning("[IDX/LLM] 파싱 실패 -> 정규식 폴백: %s (%s)", filename, e)
        out = _parse_with_regex(filename)
        logger.debug("[IDX/REGEX] %s -> %s", filename, out)
        return out


# =========================
# Chroma 클라이언트/스토어
# =========================
_client: Optional[chromadb.PersistentClient] = None

@trace()
def _get_client() -> chromadb.PersistentClient:
    """
    한 개의 PersistentClient를 공유하여 자동 영속.
    langchain_chroma.Chroma는 .persist()가 없으므로 client 기반으로 사용.
    """
    global _client
    if _client is None:
        os.makedirs(CHROMA_PERSIST, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST)
        logger.debug("[IDX] PersistentClient init: %s", CHROMA_PERSIST)
    return _client

@trace()
def _get_vs(coll_name: str) -> Chroma:
    """컬렉션명으로 LangChain Chroma VectorStore 생성."""
    return Chroma(client=_get_client(), collection_name=coll_name, embedding_function=get_embeddings())


# =========================
# 파일 스캔/인덱싱
# =========================
@trace()
def _iter_pdf_paths(data_dir: str) -> List[Path]:
    """data_dir 하위의 모든 PDF 경로를 재귀 수집(대/소문자 무시)."""
    base = Path(data_dir)
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]

def _to_one_based_page(meta: dict) -> None:
    """
    PyMuPDFLoader가 넣는 page 메타(보통 0-based)를 1-based로 보정.
    - 정수/숫자형 문자열/float 모두 처리
    - 잘못된 값/없음은 건너뜀
    """
    try:
        pg = meta.get("page", None)
        if pg is None:
            return
        # "12", 12, 12.0 등 처리
        if isinstance(pg, (int, float)) or (isinstance(pg, str) and pg.strip().isdigit()):
            new_pg = int(float(pg)) + 1
            meta["page"] = new_pg
    except Exception:
        # 보정 실패해도 인덱싱은 진행
        pass



# =========================
# 관리/인벤토리 유틸
# =========================
@trace()
def list_collections() -> List[str]:
    """현재 Chroma에 존재하는 모든 컬렉션 이름을 반환."""
    try:
        client = _get_client()
        cols = client.list_collections()
        names = [getattr(c, "name", "") for c in cols]
        logger.debug("[IDX] collections=%s", names)
        return names
    except Exception as e:
        logger.exception("[IDX] list_collections 실패: %s", e)
        return []


@trace()
def list_index_inventory(max_rows_per_collection: int = 500) -> List[Dict[str, str]]:
    """
    모든 컬렉션에서 'file_name' 메타를 모아 (파일명/컬렉션/청크수) 목록을 만든다.
    - 반환: [{'file_name': str, 'collection': str, 'chunks': int}]
    """
    rows: List[Dict[str, str]] = []
    try:
        client = _get_client()
        for name in list_collections():
            try:
                col = client.get_collection(name=name)
                total = col.count()
                logger.debug("[IDX] peek %s total=%s", name, total)
                # batch get (offset/limit 지원)
                step, offset = 100, 0
                counter: Dict[str, int] = {}
                while offset < total and offset < max_rows_per_collection:
                    got = col.get(
                        include=["metadatas"],
                        limit=min(step, max_rows_per_collection - offset),
                        offset=offset,
                    )
                    mds = got.get("metadatas", []) or []
                    for md in mds:
                        fn = (md or {}).get("file_name", None)
                        if fn:
                            counter[fn] = counter.get(fn, 0) + 1
                    offset += step
                # rows 생성
                for fn, cnt in sorted(counter.items(), key=lambda x: (x[0], -x[1])):
                    rows.append({"file_name": fn, "collection": name, "chunks": str(cnt)})
            except Exception as ce:
                logger.exception("[IDX] 컬렉션 인벤토리 실패: %s (%s)", name, ce)
                continue
    except Exception as e:
        logger.exception("[IDX] list_index_inventory 실패: %s", e)
    return rows


@trace()
def delete_by_filename(collection_name: str, file_name: str) -> Tuple[int, int]:
    """
    특정 컬렉션에서 특정 파일명의 모든 청크를 삭제.
    - 반환: (before_count, after_count)
    """
    client = _get_client()
    col = client.get_collection(name=collection_name)
    try:
        before = col.count()
        col.delete(where={"file_name": file_name})
        after = col.count()
        logger.info("[IDX] deleted file='%s' in coll='%s' (before=%d -> after=%d)",
                    file_name, collection_name, before, after)
        return before, after
    except Exception as e:
        logger.exception("[IDX] delete_by_filename 실패: file=%s coll=%s err=%s",
                         file_name, collection_name, e)
        return (-1, -1)


# =========================
# 인덱싱(재인덱싱 모드 지원)
# =========================
@trace()
def index_pdfs_by_collection(
    data_dir: str = "./data",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    on_progress: Optional[Callable[[str], None]] = None,
    reindex: bool = False,
    target_files: Optional[List[str]] = None,
) -> int:
    """
    PDF를 인덱싱해서 law_docs_<영문카테고리> 컬렉션으로 저장.
    - reindex=True: 색인 전 동일 (collection, file_name) 데이터 삭제 후 재색인
    - target_files: 지정된 파일명만 처리 (None이면 data_dir 전체)
    - 반환: 총 적재된 청크 수
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_pdfs = _iter_pdf_paths(data_dir)
    pdfs = [p for p in all_pdfs if (not target_files) or (p.name in set(target_files))]

    msg = f"[INDEX] PDFs={len(pdfs)} dir={data_dir} chunk={chunk_size}/{chunk_overlap} reindex={reindex}"
    logger.info(msg); on_progress and on_progress(msg)

    if not pdfs:
        on_progress and on_progress("⚠️ data/ 폴더에 색인할 PDF가 없습니다.")
        logger.warning("[INDEX] no pdfs found")
        return 0

    total_chunks = 0
    per_coll: Dict[str, int] = {}

    for i, path in enumerate(pdfs, 1):
        fname = path.name

        # 1) 파일명 → 메타
        meta = classify_filename_with_llm(fname)
        law_name = meta.get("law_name") or "미상"
        # 2) 법령명 → 카테고리
        category_ko = classify_law_category_with_llm(law_name)
        # 3) 컬렉션명
        coll = collection_name_for(category_ko)
        logger.info("[IDX] file='%s' -> law='%s' cat='%s' coll='%s'",
                    fname, law_name, category_ko, coll)

        try:
            on_progress and on_progress(f"({i}/{len(pdfs)}) 준비: {fname} → {coll}")

            # ✅ 재인덱싱: 기존 동일 파일명 삭제
            if reindex:
                delete_by_filename(coll, fname)
                on_progress and on_progress(f"   - 기존 청크 삭제 완료: {fname} in {coll}")

            # 4) PDF 로드  ← 기존: PyMuPDFLoader(...).load()
            #    변경: 페이지 텍스트 추출 + 구조 파서 + 조문 단위 문서 생성
            pages = _extract_pages_text(str(path))
            segs = _parse_korean_law_structure(pages)

            # 5) 메타 보강 (파일 공통) + 페이지 1베이스는 이미 세그에서 보정됨
            base_meta = {
                "source_path": str(path),
                "file_name": fname,
                "law_name": law_name,
                "doc_type": meta.get("doc_type", ""),
                "law_number": meta.get("law_number", ""),
                "effective_date": meta.get("effective_date", ""),
                "category":  category_ko,
            }

            # 6) 세그먼트 → Document (긴 조문은 보조 분할)
            docs = _segments_to_documents(
                segs,
                base_meta=base_meta,
                splitter=splitter,           # 기존 splitter 재사용 (chunk_size/overlap 반영)
                long_threshold=2400,         # 필요 시 조절
            )

            if not docs:
                logger.warning("[INDEX] %s → 0 segments", fname)
                _notify(on_progress, f"⚠️ {fname} → 0 segments")
                continue

            # 7) 저장 (자동 영속)
            vs = _get_vs(coll)
            vs.add_documents(docs)

            n = len(docs)
            total_chunks += n
            per_coll[coll] = per_coll.get(coll, 0) + n
            logger.debug("[INDEX] stored: file=%s docs=%d coll=%s", fname, n, coll)
            _notify(on_progress, f"✅ {fname} → {n} docs (coll={coll})")

        except Exception as e:
            logger.exception("[INDEX] 실패: %s (%s)", fname, e)
            on_progress and on_progress(f"❌ 실패: {fname} ({e})")

    detail = ", ".join(f"{k}:{v}" for k, v in per_coll.items()) if per_coll else "-"
    done = f"[INDEX] 완료: 총 {total_chunks} chunks, 컬렉션 분포= {{{detail}}}"
    logger.info(done); on_progress and on_progress(done)
    return total_chunks
