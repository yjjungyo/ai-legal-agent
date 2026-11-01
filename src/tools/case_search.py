# -*- coding: utf-8 -*-


"""
대법원 판례 검색 툴
-------------------
- 1차: Tavily API로 'site:glaw.scourt.go.kr' 제한 검색 → URL 후보 수집
- 2차: 각 URL을 정중하게 크롤 → 사건명/사건번호/선고일자/법원/요지 파싱
- 보조: 국가법령정보센터(law.go.kr)의 판례 상세도 일부 파싱 지원
- 출력: 표준화된 CaseResult 목록 + (옵션) LangChain Document 변환

설계 원칙
- 상세 주석 + 풍부한 DEBUG 로그 (@trace 데코레이터)
- 예외는 상위로 전파하되, URL 단위 실패는 개별 경고 후 스킵(견고성)
- 민감한 본문은 길이만 로그에 남김
"""

from __future__ import annotations

import os
import re
import time
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import httpx
from bs4 import BeautifulSoup

# LangChain 타입(문서 변환에 사용; 프로젝트에 이미 존재)
from langchain.schema import Document

# 로깅 유틸 (이미 추가하신 파일 사용)
from src.utils.logging_utils import setup_logging, trace

logger = setup_logging()


# =========================
# 환경 설정 로딩
# =========================
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))
HTTP_UA = os.getenv("HTTP_USER_AGENT", "AI-Legal-Agent/1.0")
HTTP_MAX_REDIRECTS = int(os.getenv("HTTP_MAX_REDIRECTS", "5"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()


# =========================
# 데이터 모델
# =========================
@dataclass
class CaseResult:
    """표준화된 판례 검색 결과"""
    title: str                  # 사건명
    case_no: str                # 사건번호 (예: 2016다12345)
    court: str                  # 법원명 (예: 대법원)
    date: str                   # 선고일자 (YYYY-MM-DD 또는 원문 표기)
    url: str                    # 원문 URL
    summary: str = ""           # 요지(가능하면)
    origin: str = ""            # 'glaw.scourt.go.kr' / 'law.go.kr' 등
    raw: Dict = None            # 원본 파싱 결과(필요시 디버깅용)

# =========================
# HTTP 클라이언트
# =========================
def _new_client() -> httpx.Client:
    """
    httpx 동기 클라이언트 생성.
    - 표준 UA/타임아웃/리다이렉트 제한
    - HTTP/2 사용 시 서버 조건에 따라 자동 업그레이드
    """
    headers = {
        "User-Agent": HTTP_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Cache-Control": "no-cache",
    }
    return httpx.Client(
        headers=headers,
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
        max_redirects=HTTP_MAX_REDIRECTS,
        http2=True,
    )


# =========================
# Tavily 검색
# =========================
@trace()
def tavily_search_supreme(query: str, max_results: int = 5) -> List[Tuple[str, str]]:
    """
    Tavily API로 대법원 종합법률정보 도메인에 한해 검색.
    반환: [(title, url), ...]
    """
    if not TAVILY_API_KEY:
        logger.warning("[TAVILY] API 키가 없어 Tavily 검색을 건너뜁니다.")
        return []

    try:
        from tavily import TavilyClient  # 늦은 임포트(미설치 시 다른 기능은 동작)
    except Exception as e:
        logger.warning(f"[TAVILY] 패키지 임포트 실패: {e}")
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)

    # 검색 쿼리: site 제한 + 사용자의 질의
    q = f"site:glaw.scourt.go.kr 판례 {query}"
    logger.debug(f"[TAVILY] query='{q}' max_results={max_results}")

    # Tavily 응답 구조: {'results': [{'title':..., 'url':..., 'content':...}, ...]}
    res = client.search(
        query=q,
        max_results=max_results,
        include_domains=["glaw.scourt.go.kr"],  # 도메인 제한
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
    )
    out: List[Tuple[str, str]] = []
    for r in res.get("results", []):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        if not url:
            continue
        out.append((title, url))
        logger.debug(f"[TAVILY] hit: title='{title}' url={url}")

    return out


# =========================
# 페이지 파서 (도메인별)
# =========================
@trace()
def _parse_scourt(html: str, url: str) -> CaseResult | None:
    """
    대법원 종합법률정보(glaw.scourt.go.kr) 판례 상세 페이지 추정 HTML 파싱.
    - 사이트 구조가 수시로 변할 수 있어, 넓은 휴리스틱을 사용
    - 가능한 항목: 사건명(title), 사건번호(case_no), 선고일자(date), 법원(court), 요지(summary)
    """
    soup = BeautifulSoup(html, "lxml")

    # 1) 제목 후보: h1, h2, title 등
    title = ""
    for sel in ["h1", "h2", "div#content h1", "div#content h2", "title"]:
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            title = node.get_text(strip=True)
            break

    # 2) 키워드 테이블/정의목록 후보에서 사건번호/선고일자/법원 추출
    text = soup.get_text("\n", strip=True)
    # 사건번호 정규식(예: 2022도12345, 2016다12345 등)
    m_case = re.search(r"(?:사건번호|사건|사건번호\s*:?)\s*([0-9]{2,4}[가-힣][0-9\-]+)", text)
    case_no = m_case.group(1) if m_case else ""

    # 선고일자
    m_date = re.search(r"(?:선고일자|선고)\s*:?\s*([0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2})", text)
    date = m_date.group(1).replace(".", "-") if m_date else ""

    # 법원
    m_court = re.search(r"(대법원|특허법원|고등법원|지방법원)", text)
    court = m_court.group(1) if m_court else "대법원"

    # 요지/판시사항 후보
    summary = ""
    for key in ["요지", "판시사항", "주문", "이유"]:
        # 문단 단위로 넉넉히 뽑되 너무 길면 자름
        m = re.search(rf"{key}\s*[:：]?\s*(.+?)(?:\n{1,2}[가-힣]{1,10}\s*[:：]|\Z)", text, re.S)
        if m:
            summary = m.group(1).strip()
            if len(summary) > 800:
                summary = summary[:800] + f"...(+{len(summary)-800})"
            break

    title = title or case_no or "판례(제목 미상)"
    return CaseResult(
        title=title,
        case_no=case_no,
        court=court,
        date=date,
        url=url,
        summary=summary,
        origin="glaw.scourt.go.kr",
        raw={"source": "scourt", "extracted_len": len(text)},
    )


@trace()
def _parse_law_go_kr(html: str, url: str) -> CaseResult | None:
    """
    국가법령정보센터(law.go.kr)의 판례 상세 페이지 파싱.
    - 흔한 URL: https://www.law.go.kr/precInfoP.do?precSeq=...
    - 테이블/정의목록에서 사건명/사건번호/선고일자/법원 추출
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)

    # 사건명
    title = ""
    for th in soup.select("th"):
        t = th.get_text(strip=True)
        if "사건명" in t:
            td = th.find_next("td")
            if td:
                title = td.get_text(strip=True)
                break
    if not title:
        # 보조: 페이지 제목
        node = soup.select_one("h1, h2, title")
        title = node.get_text(strip=True) if node else ""

    # 사건번호
    case_no = ""
    for th in soup.select("th"):
        if "사건번호" in th.get_text(strip=True):
            td = th.find_next("td")
            if td:
                case_no = td.get_text(strip=True)
                break

    # 선고일자
    date = ""
    for th in soup.select("th"):
        if "선고일자" in th.get_text(strip=True):
            td = th.find_next("td")
            if td:
                date = td.get_text(strip=True).replace(".", "-")
                break

    # 법원
    court = "대법원"
    for th in soup.select("th"):
        if "법원" in th.get_text(strip=True):
            td = th.find_next("td")
            if td:
                court = td.get_text(strip=True)
                break

    # 요지/판시사항
    summary = ""
    for th in soup.select("th"):
        t = th.get_text(strip=True)
        if "판시사항" in t or "판결요지" in t or "요지" in t:
            td = th.find_next("td")
            if td:
                summary = td.get_text(" ", strip=True)
                if len(summary) > 800:
                    summary = summary[:800] + f"...(+{len(summary)-800})"
                break

    title = title or case_no or "판례(제목 미상)"
    return CaseResult(
        title=title,
        case_no=case_no,
        court=court,
        date=date,
        url=url,
        summary=summary,
        origin="law.go.kr",
        raw={"source": "law.go.kr", "extracted_len": len(text)},
    )


# =========================
# 페치 & 파싱
# =========================
@trace()
def _fetch_and_parse(url: str, client: httpx.Client) -> Optional[CaseResult]:
    """
    주어진 URL을 요청해 HTML을 받고, 도메인별 파서로 CaseResult 변환.
    파싱 실패 시 None 반환.
    """
    try:
        # 예의상 지연(서버 보호) – 필요 시 조절
        time.sleep(0.5)
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text
        dom = httpx.URL(url).host or ""

        if "glaw.scourt.go.kr" in dom:
            return _parse_scourt(html, url)
        if "law.go.kr" in dom:
            return _parse_law_go_kr(html, url)

        # 기타 도메인(검색 결과가 섞였을 때) – 최소한의 공통 파싱
        soup = BeautifulSoup(html, "lxml")
        title = soup.select_one("h1, h2, title")
        title_text = title.get_text(strip=True) if title else url
        body_text = soup.get_text("\n", strip=True)
        summary = body_text[:800] + f"...(+{len(body_text)-800})" if len(body_text) > 800 else body_text

        return CaseResult(
            title=title_text,
            case_no="",
            court="",
            date="",
            url=url,
            summary=summary,
            origin=dom,
            raw={"source": dom, "extracted_len": len(body_text)},
        )
    except Exception as e:
        logger.warning(f"[FETCH] 실패: {url} ({e})")
        return None


# =========================
# 공개 API
# =========================
@trace()
def search_cases(
    query: str,
    max_results: int = 5,
    also_law_go_kr: bool = True,
) -> List[CaseResult]:
    """
    판례 검색(대법원 중심).
    1) Tavily로 'site:glaw.scourt.go.kr' 검색 → URL 후보 수집
    2) (옵션) Tavily가 없거나 결과 부족 시 'law.go.kr 판례'도 검색
    3) 각 URL을 크롤링/파싱 → CaseResult 목록으로 반환
    """
    urls: List[str] = []
    results: List[CaseResult] = []

    # Tavily 우선
    hits = tavily_search_supreme(query, max_results=max_results)
    urls.extend([u for _, u in hits])

    # 보완: 필요 시 law.go.kr 검색(간단히 쿼리 변경)
    if also_law_go_kr:
        from_results = tavily_search_supreme(f"site:law.go.kr 판례 {query}", max_results=max_results)
        urls.extend([u for _, u in from_results])

    # 중복 제거
    seen = set()
    unique_urls = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        unique_urls.append(u)

    logger.debug(f"[SEARCH] 후보 URL {len(unique_urls)}건")

    # 크롤링 & 파싱
    with _new_client() as client:
        for u in unique_urls[:max_results]:
            cr = _fetch_and_parse(u, client)
            if cr:
                results.append(cr)

    logger.debug(f"[SEARCH] 파싱 성공 {len(results)}건")
    return results


@trace()
def cases_to_documents(cases: List[CaseResult]) -> List[Document]:
    """
    CaseResult 목록을 LangChain Document로 변환.
    - page_content: 제목 + 요지(가능하면)
    - metadata: 사건번호/법원/선고일자/URL/출처도메인
    """
    docs: List[Document] = []
    for c in cases:
        # 본문 구성(요지가 없으면 최소한 제목+URL 포함)
        body_lines = [c.title]
        if c.summary:
            body_lines.append("")
            body_lines.append(c.summary)
        else:
            body_lines.append("")
            body_lines.append(f"(요약 없음: {c.url})")

        content = "\n".join(body_lines)
        meta = {
            "title": c.title,
            "case_no": c.case_no,
            "court": c.court,
            "date": c.date,
            "url": c.url,
            "origin": c.origin,
        }
        docs.append(Document(page_content=content, metadata=meta))
    return docs
